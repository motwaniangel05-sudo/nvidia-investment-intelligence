"""Full dynamic pipeline: question -> plan -> parallel agents -> Red-Team
-> evidence aggregation -> local Qwen synthesis -> final answer."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait

from core import dynamic_orchestrator as do
from core.qwen_synthesizer import synthesize as qwen_synthesize

AGENT_TIMEOUT = 180  # seconds to wait for all agents


def _execute_parallel(plan, registry, timeout):
    response = do.DynamicResponse(plan=plan)
    jobs = []
    for task in plan.tasks:
        if task.agent == do.RED_TEAM:
            continue
        adapter = registry.get(task.agent)
        if adapter is None:
            response.runs.append(do.AgentRun(task.agent, task.agent, "failed", 0.0,
                                             "no adapter registered"))
            continue
        ctx = do.AgentContext(query=plan.query, company_ticker=plan.company_ticker,
                              intent=plan.intent, reasoning_task=task.reasoning_task,
                              analysis=plan.analysis)
        jobs.append((task, adapter, ctx))

    if jobs:
        pool = ThreadPoolExecutor(max_workers=len(jobs))
        futures = {pool.submit(do._run_agent, adapter, ctx): (task, adapter)
                   for task, adapter, ctx in jobs}
        done, _ = wait(list(futures), timeout=timeout)
        for fut, (task, adapter) in futures.items():
            if fut in done:
                try:
                    result, run = fut.result()
                except Exception as exc:
                    response.runs.append(do.AgentRun(task.agent, adapter.agent_name, "failed",
                                                     0.0, str(exc)[:200]))
                    continue
                response.agent_results[adapter.agent_name] = result
                response.standard_results[adapter.agent_name] = do.normalize_result(result)
                response.runs.append(run)
            else:
                response.runs.append(do.AgentRun(task.agent, adapter.agent_name, "failed",
                                                 float(timeout), f"timed out after {timeout}s"))
        pool.shutdown(wait=False, cancel_futures=True)

    # Red-Team needs the outputs of the other agents, so it runs after them.
    if do.RED_TEAM in plan.agents and response.agent_results:
        do._run_red_team(response)
    return response


def run(query, company_ticker=None, timeout=AGENT_TIMEOUT, registry=None,
        synthesizer=qwen_synthesize, config=None):
    """answer = run(user_query) -> dict with the final answer and everything behind it."""
    started = time.time()
    errors = []
    registry = registry if registry is not None else do.AGENT_REGISTRY

    plan = do.plan_query(query, config=config, include_verification=True)
    if company_ticker:
        plan.company_ticker = company_ticker.strip().upper()

    t0 = time.time()
    response = _execute_parallel(plan, registry, timeout)
    agents_seconds = round(time.time() - t0, 2)
    for r in response.runs:
        if r.status == "failed":
            errors.append(f"{r.agent}: {r.error}")

    context = None
    try:
        context = do.aggregate_evidence(
            plan.query, response.standard_results.values(), intent=plan.intent,
            company_ticker=plan.company_ticker,
            user_prices=plan.analysis.user_prices if plan.analysis else None)
        response.evidence_context = context
    except Exception as exc:
        errors.append(f"aggregator: {exc}")
    context_dict = context.to_dict() if context is not None else {"query": query}

    t1 = time.time()
    try:
        final = synthesizer(query, context_dict)
    except Exception as exc:
        errors.append(f"synthesis: {exc}")
        final = {"answer": "Synthesis failed.", "source": "none", "confidence": 0.0}
    qwen_seconds = round(time.time() - t1, 2)
    if final.get("source") == "fallback":
        errors.append("qwen: unavailable or invalid output, fallback answer used")

    return {
        "query": query,
        "selected_agents": plan.agents,
        "agent_runs": [vars(r) for r in response.runs],
        "agent_results": response.to_dict().get("results", {}),
        "verification": context_dict.get("verification_flags") or context_dict.get("conflicts"),
        "final_response": final,
        "sources": context_dict.get("sources", []),
        "errors": errors,
        "metadata": {
            "total_seconds": round(time.time() - started, 2),
            "agents_seconds": agents_seconds,
            "qwen_seconds": qwen_seconds,
            "parallel_agents": True,
            "agent_timeout": timeout,
        },
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('Usage: python3 -m core.dynamic_pipeline "your question"')
        return 1
    out = run(argv[0])
    print("\n=== SELECTED AGENTS ===")
    for r in out["agent_runs"]:
        print(f"- {r['agent']}: {r['status']} ({r['duration_seconds']:.1f}s) {r['error'][:100]}")
    f = out["final_response"]
    print("\n=== FINAL ANSWER (" + str(f.get("source")) + ") ===")
    print(f.get("answer"))
    print("\nRecommendation:", f.get("recommendation"))
    for key in ("risks", "uncertainties", "assumptions"):
        print(f"\n{key.upper()}:")
        for item in f.get(key, []):
            print("-", item)
    print("\nConfidence:", f.get("confidence"))
    print("Errors:", out["errors"])
    print("Timing:", out["metadata"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
