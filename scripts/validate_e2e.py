"""End-to-end validation (Prompt 10). Run: PYTHONPATH=. python3 scripts/validate_e2e.py"""
import json
import logging
import os
import sys
import time

logging.disable(logging.CRITICAL)

from core import dynamic_pipeline as dp  # noqa: E402
from core import report_builder as rb  # noqa: E402
from core.position_calc import calculate_position  # noqa: E402

QUERIES = [
    "What is NVIDIA's financial health?",
    "Compare NVIDIA with AMD.",
    "What are NVIDIA's biggest risks?",
    "What does the DCF valuation indicate?",
    "What happened in NVIDIA's latest earnings?",
    "NVIDIA is trading at $500 today, yesterday it was $200, and my average purchase price "
    "is $120. Should I buy more or sell my current holdings?",
]
SKIP = lambda q, c: {"answer": "", "source": "skipped"}  # noqa: E731
LINES = []


def log(text=""):
    print(text)
    LINES.append(text)


def analyze(q, registry=None, ticker=None, ask=None):
    t0 = time.time()
    out = dp.run(q, company_ticker=ticker, registry=registry, synthesizer=SKIP)
    ctx = out.get("context") or {}
    kwargs = {"ask": ask} if ask else {}
    rep = rb.build_report(q, ctx, out["agent_runs"], out["errors"], **kwargs)
    return out, ctx, rep, round(time.time() - t0, 1)


def run_queries():
    log("# Validation report\n")
    log(f"Qwen server (must be local): {rb.OLLAMA_URL}   model: {rb.MODEL}\n")
    log("## Six test questions\n")
    for i, q in enumerate(QUERIES, 1):
        out, ctx, rep, secs = analyze(q)
        agents = out["selected_agents"]
        failed = [r["agent"] for r in out["agent_runs"] if r["status"] == "failed"]
        counts = {k: len(ctx.get(k) or []) for k in (
            "financial_metrics", "valuation", "risks", "news_events", "competitor_findings",
            "sources", "conflicts", "missing_evidence")}
        log(f"### Test {i}: {q[:70]}")
        log(f"- Selected agents ({len(agents)}): {', '.join(agents)}")
        log("- Agent runs: " + ", ".join(
            f"{r['agent']}={r['status']}({r['duration_seconds']:.1f}s)" for r in out["agent_runs"]))
        log(f"- Agent failures: {failed or 'none'}")
        log(f"- Evidence counts: {counts}")
        log(f"- Position math ran: {bool(ctx.get('position_calculations'))}")
        log(f"- Context size sent to the report: {len(json.dumps(ctx, default=str))} characters")
        log(f"- Qwen reachable: {rep['qwen_ok']}   removed numbers: {rep['removed_numbers']}")
        log(f"- Confidence (code): {rep['confidence']}")
        log(f"- Errors: {out['errors'] or 'none'}")
        log(f"- Qwen summary: {rep['parts']['summary'][:220]}")
        log(f"- Total latency: {secs}s\n")


def scenario(name, fn):
    try:
        log(f"- PASS: {name}: {fn()}")
    except Exception as exc:  # a crash here means the system did not degrade gracefully
        log(f"- CRASH: {name}: {type(exc).__name__}: {exc}")


def ollama_off():
    q = QUERIES[5]
    off = lambda p: rb.ask_summary(p, base_url="http://localhost:9999")  # noqa: E731
    out, ctx, rep, secs = analyze(q, ask=off)
    assert rep["qwen_ok"] is False and "## Executive Conclusion" in rep["markdown"]
    return "report still complete, Qwen marked as unavailable"


def one_agent_missing():
    reg = {k: v for k, v in dp.do.AGENT_REGISTRY.items() if k != "risk"}
    out, ctx, rep, secs = analyze(QUERIES[0], registry=reg)
    assert any("risk" in e for e in out["errors"]) and "## Verification" in rep["markdown"]
    return "error recorded: " + "; ".join(out["errors"])[:100]


def missing_financial_data():
    out, ctx, rep, secs = analyze(QUERIES[0], ticker="ZZZZ")
    return "agent statuses " + str({r["agent"]: r["status"] for r in out["agent_runs"]})[:140]


def malformed_agent():
    class Bad:
        agent_name = "FinancialAgent"

        def run(self, context):
            return "garbage"

    reg = dict(dp.do.AGENT_REGISTRY)
    reg["financial"] = Bad()
    out, ctx, rep, secs = analyze(QUERIES[0], registry=reg)
    return "errors: " + str(out["errors"])[:140]


def empty_evidence():
    rep = rb.build_report("q", {}, ask=lambda p: "never called")
    assert "No points to summarize." in rep["markdown"]
    return "empty evidence handled, confidence " + str(rep["confidence"])


def weird_prices():
    calculate_position({"current_price": -5.0, "previous_price": 0, "cost_basis": 0})
    return "negative and zero prices handled"


def failure_cases():
    log("## Failure cases\n")
    scenario("Ollama not reachable", ollama_off)
    scenario("one agent unavailable (risk)", one_agent_missing)
    scenario("missing financial data (ticker ZZZZ)", missing_financial_data)
    scenario("malformed agent output", malformed_agent)
    scenario("empty evidence", empty_evidence)
    scenario("invalid user prices", weird_prices)
    log("- NOT TESTED HERE: invalid valuation assumptions (covered by the earlier agent tests)\n")


def prohibited():
    log("## Prohibited libraries loaded in this run\n")
    names = {"langchain", "langgraph", "openai", "anthropic"}
    found = sorted(m for m in sys.modules
                   if m.split(".")[0] in names or m.startswith("google.generativeai"))
    log(f"- {found or 'none'}\n")


if __name__ == "__main__":
    run_queries()
    failure_cases()
    prohibited()
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/validation_report.md", "w") as f:
        f.write("\n".join(LINES))
    print("Saved to outputs/validation_report.md")
