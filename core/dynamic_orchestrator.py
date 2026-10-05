"""
Dynamic orchestrator: natural-language question -> Query Analyzer ->
Agent Planner -> selected existing agents -> Red-Team -> (optional) Synthesis.

Runs alongside core/orchestrator.py (the keyword router), which is unchanged.
Existing agents are called through small adapters, so none of them had to
change their interface.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from agents.financial_agent import FinancialAgent
from agents.market_agent import MarketAgent
from agents.news_agent import NewsAgent
from agents.red_team_agent import run_verification
from agents.research_agent import ResearchAgent
from agents.risk_agent import RiskAgent
from agents.synthesis_agent import synthesize
from agents.valuation_agent import ValuationAgent
from core.agent_planner import plan_query
from core.logger import get_logger
from core.result_adapters import normalize_result, normalize_synthesis, normalize_verification
from core.schemas import AgentResult
from core.task_schema import (
    FINANCIAL, MARKET, NEWS, RED_TEAM, RESEARCH, RISK, VALUATION,
    AgentContext, AgentRun, DynamicResponse, ExecutionPlan,
)

log = get_logger(__name__)


@dataclass
class AgentAdapter:
    """Wraps one existing BaseAgent subclass behind the common AgentContext."""
    key: str
    agent_class: type

    @property
    def agent_name(self) -> str:
        return self.agent_class.agent_name

    def run(self, context: AgentContext) -> AgentResult:
        # Existing agents take a plain task string. The original question is the
        # most useful string (ResearchAgent uses it as its retrieval query).
        return self.agent_class(company_ticker=context.company_ticker).run(context.query)


AGENT_REGISTRY: Dict[str, AgentAdapter] = {
    RESEARCH: AgentAdapter(RESEARCH, ResearchAgent),
    FINANCIAL: AgentAdapter(FINANCIAL, FinancialAgent),
    MARKET: AgentAdapter(MARKET, MarketAgent),
    NEWS: AgentAdapter(NEWS, NewsAgent),
    RISK: AgentAdapter(RISK, RiskAgent),
    VALUATION: AgentAdapter(VALUATION, ValuationAgent),
}
RED_TEAM_NAME = "RedTeamAgent"
SYNTHESIS_NAME = "SynthesisAgent"


def _run_agent(adapter: AgentAdapter, context: AgentContext) -> Tuple[AgentResult, AgentRun]:
    name = adapter.agent_name
    log.info("[AGENT] %s started (%s)", name, context.reasoning_task)
    start = time.perf_counter()
    error = ""
    try:
        result = adapter.run(context)
    except Exception as e:  # adapter/constructor failures; BaseAgent.run catches the rest
        error = str(e)
        result = AgentResult(
            agent_name=name, task=context.query, status="failed",
            warnings=[f"Agent execution failed: {e}"],
        )
    duration = time.perf_counter() - start

    if result.status == "failed":
        error = error or "; ".join(result.warnings) or "unknown error"
        log.error("[AGENT] %s error: %s", name, error)
    else:
        log.info(
            "[AGENT] %s completed (status=%s, findings=%d, %.2fs)",
            name, result.status, len(result.findings), duration,
        )
    return result, AgentRun(adapter.key, name, result.status, round(duration, 3), error)


def _run_red_team(response: DynamicResponse, verifier: Callable = None) -> None:
    verifier = verifier or run_verification
    log.info("[AGENT] %s started (verify agent findings)", RED_TEAM_NAME)
    start = time.perf_counter()
    try:
        verification = verifier(response.agent_results)
    except Exception as e:
        log.error("[AGENT] %s error: %s", RED_TEAM_NAME, e)
        response.runs.append(AgentRun(RED_TEAM, RED_TEAM_NAME, "failed", round(time.perf_counter() - start, 3), str(e)))
        response.standard_results[RED_TEAM_NAME] = normalize_verification(None, error=str(e))
        return
    duration = time.perf_counter() - start
    log.info("[AGENT] %s completed (issues=%d, %.2fs)", RED_TEAM_NAME, len(verification), duration)
    response.runs.append(AgentRun(RED_TEAM, RED_TEAM_NAME, "success", round(duration, 3)))
    response.verification = verification
    response.standard_results[RED_TEAM_NAME] = normalize_verification(verification)


def execute_plan(plan: ExecutionPlan, registry: Optional[Dict[str, AgentAdapter]] = None,
                 synthesize_report: bool = False) -> DynamicResponse:
    """Run the agents in the plan, in plan order, then Red-Team and synthesis."""
    registry = registry if registry is not None else AGENT_REGISTRY
    response = DynamicResponse(plan=plan)

    for task in plan.tasks:
        if task.agent == RED_TEAM:
            continue
        adapter = registry.get(task.agent)
        if adapter is None:
            log.error("[AGENT] %s error: no adapter registered", task.agent)
            response.runs.append(AgentRun(task.agent, task.agent, "failed", 0.0, "no adapter registered"))
            continue
        context = AgentContext(
            query=plan.query,
            company_ticker=plan.company_ticker,
            intent=plan.intent,
            reasoning_task=task.reasoning_task,
            analysis=plan.analysis,
        )
        result, run = _run_agent(adapter, context)
        response.agent_results[adapter.agent_name] = result
        response.standard_results[adapter.agent_name] = normalize_result(result)
        response.runs.append(run)

    if RED_TEAM in plan.agents and response.agent_results:
        _run_red_team(response)

    if synthesize_report:
        log.info("[AGENT] %s started", SYNTHESIS_NAME)
        response.synthesis = synthesize(response.to_legacy_dict())
        response.standard_results[SYNTHESIS_NAME] = normalize_synthesis(response.synthesis)
        log.info("[AGENT] %s completed (status=%s)", SYNTHESIS_NAME, response.synthesis.overall_status)

    return response


def run_dynamic_query(query: str, company_ticker: Optional[str] = None,
                      run_verification_step: bool = True, synthesize_report: bool = False,
                      config: Optional[dict] = None,
                      registry: Optional[Dict[str, AgentAdapter]] = None) -> DynamicResponse:
    """Main entry: question in, DynamicResponse (plan + results) out."""
    log.info("[ORCHESTRATOR] Query: %s", query)
    plan = plan_query(query, config=config, include_verification=run_verification_step)
    if company_ticker:
        plan.company_ticker = company_ticker.strip().upper()
    return execute_plan(plan, registry=registry, synthesize_report=synthesize_report)


def format_response(response: DynamicResponse) -> str:
    """Short plain-text summary for the CLI."""
    plan = response.plan
    lines = [
        f"Query: {plan.query}",
        f"Intent: {plan.intent} | Ticker: {plan.company_ticker}",
        "Plan:",
    ]
    lines += [f"  - {t.agent}: {t.reasoning_task}  [{t.reason}]" for t in plan.tasks]
    lines.append("Runs:")
    for r in response.runs:
        extra = f" -- {r.error}" if r.error else ""
        lines.append(f"  - {r.agent_name}: {r.status} ({r.duration_seconds}s){extra}")
    for name, result in response.agent_results.items():
        lines.append(f"{name}: {len(result.findings)} findings")
        lines += [f"    * {f.claim}" for f in result.findings[:2]]
    if RED_TEAM in plan.agents:
        lines.append(f"Red-Team issues: {len(response.verification)}")
    return "\n".join(lines)


def _logs_to_stderr() -> None:
    """Keep stdout clean for --json: the project logger writes to stdout."""
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout:
            handler.setStream(sys.stderr)


def main(argv: Optional[List[str]] = None, registry: Optional[Dict[str, AgentAdapter]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question; agents are selected automatically.")
    parser.add_argument("query", nargs="+", help="natural-language question")
    parser.add_argument("--ticker", help="override the company ticker")
    parser.add_argument("--plan-only", action="store_true", help="print the plan as JSON and exit")
    parser.add_argument("--no-verify", action="store_true", help="skip Red-Team verification")
    parser.add_argument("--json", action="store_true", help="print plan + standard agent results as JSON")
    args = parser.parse_args(argv)
    query = " ".join(args.query)
    if args.json or args.plan_only:
        _logs_to_stderr()

    if args.plan_only:
        plan = plan_query(query, include_verification=not args.no_verify)
        if args.ticker:
            plan.company_ticker = args.ticker.strip().upper()
        print(json.dumps(plan.to_dict(), indent=2))
        return 0

    response = run_dynamic_query(
        query, company_ticker=args.ticker,
        run_verification_step=not args.no_verify, registry=registry,
    )
    print(response.to_json(indent=2) if args.json else format_response(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
