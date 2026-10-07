"""
Agent Planner: turns a QueryAnalysis into an ExecutionPlan, i.e. which
existing agents to run, in what order, and what each one contributes.

The plan is built from an explicit rule table (one playbook per primary
intent, plus one agent per extra detected intent), so every selection can be
traced to a rule. It never runs every agent by default.
"""

from typing import Dict, List, Optional

from core.logger import get_logger
from core.query_analyzer import analyze_query
from core.task_schema import (
    FINANCIAL, MARKET, NEWS, PRODUCER_AGENTS, RED_TEAM, RESEARCH, RISK, VALUATION,
    AgentTask, ExecutionPlan, Intent, QueryAnalysis,
)

log = get_logger(__name__)

# Primary intent -> evidence-producing agents to run.
PLAYBOOKS: Dict[Intent, List[str]] = {
    Intent.FINANCIAL: [FINANCIAL, RESEARCH, RISK],
    Intent.INVESTMENT_DECISION: [FINANCIAL, MARKET, VALUATION, RISK, NEWS],
    Intent.COMPETITOR: [RESEARCH, FINANCIAL, MARKET, RISK],
    Intent.NEWS: [NEWS, FINANCIAL, MARKET, RESEARCH],
    Intent.VALUATION: [VALUATION, FINANCIAL, MARKET],
    Intent.RISK: [RISK, RESEARCH, FINANCIAL],
    Intent.MARKET: [MARKET, FINANCIAL, RESEARCH],
    Intent.RESEARCH: [RESEARCH, FINANCIAL],
    Intent.VERIFICATION: [RESEARCH, FINANCIAL],
    Intent.GENERAL: [RESEARCH, FINANCIAL],
}

# A secondary intent adds only its own core agent, so plans stay small.
INTENT_TO_AGENT: Dict[Intent, str] = {
    Intent.RESEARCH: RESEARCH,
    Intent.FINANCIAL: FINANCIAL,
    Intent.MARKET: MARKET,
    Intent.COMPETITOR: MARKET,
    Intent.NEWS: NEWS,
    Intent.RISK: RISK,
    Intent.VALUATION: VALUATION,
}

REASONING_TASKS: Dict[str, str] = {
    RESEARCH: "gather qualitative business context from filings",
    FINANCIAL: "evaluate financial health",
    MARKET: "evaluate market and competitive position",
    NEWS: "check recent events",
    RISK: "identify risks",
    VALUATION: "evaluate current valuation",
    RED_TEAM: "verify agent findings",
}


def _reasoning_task(agent: str, analysis: QueryAnalysis) -> str:
    if agent == MARKET and analysis.competitor_tickers:
        return f"compare {analysis.target_ticker} against {', '.join(analysis.competitor_tickers)}"
    if agent == VALUATION and "current_price" in analysis.user_prices:
        return "evaluate current valuation (user-supplied price shown separately, not used as data)"
    return REASONING_TASKS[agent]


def plan_from_analysis(analysis: QueryAnalysis, include_verification: bool = True) -> ExecutionPlan:
    reasons: Dict[str, str] = {}
    for agent in PLAYBOOKS[analysis.primary_intent]:
        reasons.setdefault(agent, f"playbook for primary intent '{analysis.primary_intent.value}'")
    for intent in analysis.intents:
        agent = INTENT_TO_AGENT.get(intent)
        if agent:
            terms = ", ".join(analysis.matched_terms.get(intent.value, []))
            reasons.setdefault(agent, f"intent '{intent.value}' detected ({terms})")

    tasks = [
        AgentTask(agent=a, reasoning_task=_reasoning_task(a, analysis), reason=reasons[a])
        for a in PRODUCER_AGENTS if a in reasons
    ]
    if include_verification and tasks:
        tasks.append(AgentTask(
            agent=RED_TEAM,
            reasoning_task=REASONING_TASKS[RED_TEAM],
            reason="always verify evidence-producing agents",
        ))

    plan = ExecutionPlan(
        query=analysis.query,
        intent=analysis.primary_intent.value,
        company_ticker=analysis.target_ticker,
        tasks=tasks,
        analysis=analysis,
    )
    log.info("[PLANNER] Selected: %s", ", ".join(plan.agents))
    return plan


def plan_query(query: str, config: Optional[dict] = None, include_verification: bool = True) -> ExecutionPlan:
    """Main entry: analyze the question and return the structured plan."""
    return plan_from_analysis(analyze_query(query, config), include_verification)
