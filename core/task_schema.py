"""
Structured objects for the dynamic orchestration pipeline:

    query -> QueryAnalysis -> ExecutionPlan -> AgentContext per agent -> DynamicResponse

Everything here is plain data (dataclasses), so plans are inspectable,
serializable with to_dict(), and testable without running any agent.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core import agent_result as standard
from core.schemas import AgentResult


class Intent(str, Enum):
    """Intents the Query Analyzer can detect. INVESTMENT_DECISION and GENERAL
    are primary-intent labels only; the rest map to specific agents."""
    RESEARCH = "research"
    FINANCIAL = "financial"
    MARKET = "market"
    COMPETITOR = "competitor"
    NEWS = "news"
    RISK = "risk"
    VALUATION = "valuation"
    VERIFICATION = "verification"
    INVESTMENT_DECISION = "investment_decision"
    GENERAL = "general"


# Agent keys used by the planner and orchestrator.
RESEARCH = "research"
FINANCIAL = "financial"
MARKET = "market"
NEWS = "news"
RISK = "risk"
VALUATION = "valuation"
RED_TEAM = "red_team"
SYNTHESIS = "synthesis"

# Fixed execution order for evidence-producing agents. Red-Team always runs
# after them, because it verifies their output.
PRODUCER_AGENTS = [RESEARCH, FINANCIAL, MARKET, NEWS, RISK, VALUATION]


@dataclass
class QueryAnalysis:
    query: str
    primary_intent: Intent
    intents: List[Intent] = field(default_factory=list)
    target_ticker: str = ""
    mentioned_tickers: List[str] = field(default_factory=list)
    competitor_tickers: List[str] = field(default_factory=list)
    # User-supplied numbers, e.g. {"current_price": 500.0, "cost_basis": 120.0}.
    # Kept separate from stored data; never used to override it.
    user_prices: Dict[str, float] = field(default_factory=dict)
    # intent value -> the terms that triggered it (traceability)
    matched_terms: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["primary_intent"] = self.primary_intent.value
        d["intents"] = [i.value for i in self.intents]
        return d


@dataclass
class AgentTask:
    agent: str           # agent key, e.g. "financial"
    reasoning_task: str  # what this agent contributes, e.g. "evaluate financial health"
    reason: str          # why the planner selected it


@dataclass
class ExecutionPlan:
    query: str
    intent: str
    company_ticker: str
    tasks: List[AgentTask] = field(default_factory=list)
    analysis: Optional[QueryAnalysis] = None

    @property
    def agents(self) -> List[str]:
        return [t.agent for t in self.tasks]

    @property
    def reasoning_tasks(self) -> List[str]:
        return [t.reasoning_task for t in self.tasks]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "company_ticker": self.company_ticker,
            "agents": self.agents,
            "reasoning_tasks": self.reasoning_tasks,
            "tasks": [asdict(t) for t in self.tasks],
            "analysis": self.analysis.to_dict() if self.analysis else None,
        }


@dataclass
class AgentContext:
    """Common input handed to every agent adapter."""
    query: str
    company_ticker: str
    intent: str
    reasoning_task: str
    analysis: Optional[QueryAnalysis] = None


@dataclass
class AgentRun:
    """Execution record for one agent (for logs, UI traces and tests)."""
    agent: str
    agent_name: str
    status: str  # "success" | "partial" | "failed"
    duration_seconds: float
    error: str = ""


@dataclass
class DynamicResponse:
    plan: ExecutionPlan
    agent_results: Dict[str, AgentResult] = field(default_factory=dict)
    verification: list = field(default_factory=list)
    runs: List[AgentRun] = field(default_factory=list)
    synthesis: Any = None
    # Standard contract (core.agent_result.AgentResult) per agent, incl. Red-Team/Synthesis.
    standard_results: Dict[str, "standard.AgentResult"] = field(default_factory=dict)
    evidence_context: Any = None  # core.evidence_aggregator.EvidenceContext

    def to_dict(self) -> Dict[str, Any]:
        """Plan plus standard results; this is what the Qwen layer will consume."""
        context = self.evidence_context.to_dict() if self.evidence_context is not None else None
        return {"plan": self.plan.to_dict(), **standard.results_to_dict(self.standard_results.values()),
                "evidence_context": context}

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Same shape as core.orchestrator.run_query(), so existing consumers
        (synthesize(), report_view) work unchanged."""
        return {
            "query": self.plan.query,
            "company_ticker": self.plan.company_ticker,
            "agents_activated": list(self.agent_results.keys()),
            "agent_results": self.agent_results,
            "verification": self.verification,
        }
