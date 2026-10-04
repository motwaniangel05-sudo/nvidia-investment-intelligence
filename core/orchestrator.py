"""
Orchestrator: routes a user query to the relevant agents, executes them,
runs Red-Team verification, and returns a consolidated response.

Per master prompt: "Do NOT unnecessarily run every agent." Query
classification is keyword-based (transparent, debuggable), consistent
with this project's no-pretrained-LLM constraint -- not an attempt to
"understand" the query semantically, just to match it to relevant agents.
"""

import re
from typing import Dict, List

from agents.synthesis_agent import synthesize
from core.logger import get_logger
from core.schemas import AgentResult
from agents.financial_agent import FinancialAgent
from agents.market_agent import MarketAgent
from agents.news_agent import NewsAgent
from agents.red_team_agent import run_verification
from agents.research_agent import ResearchAgent
from agents.risk_agent import RiskAgent
from agents.valuation_agent import ValuationAgent

log = get_logger(__name__)

# Keyword triggers per agent. A query activates an agent if ANY of its
# keywords appear (case-insensitive substring match). Deliberately broad
# but explicit -- every activation decision is traceable to a specific
# matched keyword, not opaque "AI judgment".
AGENT_KEYWORDS = {
    "ResearchAgent": [
        "business", "segment", "product", "strategy", "company", "overview",
        "operations", "what does", "what is", "describe",
    ],
    "FinancialAgent": [
        "financial", "revenue", "profit", "margin", "earnings", "growth",
        "cagr", "roe", "roa", "cash flow", "balance sheet", "income",
    ],
    "MarketAgent": [
        "competitor", "compare", "competition", "market position",
        "versus", "vs", "industry", "rival",
    ],
    "NewsAgent": [
        "news", "event", "announcement", "recent", "8-k", "press release",
    ],
    "RiskAgent": [
        "risk", "threat", "uncertainty", "exposure", "vulnerability",
        "due diligence", "concern",
    ],
    "ValuationAgent": [
        "valuation", "value", "worth", "dcf", "price target", "fair value",
        "p/e", "pe ratio", "ebitda", "intrinsic",
    ],
}

AGENT_CLASSES = {
    "ResearchAgent": ResearchAgent,
    "FinancialAgent": FinancialAgent,
    "MarketAgent": MarketAgent,
    "NewsAgent": NewsAgent,
    "RiskAgent": RiskAgent,
    "ValuationAgent": ValuationAgent,
}


def _keyword_matches(kw: str, text: str) -> bool:
    """Short keywords (<4 chars) must match as whole tokens, so 'roa' does not
    fire inside 'broad'. Longer keywords stay substring matches ('earn' -> 'earnings')."""
    if len(kw) < 4:
        return re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", text) is not None
    return kw in text


def classify_query(query: str) -> List[str]:
    """
    Return the list of agent names whose keywords match the query.
    If NO agent matches, default to ["ResearchAgent", "FinancialAgent"]
    (the two most broadly informative agents) rather than running nothing
    or running everything -- a documented, sensible fallback.
    """
    query_lower = query.lower()
    matched = []
    for agent_name, keywords in AGENT_KEYWORDS.items():
        if any(_keyword_matches(kw, query_lower) for kw in keywords):
            matched.append(agent_name)

    if not matched:
        log.info("No keyword match for query '%s'; using default fallback agents", query)
        matched = ["ResearchAgent", "FinancialAgent"]

    return matched


def run_query(
    company_ticker: str,
    query: str,
    run_verification_step: bool = True,
    synthesize_report: bool = False,
) -> Dict:
    """
    Main entry: classify the query, run matched agents, optionally run
    Red-Team verification, return everything in a structured response.
    """
    matched_agents = classify_query(query)
    log.info("Query '%s' matched agents: %s", query, matched_agents)

    agent_results: Dict[str, AgentResult] = {}
    for agent_name in matched_agents:
        agent_class = AGENT_CLASSES[agent_name]
        agent = agent_class(company_ticker=company_ticker)
        log.info("Running %s...", agent_name)
        result = agent.run(query)
        agent_results[agent_name] = result

    verification = []
    if run_verification_step and agent_results:
        log.info("Running Red-Team verification...")
        verification = run_verification(agent_results)

    response = {
        "query": query,
        "company_ticker": company_ticker,
        "agents_activated": matched_agents,
        "agent_results": agent_results,
        "verification": verification,
    }
    if synthesize_report:
        response["synthesis"] = synthesize(response)
    return response


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "Analyze NVIDIA's financial health and valuation"

    response = run_query("NVDA", query)

    print(f"Query: {response['query']}")
    print(f"Agents activated: {response['agents_activated']}")
    print()

    for agent_name, result in response["agent_results"].items():
        print(f"=== {agent_name} ({result.status}, {len(result.findings)} findings) ===")
        for f in result.findings[:3]:
            print(f"  - {f.claim}")
        if len(result.findings) > 3:
            print(f"  ... and {len(result.findings) - 3} more")
        print()

    print(f"=== Red-Team Verification ({len(response['verification'])} issues) ===")
    for v in response["verification"][:5]:
        print(f"  [{v.verification_result}] {v.claim[:100]}")
