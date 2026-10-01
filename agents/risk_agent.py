"""
RiskAgent: wraps tools/risk_tools.py's identify_risks() into the standard
AgentResult/Finding pattern used by all agents in this project.
"""

from core.logger import get_logger
from core.schemas import AgentResult, Finding
from agents.base_agent import BaseAgent
from tools.risk_tools import identify_risks

log = get_logger(__name__)


class RiskAgent(BaseAgent):
    agent_name = "RiskAgent"

    def retrieve(self, query: str, top_k: int = 5):
        """Not used directly -- risk_tools.py calls rag.retriever internally
        per risk category, rather than a single agent-level retrieval."""
        return []

    def analyze(self, task: str) -> AgentResult:
        risks = identify_risks(self.company_ticker)

        if not risks:
            return AgentResult(
                agent_name=self.agent_name, task=task, status="partial",
                warnings=["No risk categories produced usable evidence."],
            )

        findings = []
        severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        for risk in risks:
            severity_counts[risk["severity"]] += 1
            top_ev = risk["evidence"][0] if risk["evidence"] else None

            findings.append(Finding(
                claim=(
                    f"{risk['category'].replace('_', ' ').title()} risk: "
                    f"severity={risk['severity']} "
                    f"({risk['distinct_years_mentioned']} distinct fiscal years mentioned)."
                ),
                evidence_text=(
                    f"{risk['severity_methodology']} "
                    f"{risk['uncertainty_corroboration']} "
                    f"Example evidence: {top_ev['text'] if top_ev else 'N/A'}"
                ),
                source_chunk_id=top_ev["chunk_id"] if top_ev else f"{self.company_ticker}_{risk['category']}_no_evidence",
                source_form="10-K/10-Q",
                source_filing_date=top_ev["filing_date"] if top_ev else "n/a",
                confidence=1.0,  # methodology is deterministic given the data, not a retrieval guess
            ))

        return AgentResult(
            agent_name=self.agent_name,
            task=task,
            findings=findings,
            metrics={
                "risk_categories_analyzed": len(risks),
                "severity_distribution": severity_counts,
            },
            warnings=[
                "Severity is a frequency-based proxy (distinct fiscal years "
                "mentioning a risk topic), not a definitive real-world impact "
                "measure. See individual finding evidence_text for full "
                "methodology disclosure."
            ],
            status="success",
        )


if __name__ == "__main__":
    agent = RiskAgent(company_ticker="NVDA")
    result = agent.run("Identify and assess NVIDIA's key business risks")

    print(f"Status: {result.status}")
    print(f"Metrics: {result.metrics}")
    print(f"Warnings: {result.warnings}")
    print(f"\nTotal findings: {len(result.findings)}\n")

    for f in result.findings:
        print(f"- {f.claim}")
