"""
FinancialAgent: produces structured findings from computed financial
metrics (Phase 6's tools/financial_tools.py). No RAG retrieval needed --
every finding is a directly calculated number, not retrieved text.

Confidence is 1.0 for all findings (the calculation is exact, given the
underlying data), which differs conceptually from ResearchAgent's
retrieval-score-based confidence. This distinction should be made
explicit in the Synthesis Agent (Phase 15) when combining findings from
multiple agent types.
"""

from core.logger import get_logger
from core.schemas import AgentResult, Finding
from agents.base_agent import BaseAgent
from tools.financial_tools import build_financial_summary

log = get_logger(__name__)


class FinancialAgent(BaseAgent):
    agent_name = "FinancialAgent"

    def retrieve(self, query: str, top_k: int = 5):
        """Not used -- FinancialAgent works from structured data, not RAG."""
        return []

    def analyze(self, task: str) -> AgentResult:
        """
        task is currently informational only (e.g. "Analyze NVIDIA's
        financial health") -- this agent always computes the full
        financial summary rather than parsing the task text, since our
        calculations are cheap to compute in full. A future refinement
        could parse task for a specific year range.
        """
        summary = build_financial_summary(self.company_ticker)

        findings = []
        warnings = []

        findings.extend(self._revenue_findings(summary))
        findings.extend(self._margin_findings(summary))
        findings.extend(self._returns_findings(summary))

        if not summary["free_cash_flow"]:
            warnings.append(
                "Free Cash Flow could not be computed: CapEx data is unavailable "
                "at annual granularity in the source XBRL data (see Phase 6 notes)."
            )

        status = "success" if findings else "partial"

        return AgentResult(
            agent_name=self.agent_name,
            task=task,
            findings=findings,
            metrics={
                "years_of_revenue_data": len(summary["revenue"]),
                "revenue_cagr_pct": summary["revenue_cagr"],
            },
            warnings=warnings,
            status=status,
        )

    def _revenue_findings(self, summary: dict) -> list:
        findings = []
        cagr = summary["revenue_cagr"]
        if cagr is not None:
            years = sorted(summary["revenue"].keys())
            findings.append(Finding(
                claim=f"{self.company_ticker} revenue grew at a {cagr:.1f}% CAGR "
                      f"from {years[0]} to {years[-1]}.",
                evidence_text=(
                    f"Revenue: {years[0]}=${summary['revenue'][years[0]]:,.0f}, "
                    f"{years[-1]}=${summary['revenue'][years[-1]]:,.0f}. "
                    f"CAGR = ((end/start)^(1/years) - 1) * 100."
                ),
                source_chunk_id=f"{self.company_ticker}_revenue_cagr",
                source_form="XBRL_metrics",
                source_filing_date=years[-1],
                confidence=1.0,
            ))

        for period, growth in summary["revenue_yoy_growth"].items():
            findings.append(Finding(
                claim=f"{self.company_ticker} revenue grew {growth:+.1f}% YoY "
                      f"for the fiscal year ended {period}.",
                evidence_text=f"YoY growth computed from consecutive annual revenue figures.",
                source_chunk_id=f"{self.company_ticker}_revenue_yoy_{period}",
                source_form="XBRL_metrics",
                source_filing_date=period,
                confidence=1.0,
            ))
        return findings

    def _margin_findings(self, summary: dict) -> list:
        findings = []
        net_margins = summary["margins"]["net_margin"]
        for period, margin in net_margins.items():
            findings.append(Finding(
                claim=f"{self.company_ticker} net margin was {margin:.1f}% "
                      f"for the fiscal year ended {period}.",
                evidence_text="Net margin = Net Income / Revenue * 100.",
                source_chunk_id=f"{self.company_ticker}_net_margin_{period}",
                source_form="XBRL_metrics",
                source_filing_date=period,
                confidence=1.0,
            ))
        return findings

    def _returns_findings(self, summary: dict) -> list:
        findings = []
        roe = summary["returns"]["roe"]
        for period, value in roe.items():
            findings.append(Finding(
                claim=f"{self.company_ticker} ROE was {value:.1f}% "
                      f"for the fiscal year ended {period}.",
                evidence_text="ROE = Net Income / Stockholders' Equity * 100.",
                source_chunk_id=f"{self.company_ticker}_roe_{period}",
                source_form="XBRL_metrics",
                source_filing_date=period,
                confidence=1.0,
            ))
        return findings


if __name__ == "__main__":
    agent = FinancialAgent(company_ticker="NVDA")
    result = agent.run("Analyze NVIDIA's financial health")

    print(f"Status: {result.status}")
    print(f"Metrics: {result.metrics}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    print(f"\nTotal findings: {len(result.findings)}\n")

    for f in result.findings[:8]:
        print(f"- {f.claim}")
