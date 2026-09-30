"""
MarketAgent: compares the target company against configured competitors
on revenue, growth, and margins, using real XBRL data (Phase 9 extension).

Per master prompt requirement: never assume a company is a competitor
just because it's in the same industry -- config.yaml's `reason` field
is surfaced explicitly in every finding.

Fiscal year misalignment note: companies compared here have different
fiscal year-end dates (NVDA=late Jan, AMD/INTC=late Dec, AVGO=~Oct/Nov).
Comparisons are grouped by the calendar year each fiscal year ends in,
which is an approximation -- explicitly disclosed in every comparison
finding, not silently smoothed over.
"""

from typing import Dict, Optional

from core.config_loader import load_config
from core.logger import get_logger
from core.schemas import AgentResult, Finding
from agents.base_agent import BaseAgent
from tools.financial_tools import calculate_cagr, get_annual_series

log = get_logger(__name__)


def _year_indexed(series: Dict[str, float]) -> Dict[int, float]:
    """Re-key a period_end-keyed series by calendar year (of period_end)."""
    return {int(period_end[:4]): value for period_end, value in series.items()}


class MarketAgent(BaseAgent):
    agent_name = "MarketAgent"

    def retrieve(self, query: str, top_k: int = 5):
        """Not used -- MarketAgent works from structured financial data."""
        return []

    def analyze(self, task: str) -> AgentResult:
        config = load_config()
        competitors = config.get("competitors", [])

        if not competitors:
            return AgentResult(
                agent_name=self.agent_name, task=task, status="partial",
                warnings=["No competitors configured in config.yaml."],
            )

        findings = []
        warnings = []

        target_revenue = get_annual_series("revenue", self.company_ticker)
        target_cagr = calculate_cagr(target_revenue)
        target_by_year = _year_indexed(target_revenue)

        if not target_revenue:
            warnings.append(f"No revenue data available for target company {self.company_ticker}.")

        for comp in competitors:
            comp_ticker = comp["ticker"]
            reason = comp["reason"]

            findings.append(Finding(
                claim=f"{comp_ticker} ({comp['name']}) is included as a comparable: {reason}",
                evidence_text=f"Inclusion criterion from project configuration: {reason}",
                source_chunk_id=f"{comp_ticker}_inclusion_rationale",
                source_form="config",
                source_filing_date="n/a",
                confidence=1.0,
            ))

            comp_revenue = get_annual_series("revenue", comp_ticker)
            if not comp_revenue:
                warnings.append(f"No revenue data available for competitor {comp_ticker}.")
                continue

            comp_cagr = calculate_cagr(comp_revenue)
            comp_by_year = _year_indexed(comp_revenue)

            if comp_cagr is not None and target_cagr is not None:
                comparison = "faster" if target_cagr > comp_cagr else "slower"
                findings.append(Finding(
                    claim=(
                        f"{self.company_ticker} revenue CAGR ({target_cagr:.1f}%) was "
                        f"{comparison} than {comp_ticker}'s ({comp_cagr:.1f}%) over "
                        f"their respective available fiscal history."
                    ),
                    evidence_text=(
                        f"{self.company_ticker} CAGR computed over "
                        f"{len(target_revenue)} fiscal years; {comp_ticker} CAGR "
                        f"computed over {len(comp_revenue)} fiscal years. Fiscal "
                        f"year-end dates differ between companies (see agent notes)."
                    ),
                    source_chunk_id=f"{self.company_ticker}_vs_{comp_ticker}_cagr",
                    source_form="XBRL_metrics",
                    source_filing_date=max(target_revenue.keys()),
                    confidence=1.0,
                ))

            common_years = sorted(set(target_by_year.keys()) & set(comp_by_year.keys()))
            if common_years:
                latest_year = common_years[-1]
                target_val = target_by_year[latest_year]
                comp_val = comp_by_year[latest_year]
                ratio = target_val / comp_val if comp_val else None
                if ratio is not None:
                    findings.append(Finding(
                        claim=(
                            f"For fiscal years ending in calendar {latest_year}, "
                            f"{self.company_ticker} revenue (${target_val:,.0f}) was "
                            f"{ratio:.1f}x {comp_ticker}'s (${comp_val:,.0f})."
                        ),
                        evidence_text=(
                            f"Comparison grouped by calendar year of fiscal year-end "
                            f"(approximation due to differing fiscal calendars)."
                        ),
                        source_chunk_id=f"{self.company_ticker}_vs_{comp_ticker}_revenue_{latest_year}",
                        source_form="XBRL_metrics",
                        source_filing_date=f"{latest_year}",
                        confidence=1.0,
                    ))
            else:
                warnings.append(
                    f"No overlapping calendar years found between "
                    f"{self.company_ticker} and {comp_ticker} for direct comparison."
                )

        status = "success" if findings else "partial"
        return AgentResult(
            agent_name=self.agent_name,
            task=task,
            findings=findings,
            metrics={"competitors_compared": len(competitors)},
            warnings=warnings,
            status=status,
        )


if __name__ == "__main__":
    agent = MarketAgent(company_ticker="NVDA")
    result = agent.run("Compare NVIDIA against its competitors")

    print(f"Status: {result.status}")
    print(f"Metrics: {result.metrics}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    print(f"\nTotal findings: {len(result.findings)}\n")

    for f in result.findings:
        print(f"- {f.claim}")
