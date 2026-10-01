"""
ValuationAgent: wraps tools/valuation_tools.py's DCF, scenarios,
sensitivity analysis, P/E, and EV/EBIT into the standard AgentResult
pattern used by all agents in this project.
"""

from core.logger import get_logger
from core.schemas import AgentResult, Finding
from agents.base_agent import BaseAgent
from tools.valuation_tools import (
    calculate_ev_ebit,
    calculate_pe_ratio,
    run_dcf_scenarios,
)

log = get_logger(__name__)


class ValuationAgent(BaseAgent):
    agent_name = "ValuationAgent"

    def retrieve(self, query: str, top_k: int = 5):
        """Not used -- ValuationAgent works from structured financial data."""
        return []

    def analyze(self, task: str) -> AgentResult:
        findings = []
        warnings = []

        try:
            scenarios = run_dcf_scenarios(self.company_ticker)
        except Exception as e:
            scenarios = {}
            warnings.append(f"DCF scenarios could not be computed: {e}")

        for name, result in scenarios.items():
            is_historical = result.get("scenario_is_historical_extrapolation", False)
            flag = " (historical-CAGR extrapolation -- likely unrealistic as a base case)" if is_historical else ""
            findings.append(Finding(
                claim=(
                    f"DCF scenario '{name}'{flag}: Enterprise Value = "
                    f"${result['enterprise_value']:,.0f} "
                    f"(growth rate used: {result['assumptions']['revenue_growth_rate_used']})."
                ),
                evidence_text=result["fcf_disclaimer"],
                source_chunk_id=f"{self.company_ticker}_dcf_{name}",
                source_form="DCF_model",
                source_filing_date="n/a",
                confidence=1.0,
            ))

        pe = calculate_pe_ratio(self.company_ticker)
        if pe:
            findings.append(Finding(
                claim=f"{self.company_ticker} P/E ratio: {pe['pe_ratio']:.1f}x (price ${pe['price']:.2f} / EPS ${pe['eps']:.2f}).",
                evidence_text=pe["note"],
                source_chunk_id=f"{self.company_ticker}_pe_ratio",
                source_form="market_data",
                source_filing_date=pe["price_date"],
                confidence=1.0,
            ))
        else:
            warnings.append("P/E ratio could not be computed (missing EPS or price data).")

        ev_ebit = calculate_ev_ebit(self.company_ticker)
        if ev_ebit:
            findings.append(Finding(
                claim=f"{self.company_ticker} EV/EBIT ratio: {ev_ebit['ev_ebit_ratio']:.1f}x.",
                evidence_text=ev_ebit["note"],
                source_chunk_id=f"{self.company_ticker}_ev_ebit",
                source_form="market_data_and_XBRL",
                source_filing_date=ev_ebit["ebit_period"],
                confidence=1.0,
            ))
        else:
            warnings.append("EV/EBIT could not be computed (missing required data).")

        if not scenarios:
            warnings.append("No DCF scenarios could be computed -- revenue data may be unavailable.")

        warnings.append(
            "All DCF outputs use Operating Cash Flow as an FCF proxy "
            "(CapEx unavailable at annual granularity -- see Phase 6 notes). "
            "This likely overstates true unlevered FCF."
        )

        status = "success" if findings else "partial"
        return AgentResult(
            agent_name=self.agent_name,
            task=task,
            findings=findings,
            metrics={
                "scenarios_computed": len(scenarios),
                "pe_available": pe is not None,
                "ev_ebit_available": ev_ebit is not None,
            },
            warnings=warnings,
            status=status,
        )


if __name__ == "__main__":
    agent = ValuationAgent(company_ticker="NVDA")
    result = agent.run("Value NVIDIA using DCF and market multiples")

    print(f"Status: {result.status}")
    print(f"Metrics: {result.metrics}")
    print(f"Warnings: {result.warnings}")
    print(f"\nTotal findings: {len(result.findings)}\n")

    for f in result.findings:
        print(f"- {f.claim}")
