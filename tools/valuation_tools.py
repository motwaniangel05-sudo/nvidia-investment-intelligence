"""
Valuation engine: DCF model and multiples, built from scratch.

FCF METHODOLOGY NOTE (honest disclosure per project requirements):
True Free Cash Flow to Firm requires Operating Cash Flow minus CapEx.
CapEx is unavailable at annual granularity for NVDA in our XBRL dataset
(see Phase 6 "KNOWN LIMITATION" in financial_tools.py -- NVIDIA only
tags PaymentsToAcquirePropertyPlantAndEquipment in 10-Q year-to-date
figures that never reach a full 12-month span). Rather than fabricate a
CapEx estimate, this module uses Operating Cash Flow directly as an
explicitly labeled "FCF proxy" throughout. Every DCF output states this
limitation. This likely OVERSTATES true FCF (since real CapEx is not
subtracted), and that bias is disclosed, not hidden.

DISCOUNT RATE NOTE: computing a true WACC requires market data (beta,
risk-free rate, market risk premium, cost of debt) not collected in this
project. A stated, reasonable WACC range is used as an explicit
assumption, with sensitivity analysis across that range -- not a single
falsely-precise computed figure.
"""

from typing import Dict, List, Optional

from core.logger import get_logger
from core.schemas import get_connection
from tools.financial_tools import calculate_cagr, get_annual_series

log = get_logger(__name__)

FCF_PROXY_DISCLAIMER = (
    "FCF proxy = Operating Cash Flow only (CapEx unavailable at annual "
    "granularity for this company -- see Phase 6 notes). This likely "
    "OVERSTATES true unlevered FCF since no CapEx is subtracted."
)

DEFAULT_FORECAST_YEARS = 5
DEFAULT_TAX_RATE = 0.15  # NVIDIA's effective tax rate has varied; see assumptions in output
DEFAULT_WACC_RANGE = [0.09, 0.10, 0.11, 0.12, 0.13]
DEFAULT_TERMINAL_GROWTH_RANGE = [0.02, 0.025, 0.03, 0.035, 0.04]


def forecast_revenue(
    company_ticker: str,
    forecast_years: int = DEFAULT_FORECAST_YEARS,
    growth_rate: Optional[float] = None,
) -> Dict[str, float]:
    """
    Project revenue forward using either a given growth_rate or the
    historical CAGR (computed from get_annual_series, Phase 6) if none
    is given. Returns {period_label: projected_revenue}.

    Using full historical CAGR (e.g. NVIDIA's ~45.8%) as a perpetual
    forward assumption is aggressive and explicitly flagged as such in
    build_valuation_summary()'s assumptions section -- this function
    itself just does the arithmetic given whatever rate it's told to use.
    """
    revenue = get_annual_series("revenue", company_ticker)
    if not revenue:
        raise ValueError(f"No revenue data available for {company_ticker}")

    last_period = max(revenue.keys())
    last_value = revenue[last_period]

    if growth_rate is None:
        growth_rate = calculate_cagr(revenue)
        if growth_rate is None:
            raise ValueError(f"Could not compute a growth rate for {company_ticker}")
        growth_rate = growth_rate / 100  # calculate_cagr returns a percentage

    projections = {}
    value = last_value
    for year in range(1, forecast_years + 1):
        value = value * (1 + growth_rate)
        projections[f"Year {year}"] = value

    return projections


def calculate_fcf_proxy_margin(company_ticker: str) -> Optional[float]:
    """
    Historical FCF-proxy margin (Operating Cash Flow / Revenue), averaged
    over available years. Used to project future FCF proxy from
    forecasted revenue. Returns None if either series is unavailable.
    """
    revenue = get_annual_series("revenue", company_ticker)
    ocf = get_annual_series("operating_cash_flow", company_ticker)

    common_periods = sorted(set(revenue.keys()) & set(ocf.keys()))
    if not common_periods:
        return None

    margins = [ocf[p] / revenue[p] for p in common_periods if revenue[p] != 0]
    if not margins:
        return None

    return sum(margins) / len(margins)


def run_dcf(
    company_ticker: str,
    wacc: float,
    terminal_growth: float,
    forecast_years: int = DEFAULT_FORECAST_YEARS,
    revenue_growth_rate: Optional[float] = None,
    tax_rate: float = DEFAULT_TAX_RATE,
) -> Dict:
    """
    Full DCF flow per project spec:
    Revenue Forecast -> (simplified: FCF proxy margin applied directly,
    see note below) -> Discount -> PV -> Terminal Value -> Enterprise
    Value -> (no debt/cash netting -- see note) -> implied value.

    SIMPLIFICATION NOTE: the full EBIT->Taxes->NOPAT->+D&A->-CapEx->-ΔWC
    chain from the project spec assumes CapEx and detailed working-capital
    data we don't have (see module docstring). This implementation uses
    the historical FCF-proxy margin (OCF/Revenue) applied to forecasted
    revenue as a simplified stand-in for that full chain. This is a
    disclosed simplification, not the full textbook model.

    Equity Value here is computed as Enterprise Value only (no net debt
    adjustment), since net debt (total debt minus cash, properly matched)
    was not separately validated in this project -- also disclosed.
    """
    revenue_forecast = forecast_revenue(company_ticker, forecast_years, revenue_growth_rate)
    fcf_margin = calculate_fcf_proxy_margin(company_ticker)

    if fcf_margin is None:
        raise ValueError(f"Cannot compute FCF-proxy margin for {company_ticker}")

    fcf_proxy_forecast = {
        period: revenue * fcf_margin
        for period, revenue in revenue_forecast.items()
    }

    present_values = {}
    for i, (period, fcf) in enumerate(fcf_proxy_forecast.items(), start=1):
        pv = fcf / ((1 + wacc) ** i)
        present_values[period] = pv

    final_year_fcf = list(fcf_proxy_forecast.values())[-1]
    if wacc <= terminal_growth:
        raise ValueError(
            f"WACC ({wacc}) must exceed terminal_growth ({terminal_growth}) "
            f"for Terminal Value to be meaningful (Gordon Growth Model requirement)."
        )
    terminal_value = final_year_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + wacc) ** forecast_years)

    enterprise_value = sum(present_values.values()) + pv_terminal_value

    return {
        "company_ticker": company_ticker,
        "assumptions": {
            "wacc": wacc,
            "terminal_growth": terminal_growth,
            "forecast_years": forecast_years,
            "revenue_growth_rate_used": revenue_growth_rate,
            "fcf_proxy_margin_used": fcf_margin,
            "tax_rate_stated_not_applied": tax_rate,  # see simplification note
        },
        "fcf_disclaimer": FCF_PROXY_DISCLAIMER,
        "revenue_forecast": revenue_forecast,
        "fcf_proxy_forecast": fcf_proxy_forecast,
        "present_values": present_values,
        "sum_pv_explicit_period": sum(present_values.values()),
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
    }


if __name__ == "__main__":
    print("=== Revenue Forecast (NVDA, historical CAGR) ===")
    forecast = forecast_revenue("NVDA")
    for period, value in forecast.items():
        print(f"  {period}: ${value:,.0f}")

    print("\n=== FCF Proxy Margin ===")
    margin = calculate_fcf_proxy_margin("NVDA")
    print(f"  Average OCF/Revenue margin: {margin:.1%}" if margin else "  Unavailable")

    print("\n=== Base Case DCF (WACC=11%, Terminal Growth=3%) ===")
    dcf = run_dcf("NVDA", wacc=0.11, terminal_growth=0.03)
    print(f"  {dcf['fcf_disclaimer']}\n")
    print(f"  Sum of PV (explicit period): ${dcf['sum_pv_explicit_period']:,.0f}")
    print(f"  Terminal Value: ${dcf['terminal_value']:,.0f}")
    print(f"  PV of Terminal Value: ${dcf['pv_terminal_value']:,.0f}")
    print(f"  Enterprise Value: ${dcf['enterprise_value']:,.0f}")
