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


# Named growth scenarios -- explicit, reasoned alternatives to naive
# historical CAGR extrapolation (see Phase 12 Part 1 finding: NVIDIA's
# full historical CAGR of ~45.8% produces an implausible $4.8T DCF
# output when extrapolated 5 years forward). These are ANALYST
# ASSUMPTIONS, not derived from data -- stated explicitly as such.
GROWTH_SCENARIOS = {
    "conservative": 0.10,
    "moderate": 0.20,
    "aggressive": 0.35,
    "historical_cagr": None,  # computed from data; flagged as likely unrealistic as a base case
}


def run_dcf_scenarios(
    company_ticker: str,
    wacc: float = 0.11,
    terminal_growth: float = 0.03,
    forecast_years: int = DEFAULT_FORECAST_YEARS,
) -> Dict[str, Dict]:
    """
    Run the DCF under each named growth scenario in GROWTH_SCENARIOS,
    so the output shows a RANGE of outcomes under different explicit
    assumptions rather than one potentially indefensible point estimate.
    """
    results = {}
    for name, rate in GROWTH_SCENARIOS.items():
        try:
            # For historical_cagr (rate=None), resolve the actual CAGR
            # explicitly BEFORE calling run_dcf, so the real rate used is
            # always visible in assumptions/claim text -- not displayed
            # as "None" (a transparency gap the Red-Team Agent caught).
            resolved_rate = rate
            if resolved_rate is None:
                revenue = get_annual_series("revenue", company_ticker)
                cagr_pct = calculate_cagr(revenue)
                if cagr_pct is None:
                    raise ValueError(f"Could not compute historical CAGR for {company_ticker}")
                resolved_rate = cagr_pct / 100

            dcf = run_dcf(
                company_ticker, wacc=wacc, terminal_growth=terminal_growth,
                forecast_years=forecast_years, revenue_growth_rate=resolved_rate,
            )
            dcf["scenario_name"] = name
            dcf["scenario_is_historical_extrapolation"] = (name == "historical_cagr")
            results[name] = dcf
        except ValueError as e:
            log.warning("Scenario '%s' failed for %s: %s", name, company_ticker, e)
    return results


def sensitivity_analysis(
    company_ticker: str,
    growth_rate: float,
    wacc_range: List[float] = None,
    terminal_growth_range: List[float] = None,
    forecast_years: int = DEFAULT_FORECAST_YEARS,
) -> Dict[str, Dict[str, float]]:
    """
    WACC x Terminal Growth sensitivity grid (per project spec requirement).
    Returns {wacc_label: {terminal_growth_label: enterprise_value}}.
    Combinations where wacc <= terminal_growth are skipped (mathematically
    invalid for Gordon Growth) and recorded as None, not silently omitted.
    """
    wacc_range = wacc_range or DEFAULT_WACC_RANGE
    terminal_growth_range = terminal_growth_range or DEFAULT_TERMINAL_GROWTH_RANGE

    grid = {}
    for wacc in wacc_range:
        wacc_label = f"{wacc:.1%}"
        grid[wacc_label] = {}
        for tg in terminal_growth_range:
            tg_label = f"{tg:.1%}"
            if wacc <= tg:
                grid[wacc_label][tg_label] = None
                continue
            try:
                dcf = run_dcf(
                    company_ticker, wacc=wacc, terminal_growth=tg,
                    forecast_years=forecast_years, revenue_growth_rate=growth_rate,
                )
                grid[wacc_label][tg_label] = dcf["enterprise_value"]
            except ValueError:
                grid[wacc_label][tg_label] = None

    return grid


def calculate_pe_ratio(company_ticker: str) -> Optional[Dict]:
    """
    Price / Earnings using the most recent available market close price
    and most recent diluted EPS from metrics. Returns None if either is
    unavailable, rather than fabricating a ratio.
    """
    eps_series = get_annual_series("eps_diluted", company_ticker)
    if not eps_series:
        return None

    latest_period = max(eps_series.keys())
    latest_eps = eps_series[latest_period]
    if latest_eps <= 0:
        log.warning("Latest EPS for %s is non-positive (%s); P/E is not meaningful", company_ticker, latest_eps)
        return None

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT date, close FROM prices WHERE company_ticker = ? "
            "ORDER BY date DESC LIMIT 1", (company_ticker,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    pe_ratio = row["close"] / latest_eps
    return {
        "price": row["close"],
        "price_date": row["date"],
        "eps": latest_eps,
        "eps_period": latest_period,
        "pe_ratio": pe_ratio,
        "note": (
            "P/E compares the LATEST market price to the MOST RECENT annual "
            "diluted EPS, which may not be perfectly period-matched (price "
            "date and EPS fiscal-year-end date can differ by months)."
        ),
    }


def calculate_ev_ebit(company_ticker: str) -> Optional[Dict]:
    """
    Enterprise Value / EBIT (substitute for EV/EBITDA).

    HONEST LIMITATION: true EBITDA requires Depreciation & Amortization,
    which was not collected in this project's XBRL concept list (see
    Phase 2 CONCEPTS_OF_INTEREST). Rather than fabricate a D&A estimate,
    this function uses EBIT (Operating Income) directly -- a legitimate,
    commonly-used multiple in its own right, clearly distinguished from
    EV/EBITDA, not presented as equivalent to it.

    Enterprise Value here = market cap (price x diluted shares implied
    from EPS/NetIncome) + total debt - cash. Shares outstanding are
    APPROXIMATED as NetIncome / EPS_diluted (we did not separately
    collect a shares-outstanding concept) -- this approximation is
    disclosed in the output.
    """
    operating_income = get_annual_series("operating_income", company_ticker)
    net_income = get_annual_series("net_income", company_ticker)
    eps_diluted = get_annual_series("eps_diluted", company_ticker)
    debt = get_annual_series("long_term_debt", company_ticker)
    cash = get_annual_series("cash", company_ticker)

    if not operating_income or not net_income or not eps_diluted:
        return None

    latest_period = max(operating_income.keys())
    if latest_period not in net_income or latest_period not in eps_diluted:
        return None
    if eps_diluted[latest_period] == 0:
        return None

    implied_shares = net_income[latest_period] / eps_diluted[latest_period]

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT date, close FROM prices WHERE company_ticker = ? "
            "ORDER BY date DESC LIMIT 1", (company_ticker,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    market_cap = row["close"] * implied_shares
    latest_debt = debt.get(latest_period, 0.0)
    latest_cash = cash.get(latest_period, 0.0)
    enterprise_value = market_cap + latest_debt - latest_cash

    ebit = operating_income[latest_period]
    if ebit <= 0:
        return None

    return {
        "ebit": ebit,
        "ebit_period": latest_period,
        "market_cap": market_cap,
        "implied_shares_outstanding": implied_shares,
        "total_debt": latest_debt,
        "cash": latest_cash,
        "enterprise_value": enterprise_value,
        "ev_ebit_ratio": enterprise_value / ebit,
        "note": (
            "This is EV/EBIT, not EV/EBITDA -- D&A data was not collected "
            "in this project (see Phase 2 scope). Shares outstanding is "
            "APPROXIMATED as NetIncome/EPS_diluted, not a directly "
            "reported figure."
        ),
    }


if __name__ == "__main__":
    print()
    print("=== Growth Scenarios DCF (WACC=11%, Terminal Growth=3%) ===")
    scenarios = run_dcf_scenarios("NVDA")
    for name, result in scenarios.items():
        flag = ""
        if result["scenario_is_historical_extrapolation"]:
            flag = " [LIKELY UNREALISTIC AS BASE CASE]"
        rate = result["assumptions"]["revenue_growth_rate_used"]
        ev = result["enterprise_value"]
        print()
        print(f"  Scenario: {name}{flag}")
        print(f"    Growth rate used: {rate}")
        print(f"    Enterprise Value: ${ev:,.0f}")

    print()
    print()
    print("=== Sensitivity Analysis (growth=20% moderate scenario) ===")
    grid = sensitivity_analysis("NVDA", growth_rate=0.20)
    tg_labels = list(next(iter(grid.values())).keys())
    header = "WACC \\ Terminal Growth: " + "  ".join(f"{t:>10}" for t in tg_labels)
    print(header)
    for wacc_label, row in grid.items():
        formatted = []
        for tg_label in tg_labels:
            val = row[tg_label]
            if val is not None:
                formatted.append(f"{val/1e9:>9.0f}B")
            else:
                formatted.append(f"{'N/A':>10}")
        print(f"{wacc_label:>6}            " + "  ".join(formatted))

    print()
    print()
    print("=== P/E Ratio ===")
    pe = calculate_pe_ratio("NVDA")
    if pe:
        print(f"  Price ({pe['price_date']}): ${pe['price']:.2f}")
        print(f"  Diluted EPS ({pe['eps_period']}): ${pe['eps']:.2f}")
        print(f"  P/E Ratio: {pe['pe_ratio']:.1f}x")
        print(f"  Note: {pe['note']}")
    else:
        print("  P/E unavailable (missing EPS or price data)")
