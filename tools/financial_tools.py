"""
Financial analysis engine: transparent, from-scratch calculations on the
metrics table (Phase 4's knowledge base).

Core insight (discovered during Phase 4 verification): XBRL data cannot be
trusted by fiscal_year alone. The same concept/period can appear multiple
times across filings (prior-year comparisons), and companies sometimes
switch which XBRL tag they use for the same concept (e.g. NVIDIA used
'Revenues' then briefly 'RevenueFromContractWithCustomerExcludingAssessedTax'
around the ASC 606 transition). get_annual_series() is the one function
every other calculation in this module depends on to get this right.
"""

from datetime import date, datetime
from typing import Dict, List, Optional

from core.logger import get_logger
from core.schemas import get_connection

log = get_logger(__name__)

# Tolerance for what counts as an "annual" (~12-month) reporting period.
# NVIDIA's fiscal year isn't exactly 365 days every year (it ends on the
# last Sunday of January, so it drifts slightly), so we allow a range
# rather than an exact match.
MIN_ANNUAL_DAYS = 350
MAX_ANNUAL_DAYS = 380

# For each logical metric, the XBRL concepts that represent it, in
# preference order. Revenue has two due to the tag-switch we discovered;
# other metrics currently have one known tag each, but are listed this
# way for consistency and to make adding alternates easy later.
METRIC_CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "stockholders_equity": ["StockholdersEquity"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "long_term_debt": ["LongTermDebtNoncurrent"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}

# Point-in-time concepts (balance sheet items) don't have a meaningful
# "duration" -- period_start is NULL/absent, period_end is the as-of date.
# These are handled differently in get_annual_series (see is_point_in_time).
POINT_IN_TIME_METRICS = {"assets", "liabilities", "stockholders_equity", "cash", "long_term_debt"}


class FinancialDataError(Exception):
    """Raised when requested financial data cannot be found or is ambiguous."""


def _parse_date(value: str) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _is_annual_duration(period_start: Optional[str], period_end: Optional[str]) -> bool:
    """True if the gap between period_start and period_end is ~12 months."""
    start = _parse_date(period_start)
    end = _parse_date(period_end)
    if start is None or end is None:
        return False
    days = (end - start).days
    return MIN_ANNUAL_DAYS <= days <= MAX_ANNUAL_DAYS


def get_annual_series(
    metric_name: str,
    company_ticker: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> Dict[str, float]:
    """
    Return one clean value per fiscal-year-end date for a logical metric,
    e.g. "revenue" -> {"2016-01-31": 5010000000.0, "2017-01-29": 6910000000.0, ...}

    For flow metrics (revenue, income, cash flow -- anything with a real
    reporting period), only ~12-month periods are kept, and concepts are
    tried in METRIC_CONCEPTS priority order, taking the first with data
    for each period_end.

    For point-in-time metrics (assets, liabilities, equity, cash, debt),
    there is no "duration" to filter on -- these are balance-sheet
    snapshots as of period_end, so we take the most recently filed value
    for each distinct period_end instead.

    Keys are period_end dates (ISO strings), which double as the fiscal
    year identifier -- this deliberately avoids the unreliable fiscal_year
    column (see Phase 4 findings).
    """
    if metric_name not in METRIC_CONCEPTS:
        raise FinancialDataError(
            f"Unknown metric '{metric_name}'. Available: {list(METRIC_CONCEPTS.keys())}"
        )

    concepts = METRIC_CONCEPTS[metric_name]
    is_point_in_time = metric_name in POINT_IN_TIME_METRICS

    conn = get_connection()
    try:
        result: Dict[str, float] = {}
        result_source: Dict[str, str] = {}  # tracks which concept won, for debugging

        for concept in concepts:
            cursor = conn.execute(
                "SELECT period_start, period_end, value, filed_date, concept "
                "FROM metrics WHERE company_ticker = ? AND concept = ? AND form = '10-K' "
                "ORDER BY period_end, filed_date",
                (company_ticker, concept),
            )
            rows = cursor.fetchall()

            for row in rows:
                period_end = row["period_end"]
                if period_end in result:
                    continue  # already filled by a higher-priority concept

                if is_point_in_time:
                    # Balance-sheet snapshot: no duration check needed
                    result[period_end] = row["value"]
                    result_source[period_end] = concept
                else:
                    if _is_annual_duration(row["period_start"], row["period_end"]):
                        result[period_end] = row["value"]
                        result_source[period_end] = concept

        if start_year is not None or end_year is not None:
            filtered = {}
            for period_end, value in result.items():
                d = _parse_date(period_end)
                if d is None:
                    continue
                if start_year is not None and d.year < start_year:
                    continue
                if end_year is not None and d.year > end_year:
                    continue
                filtered[period_end] = value
            result = filtered

        log.info(
            "get_annual_series('%s', '%s'): %d periods found (sources: %s)",
            metric_name, company_ticker, len(result),
            set(result_source.values()) if result_source else "none",
        )
        return dict(sorted(result.items()))
    finally:
        conn.close()


def get_available_metrics() -> List[str]:
    """Return the list of logical metric names this module can extract."""
    return list(METRIC_CONCEPTS.keys())


def calculate_yoy_growth(series: Dict[str, float]) -> Dict[str, float]:
    """
    Year-over-year % growth between consecutive periods in a series
    (as returned by get_annual_series). Requires periods to be sorted
    chronologically, which get_annual_series already guarantees.
    """
    periods = sorted(series.keys())
    growth = {}
    for i in range(1, len(periods)):
        prev_period, curr_period = periods[i - 1], periods[i]
        prev_val, curr_val = series[prev_period], series[curr_period]
        if prev_val == 0:
            log.warning("Skipping YoY growth for %s: prior period value is zero", curr_period)
            continue
        growth[curr_period] = ((curr_val - prev_val) / abs(prev_val)) * 100
    return growth


def calculate_cagr(series: Dict[str, float]) -> Optional[float]:
    """
    Compound Annual Growth Rate across the full span of the series:
    CAGR = ((end/start)^(1/years) - 1) * 100
    Returns None if fewer than 2 periods, or if the start value is not
    positive (CAGR is undefined/misleading for negative or zero starts).
    """
    periods = sorted(series.keys())
    if len(periods) < 2:
        log.warning("Cannot compute CAGR: need at least 2 periods, got %d", len(periods))
        return None

    start_period, end_period = periods[0], periods[-1]
    start_val, end_val = series[start_period], series[end_period]

    if start_val <= 0:
        log.warning("Cannot compute CAGR: start value is not positive (%s)", start_val)
        return None

    years = (_parse_date(end_period) - _parse_date(start_period)).days / 365.25
    if years <= 0:
        return None

    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
    return cagr


def calculate_margins(company_ticker: str, start_year: int = None, end_year: int = None) -> Dict[str, Dict[str, float]]:
    """
    Gross/operating/net margin (%) per period, computed as metric/revenue.
    A period only appears in a margin's results if BOTH revenue and that
    metric are available for it -- no fabricated or interpolated values.
    """
    revenue = get_annual_series("revenue", company_ticker, start_year, end_year)
    gross_profit = get_annual_series("gross_profit", company_ticker, start_year, end_year)
    operating_income = get_annual_series("operating_income", company_ticker, start_year, end_year)
    net_income = get_annual_series("net_income", company_ticker, start_year, end_year)

    def margin_series(numerator: Dict[str, float]) -> Dict[str, float]:
        result = {}
        for period, rev in revenue.items():
            if period not in numerator:
                continue
            if rev == 0:
                log.warning("Skipping margin for %s: revenue is zero", period)
                continue
            result[period] = (numerator[period] / rev) * 100
        return result

    return {
        "gross_margin": margin_series(gross_profit),
        "operating_margin": margin_series(operating_income),
        "net_margin": margin_series(net_income),
    }


def calculate_fcf(company_ticker: str, start_year: int = None, end_year: int = None) -> Dict[str, float]:
    """
    Free Cash Flow = Operating Cash Flow - CapEx, per period.
    Only periods with BOTH values available are included.
    Note: capex is stored as a positive "payment" amount in XBRL, so we
    subtract it directly (not add a negative).

    KNOWN LIMITATION (verified during Phase 6 testing against real NVDA
    data): NVIDIA's XBRL filings tag PaymentsToAcquirePropertyPlantAndEquipment
    only in 10-Q filings, as year-to-date cumulative figures through Q1/Q2/Q3
    (max ~272 days). No filing tags a full ~365-day annual figure under this
    concept, so get_annual_series() correctly finds no qualifying annual
    periods and this function returns an empty dict for NVDA. This is a
    genuine gap in the source data, not a bug -- per project policy, we do
    not estimate or fabricate the missing annual figure. Downstream callers
    (e.g. the Valuation Agent's DCF) must handle an empty FCF series
    explicitly rather than assume data will be present.
    """
    ocf = get_annual_series("operating_cash_flow", company_ticker, start_year, end_year)
    capex = get_annual_series("capex", company_ticker, start_year, end_year)

    fcf = {}
    for period, ocf_val in ocf.items():
        if period not in capex:
            log.warning("Skipping FCF for %s: capex not available", period)
            continue
        fcf[period] = ocf_val - capex[period]
    return fcf


def calculate_roe_roa(company_ticker: str, start_year: int = None, end_year: int = None) -> Dict[str, Dict[str, float]]:
    """
    ROE = Net Income / Stockholders' Equity * 100
    ROA = Net Income / Total Assets * 100
    Both use point-in-time balance-sheet figures as of the SAME period_end
    as the net_income figure (i.e. equity/assets at fiscal year-end,
    matched to that year's income).
    """
    net_income = get_annual_series("net_income", company_ticker, start_year, end_year)
    equity = get_annual_series("stockholders_equity", company_ticker, start_year, end_year)
    assets = get_annual_series("assets", company_ticker, start_year, end_year)

    roe, roa = {}, {}
    for period, ni in net_income.items():
        if period in equity and equity[period] != 0:
            roe[period] = (ni / equity[period]) * 100
        else:
            log.warning("Skipping ROE for %s: equity unavailable or zero", period)

        if period in assets and assets[period] != 0:
            roa[period] = (ni / assets[period]) * 100
        else:
            log.warning("Skipping ROA for %s: assets unavailable or zero", period)

    return {"roe": roe, "roa": roa}


def build_financial_summary(company_ticker: str, start_year: int = None, end_year: int = None) -> Dict:
    """
    Full financial summary combining every calculation above. This is
    the main entry point Phase 8's Financial Agent will call.
    """
    revenue = get_annual_series("revenue", company_ticker, start_year, end_year)
    net_income = get_annual_series("net_income", company_ticker, start_year, end_year)

    return {
        "company_ticker": company_ticker,
        "revenue": revenue,
        "revenue_yoy_growth": calculate_yoy_growth(revenue),
        "revenue_cagr": calculate_cagr(revenue),
        "net_income": net_income,
        "net_income_yoy_growth": calculate_yoy_growth(net_income),
        "margins": calculate_margins(company_ticker, start_year, end_year),
        "free_cash_flow": calculate_fcf(company_ticker, start_year, end_year),
        "returns": calculate_roe_roa(company_ticker, start_year, end_year),
    }


if __name__ == "__main__":
    print("Testing get_annual_series() on real NVIDIA revenue data:\n")
    revenue = get_annual_series("revenue", "NVDA")
    for period_end, value in revenue.items():
        print(f"  {period_end}: ${value:,.0f}")

    print(f"\nTotal periods found: {len(revenue)}")

    missing_years = []
    years_present = {int(pe[:4]) for pe in revenue.keys()}
    for y in range(2016, 2027):
        if y not in years_present and (y + 1) not in years_present:
            # crude check -- fiscal year end dates can land in either
            # calendar year depending on the company, so this just flags
            # a gap for manual review, not a definitive verdict
            missing_years.append(y)
    if missing_years:
        print(f"Possible gaps (needs manual check): {missing_years}")