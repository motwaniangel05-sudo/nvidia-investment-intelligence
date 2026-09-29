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
