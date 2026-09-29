"""Tests for the financial analysis engine's core annual series extraction."""

import pytest

from core.schemas import get_connection, initialize_schema
from tools.financial_tools import FinancialDataError, get_annual_series


def seed_metric(conn, ticker, concept, value, period_start, period_end, form="10-K"):
    conn.execute(
        "INSERT INTO metrics (company_ticker, concept, unit, value, "
        "period_start, period_end, form) VALUES (?, ?, 'USD', ?, ?, ?, ?)",
        (ticker, concept, value, period_start, period_end, form),
    )


def make_test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    monkeypatch.setattr("tools.financial_tools.get_connection", lambda: get_connection(db_path))
    return db_path


def test_keeps_only_annual_duration_periods(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    conn = get_connection(tmp_path / "test.db")

    # Annual: ~365 days
    seed_metric(conn, "NVDA", "Revenues", 5000000000, "2015-02-01", "2016-01-31")
    # Quarterly: ~90 days, same period_end as nothing else here
    seed_metric(conn, "NVDA", "Revenues", 1400000000, "2015-10-26", "2016-01-31")
    # Wait -- these share period_end, so this actually tests dedup too.
    # Add a distinct quarterly period to isolate the duration filter:
    seed_metric(conn, "NVDA", "Revenues", 1300000000, "2016-02-01", "2016-04-30")
    conn.commit()
    conn.close()

    result = get_annual_series("revenue", "NVDA")

    # The quarterly-only period (2016-04-30) should be excluded entirely
    assert "2016-04-30" not in result
    # The annual period should be present with the ANNUAL value (first row
    # inserted for that period_end wins, per our first-seen-wins rule)
    assert result["2016-01-31"] == 5000000000


def test_tag_switch_fallback_fills_gap(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    conn = get_connection(tmp_path / "test.db")

    # "Revenues" covers 2016 and 2017 only
    seed_metric(conn, "NVDA", "Revenues", 5000000000, "2015-02-01", "2016-01-31")
    seed_metric(conn, "NVDA", "Revenues", 7000000000, "2016-02-01", "2017-01-29")
    # The alternate tag covers 2018 (the "gap" year for Revenues)
    seed_metric(conn, "NVDA", "RevenueFromContractWithCustomerExcludingAssessedTax",
                9000000000, "2017-01-30", "2018-01-28")
    conn.commit()
    conn.close()

    result = get_annual_series("revenue", "NVDA")

    assert len(result) == 3
    assert result["2016-01-31"] == 5000000000
    assert result["2017-01-29"] == 7000000000
    assert result["2018-01-28"] == 9000000000  # filled by fallback concept


def test_primary_concept_takes_priority_over_fallback(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    conn = get_connection(tmp_path / "test.db")

    # Both concepts report a value for the SAME period_end -- primary wins
    seed_metric(conn, "NVDA", "Revenues", 9999000000, "2017-01-30", "2018-01-28")
    seed_metric(conn, "NVDA", "RevenueFromContractWithCustomerExcludingAssessedTax",
                1111000000, "2017-01-30", "2018-01-28")
    conn.commit()
    conn.close()

    result = get_annual_series("revenue", "NVDA")
    assert result["2018-01-28"] == 9999000000  # "Revenues" (primary) wins


def test_duplicate_period_end_first_seen_wins(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    conn = get_connection(tmp_path / "test.db")

    # Same concept, same period_end reported twice (prior-year comparison
    # in two different filings) -- should not error, first row wins
    seed_metric(conn, "NVDA", "Revenues", 5000000000, "2015-02-01", "2016-01-31")
    seed_metric(conn, "NVDA", "Revenues", 5000000000, "2015-02-01", "2016-01-31")
    conn.commit()
    conn.close()

    result = get_annual_series("revenue", "NVDA")
    assert len(result) == 1
    assert result["2016-01-31"] == 5000000000


def test_point_in_time_metric_skips_duration_filter(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    conn = get_connection(tmp_path / "test.db")

    # Balance sheet items have no real "period" -- period_start is often
    # NULL/empty. This should NOT be filtered out just because it lacks
    # a valid 12-month duration.
    seed_metric(conn, "NVDA", "Assets", 7370000000, None, "2016-01-31")
    conn.commit()
    conn.close()

    result = get_annual_series("assets", "NVDA")
    assert result["2016-01-31"] == 7370000000


def test_year_range_filtering(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    conn = get_connection(tmp_path / "test.db")

    seed_metric(conn, "NVDA", "Revenues", 5000000000, "2015-02-01", "2016-01-31")
    seed_metric(conn, "NVDA", "Revenues", 26900000000, "2021-02-01", "2022-01-30")
    conn.commit()
    conn.close()

    result = get_annual_series("revenue", "NVDA", start_year=2020, end_year=2025)
    assert "2016-01-31" not in result
    assert "2022-01-30" in result


def test_unknown_metric_raises_clear_error(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    with pytest.raises(FinancialDataError):
        get_annual_series("totally_made_up_metric", "NVDA")


def test_no_data_returns_empty_dict_not_error(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    result = get_annual_series("revenue", "NVDA_NONEXISTENT_TICKER")
    assert result == {}


def test_only_10k_form_considered(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    conn = get_connection(tmp_path / "test.db")

    # A 10-Q also reporting an annual-duration figure should NOT be picked
    # up -- we deliberately restrict to 10-K for the authoritative annual
    # figure, since 10-Qs report trailing/comparative data less consistently
    seed_metric(conn, "NVDA", "Revenues", 999000000, "2015-02-01", "2016-01-31", form="10-Q")
    conn.commit()
    conn.close()

    result = get_annual_series("revenue", "NVDA")
    assert result == {}
