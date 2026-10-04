"""Tests for growth, margin, FCF, and return calculations (Phase 6 Part 2)."""

import pytest

from core.schemas import get_connection, initialize_schema
from tools.financial_tools import (
    calculate_cagr,
    calculate_fcf,
    calculate_margins,
    calculate_roe_roa,
    calculate_yoy_growth,
)


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


def test_yoy_growth_matches_hand_calculation():
    # 100 -> 150 is +50% growth
    series = {"2016-01-31": 100.0, "2017-01-29": 150.0}
    growth = calculate_yoy_growth(series)
    assert growth["2017-01-29"] == pytest.approx(50.0)


def test_yoy_growth_handles_decline():
    series = {"2016-01-31": 200.0, "2017-01-29": 150.0}
    growth = calculate_yoy_growth(series)
    assert growth["2017-01-29"] == pytest.approx(-25.0)


def test_yoy_growth_skips_zero_prior_value():
    series = {"2016-01-31": 0.0, "2017-01-29": 150.0}
    growth = calculate_yoy_growth(series)
    assert "2017-01-29" not in growth


def test_yoy_growth_single_period_returns_empty():
    series = {"2016-01-31": 100.0}
    assert calculate_yoy_growth(series) == {}


def test_cagr_matches_hand_calculation():
    # 100 -> 400 over exactly 2 years = 100% CAGR (doubles each year)
    series = {"2016-01-31": 100.0, "2018-01-30": 400.0}
    cagr = calculate_cagr(series)
    assert cagr == pytest.approx(100.0, abs=0.5)


def test_cagr_returns_none_for_single_period():
    assert calculate_cagr({"2016-01-31": 100.0}) is None


def test_cagr_returns_none_for_nonpositive_start():
    series = {"2016-01-31": 0.0, "2017-01-29": 100.0}
    assert calculate_cagr(series) is None


def test_margins_only_include_periods_with_both_values(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    conn = get_connection(tmp_path / "test.db")

    seed_metric(conn, "NVDA", "Revenues", 1000.0, "2015-02-01", "2016-01-31")
    seed_metric(conn, "NVDA", "Revenues", 2000.0, "2016-02-01", "2017-01-29")
    seed_metric(conn, "NVDA", "GrossProfit", 600.0, "2015-02-01", "2016-01-31")
    # No GrossProfit for 2017 -- that period should be excluded from gross_margin
    conn.commit()
    conn.close()

    margins = calculate_margins("NVDA")
    assert margins["gross_margin"]["2016-01-31"] == pytest.approx(60.0)
    assert "2017-01-29" not in margins["gross_margin"]


def test_fcf_subtracts_capex_correctly(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    conn = get_connection(tmp_path / "test.db")

    seed_metric(conn, "NVDA", "NetCashProvidedByUsedInOperatingActivities",
                1000.0, "2015-02-01", "2016-01-31")
    seed_metric(conn, "NVDA", "PaymentsToAcquirePropertyPlantAndEquipment",
                200.0, "2015-02-01", "2016-01-31")
    conn.commit()
    conn.close()

    fcf = calculate_fcf("NVDA")
    assert fcf["2016-01-31"] == pytest.approx(800.0)


def test_fcf_skips_period_missing_capex(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    conn = get_connection(tmp_path / "test.db")

    seed_metric(conn, "NVDA", "NetCashProvidedByUsedInOperatingActivities",
                1000.0, "2015-02-01", "2016-01-31")
    conn.commit()
    conn.close()

    fcf = calculate_fcf("NVDA")
    assert "2016-01-31" not in fcf


def test_roe_roa_matches_hand_calculation(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    conn = get_connection(tmp_path / "test.db")

    seed_metric(conn, "NVDA", "NetIncomeLoss", 100.0, "2015-02-01", "2016-01-31")
    seed_metric(conn, "NVDA", "StockholdersEquity", 500.0, None, "2016-01-31")
    seed_metric(conn, "NVDA", "Assets", 1000.0, None, "2016-01-31")
    conn.commit()
    conn.close()

    result = calculate_roe_roa("NVDA")
    assert result["roe"]["2016-01-31"] == pytest.approx(20.0)  # 100/500
    assert result["roa"]["2016-01-31"] == pytest.approx(10.0)  # 100/1000


def test_roe_skipped_when_equity_missing(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    conn = get_connection(tmp_path / "test.db")

    seed_metric(conn, "NVDA", "NetIncomeLoss", 100.0, "2015-02-01", "2016-01-31")
    conn.commit()
    conn.close()

    result = calculate_roe_roa("NVDA")
    assert "2016-01-31" not in result["roe"]
    assert "2016-01-31" not in result["roa"]



# ---- date parsing, available metrics, CAGR guard, zero-revenue margins ----
from unittest.mock import patch as _patch

from tools import financial_tools as ft


@pytest.mark.parametrize("value", [None, "", "not-a-date", "2024/01/28", "2024-13-45"])
def test_parse_date_returns_none_for_missing_or_malformed_values(value):
    assert ft._parse_date(value) is None


def test_parse_date_parses_iso_date():
    assert ft._parse_date("2024-01-28").isoformat() == "2024-01-28"


def test_is_annual_duration_false_when_either_date_missing_or_invalid():
    assert ft._is_annual_duration(None, "2024-01-28") is False
    assert ft._is_annual_duration("2023-01-28", None) is False
    assert ft._is_annual_duration("garbage", "2024-01-28") is False


def test_is_annual_duration_true_for_365_days_false_for_quarter():
    assert ft._is_annual_duration("2023-01-28", "2024-01-28") is True
    assert ft._is_annual_duration("2023-10-30", "2024-01-28") is False


def test_get_available_metrics_lists_every_configured_metric():
    metrics = ft.get_available_metrics()
    assert metrics == list(ft.METRIC_CONCEPTS.keys())
    assert "revenue" in metrics


def test_cagr_returns_none_when_both_dates_are_the_same_day():
    # Different strings, same calendar date: years == 0
    series = {"2024-01-05": 100.0, "2024-1-05": 200.0}
    assert ft.calculate_cagr(series) is None


def test_margins_skip_zero_revenue_period_but_keep_others():
    data = {
        "revenue": {"2023-01-29": 0.0, "2024-01-28": 200.0},
        "gross_profit": {"2023-01-29": 10.0, "2024-01-28": 120.0},
        "operating_income": {"2023-01-29": 5.0, "2024-01-28": 80.0},
        "net_income": {"2023-01-29": 4.0, "2024-01-28": 50.0},
    }
    with _patch.object(ft, "get_annual_series", side_effect=lambda name, ticker, *a: data[name]):
        out = ft.calculate_margins("NVDA")

    assert out["gross_margin"] == {"2024-01-28": pytest.approx(60.0)}
    assert out["net_margin"] == {"2024-01-28": pytest.approx(25.0)}



# ---- build_financial_summary wiring ----

def test_build_financial_summary_assembles_every_section_and_passes_year_range():
    series = {
        "revenue": {"2023-01-29": 100.0, "2024-01-28": 200.0},
        "net_income": {"2023-01-29": 10.0, "2024-01-28": 40.0},
    }
    with _patch.object(ft, "get_annual_series", side_effect=lambda name, t, s=None, e=None: series[name]), \
         _patch.object(ft, "calculate_margins", return_value={"m": 1}) as mock_margins, \
         _patch.object(ft, "calculate_fcf", return_value={"f": 2}) as mock_fcf, \
         _patch.object(ft, "calculate_roe_roa", return_value={"roe": {}, "roa": {}}) as mock_ret:
        out = ft.build_financial_summary("NVDA", start_year=2020, end_year=2025)

    assert set(out) == {
        "company_ticker", "revenue", "revenue_yoy_growth", "revenue_cagr",
        "net_income", "net_income_yoy_growth", "margins", "free_cash_flow", "returns",
    }
    assert out["company_ticker"] == "NVDA"
    assert out["revenue"] == series["revenue"]
    assert out["revenue_yoy_growth"] == ft.calculate_yoy_growth(series["revenue"])
    assert out["revenue_cagr"] == ft.calculate_cagr(series["revenue"])
    assert out["net_income_yoy_growth"] == ft.calculate_yoy_growth(series["net_income"])
    assert out["margins"] == {"m": 1}
    assert out["free_cash_flow"] == {"f": 2}
    assert out["returns"] == {"roe": {}, "roa": {}}
    mock_margins.assert_called_once_with("NVDA", 2020, 2025)
    mock_fcf.assert_called_once_with("NVDA", 2020, 2025)
    mock_ret.assert_called_once_with("NVDA", 2020, 2025)
