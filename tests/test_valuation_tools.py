"""Tests for valuation engine: revenue forecast, FCF proxy, DCF math."""

from unittest.mock import patch

import pytest

from tools.valuation_tools import (
    calculate_fcf_proxy_margin,
    forecast_revenue,
    run_dcf,
)


def test_forecast_revenue_compounds_correctly_with_given_rate():
    fake_revenue = {"2024-01-28": 100.0}
    with patch("tools.valuation_tools.get_annual_series", return_value=fake_revenue):
        forecast = forecast_revenue("NVDA", forecast_years=3, growth_rate=0.10)

    # 100 * 1.1 = 110, * 1.1 = 121, * 1.1 = 133.1
    assert forecast["Year 1"] == pytest.approx(110.0)
    assert forecast["Year 2"] == pytest.approx(121.0)
    assert forecast["Year 3"] == pytest.approx(133.1)


def test_forecast_revenue_uses_historical_cagr_when_no_rate_given():
    fake_revenue = {"2016-01-31": 100.0, "2018-01-30": 400.0}  # 2 years, 100% CAGR
    with patch("tools.valuation_tools.get_annual_series", return_value=fake_revenue):
        forecast = forecast_revenue("NVDA", forecast_years=1)

    # 400 * (1 + 1.00) = 800, approximately (CAGR ~100%)
    assert forecast["Year 1"] == pytest.approx(800.0, rel=0.01)


def test_forecast_revenue_raises_on_no_data():
    with patch("tools.valuation_tools.get_annual_series", return_value={}):
        with pytest.raises(ValueError):
            forecast_revenue("NVDA")


def test_fcf_proxy_margin_averages_correctly():
    fake_revenue = {"2023-01-29": 100.0, "2024-01-28": 200.0}
    fake_ocf = {"2023-01-29": 20.0, "2024-01-28": 60.0}  # margins: 0.20, 0.30

    def fake_series(metric, ticker, start_year=None, end_year=None):
        return fake_revenue if metric == "revenue" else fake_ocf

    with patch("tools.valuation_tools.get_annual_series", side_effect=fake_series):
        margin = calculate_fcf_proxy_margin("NVDA")

    assert margin == pytest.approx(0.25)  # average of 0.20 and 0.30


def test_fcf_proxy_margin_returns_none_with_no_overlap():
    def fake_series(metric, ticker, start_year=None, end_year=None):
        if metric == "revenue":
            return {"2020-01-01": 100.0}
        return {"2025-01-01": 50.0}  # no overlapping period

    with patch("tools.valuation_tools.get_annual_series", side_effect=fake_series):
        margin = calculate_fcf_proxy_margin("NVDA")

    assert margin is None


def test_run_dcf_present_value_matches_hand_calculation():
    fake_revenue = {"2024-01-28": 100.0}
    fake_ocf = {"2024-01-28": 20.0}  # margin = 0.20

    def fake_series(metric, ticker, start_year=None, end_year=None):
        return fake_revenue if metric == "revenue" else fake_ocf

    with patch("tools.valuation_tools.get_annual_series", side_effect=fake_series):
        result = run_dcf(
            "NVDA", wacc=0.10, terminal_growth=0.02,
            forecast_years=2, revenue_growth_rate=0.0,  # flat revenue for simple math
        )

    # Revenue stays 100 each year (0% growth), FCF proxy = 100 * 0.20 = 20 each year
    # Year 1 PV = 20 / 1.10^1 = 18.1818...
    # Year 2 PV = 20 / 1.10^2 = 16.5289...
    assert result["present_values"]["Year 1"] == pytest.approx(20 / 1.10, rel=1e-4)
    assert result["present_values"]["Year 2"] == pytest.approx(20 / (1.10 ** 2), rel=1e-4)


def test_run_dcf_terminal_value_matches_gordon_growth_formula():
    fake_revenue = {"2024-01-28": 100.0}
    fake_ocf = {"2024-01-28": 20.0}

    def fake_series(metric, ticker, start_year=None, end_year=None):
        return fake_revenue if metric == "revenue" else fake_ocf

    with patch("tools.valuation_tools.get_annual_series", side_effect=fake_series):
        result = run_dcf(
            "NVDA", wacc=0.10, terminal_growth=0.03,
            forecast_years=1, revenue_growth_rate=0.0,
        )

    # final_year_fcf = 20; TV = 20 * 1.03 / (0.10 - 0.03) = 20.6 / 0.07
    expected_tv = 20 * 1.03 / 0.07
    assert result["terminal_value"] == pytest.approx(expected_tv, rel=1e-4)


def test_run_dcf_raises_when_wacc_not_greater_than_terminal_growth():
    fake_revenue = {"2024-01-28": 100.0}
    fake_ocf = {"2024-01-28": 20.0}

    def fake_series(metric, ticker, start_year=None, end_year=None):
        return fake_revenue if metric == "revenue" else fake_ocf

    with patch("tools.valuation_tools.get_annual_series", side_effect=fake_series):
        with pytest.raises(ValueError):
            run_dcf("NVDA", wacc=0.03, terminal_growth=0.03)  # equal, invalid


def test_run_dcf_includes_fcf_disclaimer():
    fake_revenue = {"2024-01-28": 100.0}
    fake_ocf = {"2024-01-28": 20.0}

    def fake_series(metric, ticker, start_year=None, end_year=None):
        return fake_revenue if metric == "revenue" else fake_ocf

    with patch("tools.valuation_tools.get_annual_series", side_effect=fake_series):
        result = run_dcf(
            "NVDA", wacc=0.10, terminal_growth=0.03, forecast_years=1,
            revenue_growth_rate=0.0,  # single-period fake data can't compute CAGR
        )

    assert "CapEx" in result["fcf_disclaimer"]
    assert "OVERSTATES" in result["fcf_disclaimer"]


def test_run_dcf_enterprise_value_is_sum_of_pv_plus_pv_terminal():
    fake_revenue = {"2024-01-28": 100.0}
    fake_ocf = {"2024-01-28": 20.0}

    def fake_series(metric, ticker, start_year=None, end_year=None):
        return fake_revenue if metric == "revenue" else fake_ocf

    with patch("tools.valuation_tools.get_annual_series", side_effect=fake_series):
        result = run_dcf(
            "NVDA", wacc=0.10, terminal_growth=0.03, forecast_years=2,
            revenue_growth_rate=0.0,  # single-period fake data can't compute CAGR
        )

    expected = result["sum_pv_explicit_period"] + result["pv_terminal_value"]
    assert result["enterprise_value"] == pytest.approx(expected)
