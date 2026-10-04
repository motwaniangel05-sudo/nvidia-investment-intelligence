"""Tests for valuation engine: revenue forecast, FCF proxy, DCF math."""

from unittest.mock import MagicMock, patch

import pytest

from tools.valuation_tools import (
    calculate_ev_ebit,
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


def test_calculate_ev_ebit_computes_correctly():
    fake_op_income = {"2024-01-28": 100.0}
    fake_net_income = {"2024-01-28": 80.0}
    fake_eps = {"2024-01-28": 2.0}  # implies 80/2 = 40 shares
    fake_debt = {"2024-01-28": 50.0}
    fake_cash = {"2024-01-28": 30.0}

    def fake_series(metric, ticker, start_year=None, end_year=None):
        return {
            "operating_income": fake_op_income,
            "net_income": fake_net_income,
            "eps_diluted": fake_eps,
            "long_term_debt": fake_debt,
            "cash": fake_cash,
        }[metric]

    fake_conn = MagicMock()
    fake_conn.execute.return_value.fetchone.return_value = {"date": "2024-01-28", "close": 10.0}

    with patch("tools.valuation_tools.get_annual_series", side_effect=fake_series), \
         patch("tools.valuation_tools.get_connection", return_value=fake_conn):
        result = calculate_ev_ebit("NVDA")

    # market_cap = 10.0 * 40 shares = 400
    # EV = 400 + 50 debt - 30 cash = 420
    # EV/EBIT = 420 / 100 = 4.2
    assert result["implied_shares_outstanding"] == pytest.approx(40.0)
    assert result["market_cap"] == pytest.approx(400.0)
    assert result["enterprise_value"] == pytest.approx(420.0)
    assert result["ev_ebit_ratio"] == pytest.approx(4.2)


def test_calculate_ev_ebit_returns_none_with_missing_data():
    with patch("tools.valuation_tools.get_annual_series", return_value={}):
        result = calculate_ev_ebit("NVDA")
    assert result is None


def test_calculate_ev_ebit_returns_none_for_nonpositive_ebit():
    fake_op_income = {"2024-01-28": -50.0}  # negative EBIT
    fake_net_income = {"2024-01-28": 80.0}
    fake_eps = {"2024-01-28": 2.0}

    def fake_series(metric, ticker, start_year=None, end_year=None):
        return {
            "operating_income": fake_op_income,
            "net_income": fake_net_income,
            "eps_diluted": fake_eps,
            "long_term_debt": {},
            "cash": {},
        }[metric]

    fake_conn = MagicMock()
    fake_conn.execute.return_value.fetchone.return_value = {"date": "2024-01-28", "close": 10.0}

    with patch("tools.valuation_tools.get_annual_series", side_effect=fake_series), \
         patch("tools.valuation_tools.get_connection", return_value=fake_conn):
        result = calculate_ev_ebit("NVDA")

    assert result is None


def test_calculate_ev_ebit_note_clarifies_not_ebitda():
    fake_op_income = {"2024-01-28": 100.0}
    fake_net_income = {"2024-01-28": 80.0}
    fake_eps = {"2024-01-28": 2.0}

    def fake_series(metric, ticker, start_year=None, end_year=None):
        return {
            "operating_income": fake_op_income,
            "net_income": fake_net_income,
            "eps_diluted": fake_eps,
            "long_term_debt": {},
            "cash": {},
        }[metric]

    fake_conn = MagicMock()
    fake_conn.execute.return_value.fetchone.return_value = {"date": "2024-01-28", "close": 10.0}

    with patch("tools.valuation_tools.get_annual_series", side_effect=fake_series), \
         patch("tools.valuation_tools.get_connection", return_value=fake_conn):
        result = calculate_ev_ebit("NVDA")

    assert "EBITDA" in result["note"]
    assert "APPROXIMATED" in result["note"]


# ---- scenario / sensitivity / P/E coverage ----
from tools import valuation_tools as vt


def _fake_dcf(*args, **kwargs):
    return {
        "enterprise_value": 1000.0,
        "assumptions": {"revenue_growth_rate_used": kwargs["revenue_growth_rate"]},
    }


# run_dcf_scenarios

def test_scenarios_resolve_historical_cagr_before_calling_run_dcf():
    with patch.object(vt, "get_annual_series", return_value={"2020": 100.0, "2024": 200.0}), \
         patch.object(vt, "calculate_cagr", return_value=25.0), \
         patch.object(vt, "run_dcf", side_effect=_fake_dcf) as mock_dcf:
        out = vt.run_dcf_scenarios("NVDA")

    assert set(out) == set(vt.GROWTH_SCENARIOS)
    assert out["historical_cagr"]["assumptions"]["revenue_growth_rate_used"] == 0.25
    assert out["historical_cagr"]["scenario_is_historical_extrapolation"] is True
    for name, result in out.items():
        assert result["scenario_name"] == name
        if name != "historical_cagr":
            assert result["scenario_is_historical_extrapolation"] is False
    assert mock_dcf.call_count == len(vt.GROWTH_SCENARIOS)


def test_scenarios_skip_historical_when_cagr_unavailable():
    with patch.object(vt, "get_annual_series", return_value={}), \
         patch.object(vt, "calculate_cagr", return_value=None), \
         patch.object(vt, "run_dcf", side_effect=_fake_dcf):
        out = vt.run_dcf_scenarios("NVDA")

    assert "historical_cagr" not in out
    assert set(out) == set(vt.GROWTH_SCENARIOS) - {"historical_cagr"}


def test_scenarios_skip_a_scenario_whose_dcf_raises_valueerror():
    def flaky(*args, **kwargs):
        if kwargs["revenue_growth_rate"] == vt.GROWTH_SCENARIOS["aggressive"]:
            raise ValueError("boom")
        return _fake_dcf(*args, **kwargs)

    with patch.object(vt, "get_annual_series", return_value={"2020": 100.0, "2024": 200.0}), \
         patch.object(vt, "calculate_cagr", return_value=25.0), \
         patch.object(vt, "run_dcf", side_effect=flaky):
        out = vt.run_dcf_scenarios("NVDA")

    assert "aggressive" not in out
    assert "historical_cagr" in out


def test_scenarios_pass_wacc_terminal_growth_and_years_through():
    with patch.object(vt, "get_annual_series", return_value={"2020": 100.0, "2024": 200.0}), \
         patch.object(vt, "calculate_cagr", return_value=25.0), \
         patch.object(vt, "run_dcf", side_effect=_fake_dcf) as mock_dcf:
        vt.run_dcf_scenarios("NVDA", wacc=0.09, terminal_growth=0.02, forecast_years=7)

    for call in mock_dcf.call_args_list:
        assert call.kwargs["wacc"] == 0.09
        assert call.kwargs["terminal_growth"] == 0.02
        assert call.kwargs["forecast_years"] == 7


# sensitivity_analysis

def test_sensitivity_grid_shape_labels_and_values():
    def fake(company_ticker, wacc, terminal_growth, forecast_years, revenue_growth_rate):
        return {"enterprise_value": round(wacc * 1000 - terminal_growth * 100, 6)}

    with patch.object(vt, "run_dcf", side_effect=fake):
        grid = vt.sensitivity_analysis(
            "NVDA", growth_rate=0.2, wacc_range=[0.08, 0.10], terminal_growth_range=[0.03, 0.10],
        )

    assert list(grid) == ["8.0%", "10.0%"]
    assert list(grid["8.0%"]) == ["3.0%", "10.0%"]
    assert grid["8.0%"]["3.0%"] == pytest.approx(77.0)
    assert grid["10.0%"]["3.0%"] == pytest.approx(97.0)


def test_sensitivity_marks_wacc_not_above_terminal_growth_as_none_without_calling_dcf():
    with patch.object(vt, "run_dcf", return_value={"enterprise_value": 1.0}) as mock_dcf:
        grid = vt.sensitivity_analysis(
            "NVDA", growth_rate=0.2, wacc_range=[0.08, 0.10], terminal_growth_range=[0.03, 0.10],
        )

    assert grid["8.0%"]["10.0%"] is None    # wacc < tg
    assert grid["10.0%"]["10.0%"] is None   # wacc == tg
    assert mock_dcf.call_count == 2          # only the two valid combinations


def test_sensitivity_records_none_when_dcf_raises():
    with patch.object(vt, "run_dcf", side_effect=ValueError("bad")):
        grid = vt.sensitivity_analysis(
            "NVDA", growth_rate=0.2, wacc_range=[0.10], terminal_growth_range=[0.03],
        )
    assert grid == {"10.0%": {"3.0%": None}}


def test_sensitivity_uses_default_ranges_when_none_given():
    with patch.object(vt, "run_dcf", return_value={"enterprise_value": 1.0}):
        grid = vt.sensitivity_analysis("NVDA", growth_rate=0.2)
    assert list(grid) == [f"{w:.1%}" for w in vt.DEFAULT_WACC_RANGE]
    first = next(iter(grid.values()))
    assert list(first) == [f"{t:.1%}" for t in vt.DEFAULT_TERMINAL_GROWTH_RANGE]


# calculate_pe_ratio

def _fake_conn(row):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = row
    return conn


def test_pe_ratio_uses_latest_eps_and_latest_price():
    conn = _fake_conn({"date": "2024-12-31", "close": 120.0})
    with patch.object(vt, "get_annual_series", return_value={"2023-01-29": 2.0, "2024-01-28": 4.0}), \
         patch.object(vt, "get_connection", return_value=conn):
        out = vt.calculate_pe_ratio("NVDA")

    assert out["pe_ratio"] == pytest.approx(30.0)
    assert out["eps"] == 4.0
    assert out["eps_period"] == "2024-01-28"
    assert out["price"] == 120.0
    assert out["price_date"] == "2024-12-31"
    assert "period-matched" in out["note"]
    conn.close.assert_called_once()


def test_pe_ratio_none_when_no_eps_data():
    with patch.object(vt, "get_annual_series", return_value={}):
        assert vt.calculate_pe_ratio("NVDA") is None


def test_pe_ratio_none_for_nonpositive_eps_without_touching_db():
    with patch.object(vt, "get_annual_series", return_value={"2024-01-28": -1.5}), \
         patch.object(vt, "get_connection") as mock_conn:
        assert vt.calculate_pe_ratio("NVDA") is None
    mock_conn.assert_not_called()


def test_pe_ratio_none_when_no_price_row_and_connection_still_closed():
    conn = _fake_conn(None)
    with patch.object(vt, "get_annual_series", return_value={"2024-01-28": 4.0}), \
         patch.object(vt, "get_connection", return_value=conn):
        assert vt.calculate_pe_ratio("NVDA") is None
    conn.close.assert_called_once()


# ---- run_dcf error branches ----

def test_run_dcf_raises_when_fcf_margin_unavailable():
    with patch.object(vt, "forecast_revenue", return_value={"2025": 100.0, "2026": 200.0}), \
         patch.object(vt, "calculate_fcf_proxy_margin", return_value=None):
        with pytest.raises(ValueError, match="FCF-proxy margin"):
            vt.run_dcf("NVDA", wacc=0.10, terminal_growth=0.03)


@pytest.mark.parametrize("wacc, tg", [(0.03, 0.03), (0.02, 0.03)])
def test_run_dcf_wacc_check_is_reached_with_valid_margin(wacc, tg):
    with patch.object(vt, "forecast_revenue", return_value={"2025": 100.0, "2026": 200.0}), \
         patch.object(vt, "calculate_fcf_proxy_margin", return_value=0.5):
        with pytest.raises(ValueError, match="must exceed"):
            vt.run_dcf("NVDA", wacc=wacc, terminal_growth=tg)


# ---- calculate_ev_ebit early exits ----

def _series_for(**by_name):
    return lambda name, ticker: by_name.get(name, {})


def test_ev_ebit_none_when_latest_period_missing_from_net_income():
    series = _series_for(
        operating_income={"2024": 100.0},
        net_income={"2023": 80.0},
        eps_diluted={"2024": 2.0},
    )
    with patch.object(vt, "get_annual_series", side_effect=series), \
         patch.object(vt, "get_connection") as mock_conn:
        assert vt.calculate_ev_ebit("NVDA") is None
    mock_conn.assert_not_called()


def test_ev_ebit_none_when_latest_period_missing_from_eps():
    series = _series_for(
        operating_income={"2024": 100.0},
        net_income={"2024": 80.0},
        eps_diluted={"2023": 2.0},
    )
    with patch.object(vt, "get_annual_series", side_effect=series), \
         patch.object(vt, "get_connection") as mock_conn:
        assert vt.calculate_ev_ebit("NVDA") is None
    mock_conn.assert_not_called()


def test_ev_ebit_none_when_eps_is_zero_instead_of_dividing():
    series = _series_for(
        operating_income={"2024": 100.0},
        net_income={"2024": 80.0},
        eps_diluted={"2024": 0.0},
    )
    with patch.object(vt, "get_annual_series", side_effect=series), \
         patch.object(vt, "get_connection") as mock_conn:
        assert vt.calculate_ev_ebit("NVDA") is None
    mock_conn.assert_not_called()


def test_ev_ebit_none_when_no_price_row_and_connection_closed():
    series = _series_for(
        operating_income={"2024": 100.0},
        net_income={"2024": 80.0},
        eps_diluted={"2024": 2.0},
        long_term_debt={"2024": 10.0},
        cash={"2024": 5.0},
    )
    conn = _fake_conn(None)
    with patch.object(vt, "get_annual_series", side_effect=series), \
         patch.object(vt, "get_connection", return_value=conn):
        assert vt.calculate_ev_ebit("NVDA") is None
    conn.close.assert_called_once()
