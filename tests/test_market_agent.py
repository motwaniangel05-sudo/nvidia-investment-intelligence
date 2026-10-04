"""Tests for MarketAgent: competitor comparison logic."""

from unittest.mock import patch

from agents.market_agent import MarketAgent, _year_indexed


FAKE_CONFIG = {
    "competitors": [
        {"name": "Test Competitor", "ticker": "TCOMP", "reason": "Test rationale for inclusion"},
    ]
}


def test_year_indexed_extracts_calendar_year():
    series = {"2016-01-31": 100.0, "2017-01-29": 150.0}
    result = _year_indexed(series)
    assert result == {2016: 100.0, 2017: 150.0}


def fake_get_annual_series(metric_name, ticker, start_year=None, end_year=None):
    if metric_name != "revenue":
        return {}
    if ticker == "NVDA":
        return {"2016-01-31": 5000000000.0, "2026-01-25": 216000000000.0}
    if ticker == "TCOMP":
        return {"2016-12-31": 4000000000.0, "2026-12-31": 34000000000.0}
    return {}


def test_market_agent_produces_inclusion_rationale():
    agent = MarketAgent("NVDA")
    with patch("agents.market_agent.load_config", return_value=FAKE_CONFIG), \
         patch("agents.market_agent.get_annual_series", side_effect=fake_get_annual_series):
        result = agent.run("Compare competitors")

    rationale_findings = [f for f in result.findings if "rationale" in f.evidence_text.lower()]
    assert len(rationale_findings) == 1
    assert "Test rationale for inclusion" in rationale_findings[0].claim


def test_market_agent_cagr_comparison_direction():
    agent = MarketAgent("NVDA")
    with patch("agents.market_agent.load_config", return_value=FAKE_CONFIG), \
         patch("agents.market_agent.get_annual_series", side_effect=fake_get_annual_series):
        result = agent.run("Compare competitors")

    cagr_findings = [f for f in result.findings if "CAGR" in f.claim]
    assert len(cagr_findings) == 1
    # NVDA grows from 5B to 216B (huge CAGR); TCOMP from 4B to 34B (smaller)
    assert "faster" in cagr_findings[0].claim


def test_market_agent_no_competitors_returns_partial():
    agent = MarketAgent("NVDA")
    with patch("agents.market_agent.load_config", return_value={"competitors": []}):
        result = agent.run("Compare competitors")

    assert result.status == "partial"
    # Two warnings expected: analyze()'s specific "no competitors configured"
    # plus BaseAgent.verify()'s generic "no findings" check -- same layered
    # pattern as ResearchAgent's equivalent test (see Phase 7 notes).
    assert len(result.warnings) == 2
    assert any("No competitors configured" in w for w in result.warnings)
    assert any("No findings were produced" in w for w in result.warnings)


def test_market_agent_missing_competitor_data_warns():
    agent = MarketAgent("NVDA")
    config = {"competitors": [{"name": "NoData Corp", "ticker": "NODATA", "reason": "test"}]}

    def no_data_series(metric_name, ticker, start_year=None, end_year=None):
        if ticker == "NVDA":
            return {"2016-01-31": 5000000000.0}
        return {}

    with patch("agents.market_agent.load_config", return_value=config), \
         patch("agents.market_agent.get_annual_series", side_effect=no_data_series):
        result = agent.run("Compare competitors")

    assert any("No revenue data available for competitor NODATA" in w for w in result.warnings)


def test_market_agent_target_missing_data_warns():
    agent = MarketAgent("NVDA")
    def empty_series(metric_name, ticker, start_year=None, end_year=None):
        return {}

    with patch("agents.market_agent.load_config", return_value=FAKE_CONFIG), \
         patch("agents.market_agent.get_annual_series", side_effect=empty_series):
        result = agent.run("Compare competitors")

    assert any("No revenue data available for target company NVDA" in w for w in result.warnings)



# ---- retrieve is unused ----
from agents.market_agent import MarketAgent as _MarketAgent


def test_retrieve_is_unused_and_returns_empty_list():
    assert _MarketAgent("NVDA").retrieve("anything") == []


# ---- calendar-year overlap handling ----

def _run_market(target_series, comp_series, config=None):
    def series(metric_name, ticker, start_year=None, end_year=None):
        return target_series if ticker == "NVDA" else comp_series

    with patch("agents.market_agent.load_config", return_value=config or FAKE_CONFIG), \
         patch("agents.market_agent.get_annual_series", side_effect=series):
        return _MarketAgent("NVDA").run("Compare competitors")


def test_no_overlapping_calendar_years_warns_and_skips_revenue_ratio():
    result = _run_market(
        {"2016-01-31": 5e9, "2026-01-25": 216e9},
        {"2018-12-31": 4e9, "2020-12-31": 34e9},
    )

    assert any(
        "No overlapping calendar years found between NVDA and TCOMP" in w
        for w in result.warnings
    )
    assert not any("x TCOMP's" in f.claim for f in result.findings)
    assert result.metrics["competitors_compared"] == 1


def test_overlapping_year_produces_revenue_ratio_finding_for_latest_common_year():
    result = _run_market(
        {"2016-01-31": 5e9, "2026-01-25": 216e9},
        {"2016-12-31": 4e9, "2026-12-31": 54e9},
    )

    f = next(x for x in result.findings if x.source_chunk_id == "NVDA_vs_TCOMP_revenue_2026")
    assert "calendar 2026" in f.claim
    assert "4.0x TCOMP's" in f.claim  # 216e9 / 54e9
    assert f.source_form == "XBRL_metrics"
    assert f.source_filing_date == "2026"
    assert f.confidence == 1.0
    assert not any("No overlapping" in w for w in result.warnings)


def test_zero_competitor_revenue_suppresses_ratio_finding_without_warning():
    result = _run_market(
        {"2016-01-31": 5e9, "2026-01-25": 216e9},
        {"2016-12-31": 4e9, "2026-12-31": 0.0},
    )

    assert not any(x.source_chunk_id.endswith("_revenue_2026") for x in result.findings)
    assert not any("No overlapping" in w for w in result.warnings)


def test_cagr_comparison_says_slower_when_target_grows_less():
    result = _run_market(
        {"2016-01-31": 100e9, "2026-01-25": 120e9},
        {"2016-12-31": 4e9, "2026-12-31": 34e9},
    )

    cagr = next(f for f in result.findings if "CAGR" in f.claim)
    assert "slower" in cagr.claim
