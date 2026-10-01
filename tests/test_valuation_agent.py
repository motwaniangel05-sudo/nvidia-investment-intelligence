"""Tests for ValuationAgent: Finding construction from valuation_tools output."""

from unittest.mock import patch

from agents.valuation_agent import ValuationAgent


def fake_scenarios():
    return {
        "conservative": {
            "enterprise_value": 1_000_000_000_000.0,
            "assumptions": {"revenue_growth_rate_used": 0.10},
            "fcf_disclaimer": "FCF proxy disclaimer text.",
            "scenario_is_historical_extrapolation": False,
        },
        "historical_cagr": {
            "enterprise_value": 4_800_000_000_000.0,
            "assumptions": {"revenue_growth_rate_used": None},
            "fcf_disclaimer": "FCF proxy disclaimer text.",
            "scenario_is_historical_extrapolation": True,
        },
    }


def fake_pe():
    return {
        "price": 228.86, "price_date": "2026-09-28",
        "eps": 4.90, "eps_period": "2026-01-25",
        "pe_ratio": 46.7, "note": "P/E note text.",
    }


def fake_ev_ebit():
    return {
        "ev_ebit_ratio": 35.2, "ebit_period": "2026-01-25",
        "note": "EV/EBIT note, mentions EBITDA and APPROXIMATED.",
    }


def test_valuation_agent_produces_finding_per_scenario():
    agent = ValuationAgent("NVDA")
    with patch("agents.valuation_agent.run_dcf_scenarios", return_value=fake_scenarios()), \
         patch("agents.valuation_agent.calculate_pe_ratio", return_value=None), \
         patch("agents.valuation_agent.calculate_ev_ebit", return_value=None):
        result = agent.run("Value NVIDIA")

    scenario_findings = [f for f in result.findings if "DCF scenario" in f.claim]
    assert len(scenario_findings) == 2


def test_valuation_agent_flags_historical_cagr_scenario():
    agent = ValuationAgent("NVDA")
    with patch("agents.valuation_agent.run_dcf_scenarios", return_value=fake_scenarios()), \
         patch("agents.valuation_agent.calculate_pe_ratio", return_value=None), \
         patch("agents.valuation_agent.calculate_ev_ebit", return_value=None):
        result = agent.run("Value NVIDIA")

    historical_finding = next(f for f in result.findings if "historical_cagr" in f.claim)
    assert "likely unrealistic" in historical_finding.claim

    conservative_finding = next(f for f in result.findings if "conservative" in f.claim)
    assert "likely unrealistic" not in conservative_finding.claim


def test_valuation_agent_includes_pe_finding_when_available():
    agent = ValuationAgent("NVDA")
    with patch("agents.valuation_agent.run_dcf_scenarios", return_value={}), \
         patch("agents.valuation_agent.calculate_pe_ratio", return_value=fake_pe()), \
         patch("agents.valuation_agent.calculate_ev_ebit", return_value=None):
        result = agent.run("Value NVIDIA")

    pe_findings = [f for f in result.findings if "P/E ratio" in f.claim]
    assert len(pe_findings) == 1
    assert "46.7x" in pe_findings[0].claim


def test_valuation_agent_warns_when_pe_unavailable():
    agent = ValuationAgent("NVDA")
    with patch("agents.valuation_agent.run_dcf_scenarios", return_value={}), \
         patch("agents.valuation_agent.calculate_pe_ratio", return_value=None), \
         patch("agents.valuation_agent.calculate_ev_ebit", return_value=None):
        result = agent.run("Value NVIDIA")

    assert any("P/E ratio could not be computed" in w for w in result.warnings)


def test_valuation_agent_includes_ev_ebit_finding_when_available():
    agent = ValuationAgent("NVDA")
    with patch("agents.valuation_agent.run_dcf_scenarios", return_value={}), \
         patch("agents.valuation_agent.calculate_pe_ratio", return_value=None), \
         patch("agents.valuation_agent.calculate_ev_ebit", return_value=fake_ev_ebit()):
        result = agent.run("Value NVIDIA")

    ev_findings = [f for f in result.findings if "EV/EBIT ratio" in f.claim]
    assert len(ev_findings) == 1
    assert "35.2x" in ev_findings[0].claim


def test_valuation_agent_always_includes_fcf_proxy_warning():
    agent = ValuationAgent("NVDA")
    with patch("agents.valuation_agent.run_dcf_scenarios", return_value=fake_scenarios()), \
         patch("agents.valuation_agent.calculate_pe_ratio", return_value=fake_pe()), \
         patch("agents.valuation_agent.calculate_ev_ebit", return_value=fake_ev_ebit()):
        result = agent.run("Value NVIDIA")

    assert any("FCF proxy" in w for w in result.warnings)


def test_valuation_agent_empty_scenarios_still_returns_valid_result():
    agent = ValuationAgent("NVDA")
    with patch("agents.valuation_agent.run_dcf_scenarios", return_value={}), \
         patch("agents.valuation_agent.calculate_pe_ratio", return_value=fake_pe()), \
         patch("agents.valuation_agent.calculate_ev_ebit", return_value=fake_ev_ebit()):
        result = agent.run("Value NVIDIA")

    assert result.status == "success"  # P/E and EV/EBIT findings still present
    assert any("No DCF scenarios" in w for w in result.warnings)
