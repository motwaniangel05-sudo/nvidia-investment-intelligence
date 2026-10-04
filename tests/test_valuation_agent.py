"""Tests for ValuationAgent: Finding construction from valuation_tools output."""

from unittest.mock import patch

from agents.valuation_agent import ValuationAgent


def fake_scenarios():
    return {
        "conservative": {
            "enterprise_value": 1_000_000_000_000.0,
            "sum_pv_explicit_period": 400_000_000_000.0,
            "pv_terminal_value": 600_000_000_000.0,
            "assumptions": {
                "revenue_growth_rate_used": 0.10,
                "wacc": 0.11, "terminal_growth": 0.03,
            },
            "fcf_disclaimer": "FCF proxy disclaimer text.",
            "scenario_is_historical_extrapolation": False,
        },
        "historical_cagr": {
            "enterprise_value": 4_800_000_000_000.0,
            "sum_pv_explicit_period": 1_000_000_000_000.0,
            "pv_terminal_value": 3_800_000_000_000.0,
            "assumptions": {
                "revenue_growth_rate_used": None,
                "wacc": 0.11, "terminal_growth": 0.03,
            },
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
        "ebit": 50_000_000_000.0,
        "enterprise_value": 1_760_000_000_000.0,
        "market_cap": 1_750_000_000_000.0,
        "total_debt": 20_000_000_000.0,
        "cash": 10_000_000_000.0,
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



# ---- retrieve, DCF failure path, metadata, status and metrics ----
from agents.red_team_agent import check_missing_disclaimers


def _run(scenarios=None, pe=None, ev_ebit=None, scenarios_error=None):
    agent = ValuationAgent("NVDA")
    dcf_patch = (
        patch("agents.valuation_agent.run_dcf_scenarios", side_effect=scenarios_error)
        if scenarios_error
        else patch("agents.valuation_agent.run_dcf_scenarios", return_value=scenarios or {})
    )
    with dcf_patch, \
         patch("agents.valuation_agent.calculate_pe_ratio", return_value=pe), \
         patch("agents.valuation_agent.calculate_ev_ebit", return_value=ev_ebit):
        return agent.run("Value NVIDIA")


def test_retrieve_is_unused_and_returns_empty_list():
    assert ValuationAgent("NVDA").retrieve("anything") == []


def test_dcf_exception_becomes_warning_and_other_findings_survive():
    result = _run(scenarios_error=RuntimeError("boom"), pe=fake_pe(), ev_ebit=fake_ev_ebit())

    assert any("DCF scenarios could not be computed: boom" in w for w in result.warnings)
    assert any("No DCF scenarios" in w for w in result.warnings)
    assert result.metrics["scenarios_computed"] == 0
    assert result.status == "success"
    assert {f.source_chunk_id for f in result.findings} == {"NVDA_pe_ratio", "NVDA_ev_ebit"}


def test_nothing_available_gives_partial_status_and_all_warnings():
    result = _run()

    assert result.status == "partial"
    assert result.findings == []
    assert result.metrics == {
        "scenarios_computed": 0, "pe_available": False, "ev_ebit_available": False,
    }
    for expected in ("P/E ratio could not be computed", "EV/EBIT could not be computed",
                     "No DCF scenarios", "FCF proxy"):
        assert any(expected in w for w in result.warnings), expected


def test_metrics_reflect_what_was_computed():
    result = _run(scenarios=fake_scenarios(), pe=fake_pe(), ev_ebit=None)
    assert result.metrics == {
        "scenarios_computed": 2, "pe_available": True, "ev_ebit_available": False,
    }


def test_scenario_finding_metadata_and_text():
    result = _run(scenarios=fake_scenarios())
    f = next(x for x in result.findings if x.source_chunk_id == "NVDA_dcf_conservative")

    assert f.source_form == "DCF_model"
    assert f.source_filing_date == "n/a"
    assert f.confidence == 1.0
    assert "Enterprise Value = $1,000,000,000,000" in f.claim
    assert "growth rate used: 0.1" in f.claim
    assert "FCF proxy disclaimer text." in f.evidence_text
    assert "WACC=0.11" in f.evidence_text
    assert "Terminal Growth=0.03" in f.evidence_text


def test_pe_finding_metadata_and_text():
    result = _run(pe=fake_pe())
    f = next(x for x in result.findings if x.source_chunk_id == "NVDA_pe_ratio")

    assert f.source_form == "market_data"
    assert f.source_filing_date == "2026-09-28"
    assert f.claim == "NVDA P/E ratio: 46.7x (price $228.86 / EPS $4.90)."
    assert "P/E = 228.86 / 4.90 = 46.7x." in f.evidence_text


def test_ev_ebit_finding_metadata_and_text():
    result = _run(ev_ebit=fake_ev_ebit())
    f = next(x for x in result.findings if x.source_chunk_id == "NVDA_ev_ebit")

    assert f.source_form == "market_data_and_XBRL"
    assert f.source_filing_date == "2026-01-25"
    assert f.claim == "NVDA EV/EBIT ratio: 35.2x."
    assert "EV/EBIT = 1,760,000,000,000 / 50,000,000,000 = 35.2x" in f.evidence_text


def test_ev_ebit_finding_satisfies_red_team_ebitda_disclaimer_rule():
    result = _run(ev_ebit=fake_ev_ebit())
    assert check_missing_disclaimers(result.findings, "ValuationAgent") == []
