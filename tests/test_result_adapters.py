"""Every agent's existing output, converted to the standard AgentResult."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.financial_agent import FinancialAgent
from agents.market_agent import MarketAgent
from agents.news_agent import NewsAgent
from agents.red_team_agent import VerificationResult, run_verification
from agents.research_agent import ResearchAgent
from agents.risk_agent import RiskAgent
from agents.synthesis_agent import synthesize
from agents.valuation_agent import ValuationAgent
from core import result_adapters as ra
from core.agent_result import AgentResult, results_from_json, results_to_json
from core.schemas import AgentResult as LegacyAgentResult
from core.schemas import Finding, get_db_path
from tests.test_financial_agent import fake_summary
from tests.test_market_agent import FAKE_CONFIG, fake_get_annual_series
from tests.test_news_agent import fake_chunks_df
from tests.test_risk_agent import fake_risks
from tests.test_valuation_agent import fake_ev_ebit, fake_pe, fake_scenarios


def _financial():
    with patch("agents.financial_agent.build_financial_summary", return_value=fake_summary()):
        return FinancialAgent("NVDA").run("How healthy is NVIDIA?")


def _market():
    with patch("agents.market_agent.load_config", return_value=FAKE_CONFIG), \
         patch("agents.market_agent.get_annual_series", side_effect=fake_get_annual_series):
        return MarketAgent("NVDA").run("Compare")


def _valuation(scenarios=None, pe=None, ev=None):
    with patch("agents.valuation_agent.run_dcf_scenarios", return_value=fake_scenarios() if scenarios is None else scenarios), \
         patch("agents.valuation_agent.calculate_pe_ratio", return_value=fake_pe() if pe is None else pe), \
         patch("agents.valuation_agent.calculate_ev_ebit", return_value=fake_ev_ebit() if ev is None else ev):
        return ValuationAgent("NVDA").run("Value NVIDIA")


def _risk(risks=None):
    with patch("agents.risk_agent.identify_risks", return_value=fake_risks() if risks is None else risks):
        return RiskAgent("NVDA").run("Risks")


def _news(tmp_path, monkeypatch, label="FACT"):
    monkeypatch.setattr("agents.news_agent.load_config", lambda: {"paths": {}})
    monkeypatch.setattr("agents.news_agent.get_path", lambda cfg, key: tmp_path)
    fake_chunks_df().to_csv(tmp_path / "chunks_NVDA_8K.csv", index=False)
    agent = NewsAgent("NVDA")
    agent._classify_sentence = MagicMock(return_value={"label": label, "confidence": 0.8})
    return agent.run("News")


def _research():
    agent = ResearchAgent("NVDA")
    agent.retrieve = MagicMock(return_value=[
        {"score": 0.31, "text": "Compute & Networking segment ...", "chunk_id": "NVDA_10-K_1",
         "form": "10-K", "filing_date": "2026-02-25"},
        {"score": 0.01, "text": "too weak", "chunk_id": "x", "form": "10-K", "filing_date": "2020-01-01"},
    ])
    return agent.run("What are NVIDIA's segments?")


def _json_round_trip(result: AgentResult):
    assert AgentResult.from_json(result.to_json()) == result


# --- each agent ------------------------------------------------------------------

def test_financial_agent():
    legacy = _financial()
    r = ra.normalize_result(legacy)
    _json_round_trip(r)
    assert r.status == "success" and r.confidence == 1.0
    assert len(r.findings) == len(legacy.findings) == len(r.claims) == len(r.evidence)
    metrics = [c.metric for c in r.calculations]
    assert metrics.count("Revenue CAGR") == 1 and "Revenue Growth (YoY)" in metrics
    assert "Net Margin" in metrics and "Return on Equity" in metrics
    cagr = r.calculations[0]
    assert (cagr.value, cagr.unit, cagr.period) == (17.5, "%", "2016-01-31 to 2017-01-29")
    assert "NVDA_revenue_cagr" in cagr.source
    for claim, calc in zip(r.claims, r.calculations):
        assert claim.value == calc.value and f"{calc.value}" in claim.claim
    yoy = next(c for c in r.calculations if c.metric == "Revenue Growth (YoY)")
    assert yoy.period == "FY ended 2017-01-29" and yoy.value == 37.9
    assert r.uncertainties and "Free Cash Flow" in r.uncertainties[0]
    assert r.metrics == legacy.metrics
    assert "revenue CAGR 17.5%" in r.summary


def test_market_agent():
    r = ra.normalize_result(_market())
    _json_round_trip(r)
    assert r.assumptions["competitors"] == ["TCOMP"]
    by_metric = {c.metric: c for c in r.calculations}
    assert by_metric["NVDA Revenue CAGR"].unit == "%"
    assert by_metric["TCOMP Revenue CAGR"].value is not None
    ratio = by_metric["Revenue ratio NVDA/TCOMP"]
    assert ratio.unit == "x" and ratio.period == "fiscal years ending in calendar 2026"
    assert r.claims[0].value == "TCOMP"
    assert set(r.claims[1].value) == {"NVDA", "TCOMP"}
    assert "TCOMP" in r.summary


def test_valuation_agent():
    legacy = _valuation()
    r = ra.normalize_result(legacy)
    _json_round_trip(r)
    dcf = [v for v in r.valuations if v.method == "DCF"]
    assert [v.scenario for v in dcf] == ["conservative", "historical_cagr"]
    assert dcf[0].enterprise_value == 1_000_000_000_000.0
    assert dcf[0].assumptions["wacc"] == 0.11 and dcf[0].assumptions["terminal_growth"] == 0.03
    assert dcf[0].assumptions["revenue_growth_rate"] == 0.1
    assert dcf[1].assumptions["revenue_growth_rate"] is None
    assert dcf[1].assumptions["flagged_unrealistic"] is True
    assert dcf[0].equity_value is None and dcf[0].implied_share_price is None and dcf[0].sensitivity == {}
    pe = next(v for v in r.valuations if v.method == "P/E")
    assert pe.multiple == 46.7
    assert pe.assumptions == {"price": 228.86, "price_date": "2026-09-28", "eps_diluted": 4.9, "eps_period": "2026-01-25"}
    ev = next(v for v in r.valuations if v.method == "EV/EBIT")
    assert ev.enterprise_value == 1_760_000_000_000.0 and ev.multiple == 35.2
    assert ev.assumptions["ebit"] == 50_000_000_000.0 and ev.assumptions["cash"] == 10_000_000_000.0
    calcs = {c.metric: c.value for c in r.calculations}
    assert calcs["PV of terminal value (conservative)"] == 600_000_000_000.0
    assert calcs["Enterprise value (conservative)"] == 1_000_000_000_000.0
    assert calcs["P/E ratio"] == 46.7 and calcs["EV/EBIT"] == 35.2
    assert "fcf_proxy" in r.assumptions
    assert "DCF, EV/EBIT, P/E" in r.summary


def test_valuation_agent_without_dcf_has_no_fcf_assumption():
    r = ra.normalize_result(_valuation(scenarios={}))
    assert r.assumptions == {} and {v.method for v in r.valuations} == {"P/E", "EV/EBIT"}


def test_risk_agent():
    r = ra.normalize_result(_risk())
    _json_round_trip(r)
    assert [(x.risk, x.severity) for x in r.risks] == [("Competitive", "HIGH"), ("Supply Chain", "MEDIUM")]
    assert r.risks[0].probability is None and r.risks[0].contradicting_evidence == []
    assert "Competitive risk evidence text." in r.risks[0].evidence[0]
    assert r.claims[0].value == "HIGH"
    assert "2 risk categories" in r.summary and "HIGH=1" in r.summary
    assert r.assumptions["severity_method"].startswith("frequency proxy")


def test_risk_agent_with_one_category_and_no_distribution():
    legacy = _risk(fake_risks()[:1])
    legacy.metrics = {}
    r = ra.normalize_result(legacy)
    assert r.summary == "1 risk category assessed."


def test_news_agent(tmp_path, monkeypatch):
    r = ra.normalize_result(_news(tmp_path, monkeypatch))
    _json_round_trip(r)
    assert r.claims and all(c.value == "FACT" for c in r.claims)
    assert all(c.source.startswith("8-K:") for c in r.claims)
    assert r.confidence == 0.8
    assert "FACT=" in r.summary and r.uncertainties


def test_news_agent_unreliable_label_kept(tmp_path, monkeypatch):
    r = ra.normalize_result(_news(tmp_path, monkeypatch, label="CLAIM"))
    assert all(c.value == "CLAIM" and "LOW RELIABILITY" in c.claim for c in r.claims)


def test_research_agent():
    r = ra.normalize_result(_research())
    _json_round_trip(r)
    assert len(r.claims) == 1 and r.claims[0].value is None
    assert r.evidence[0].text.startswith("Compute & Networking")
    assert r.source_references[0].source_id == "NVDA_10-K_1"
    assert r.summary.startswith("1 filing passage(s) retrieved")


def test_red_team_agent():
    legacy = {"FinancialAgent": _financial(), "ValuationAgent": _valuation()}
    verification = run_verification(legacy)
    r = ra.normalize_verification(verification)
    _json_round_trip(r)
    assert r.agent_name == "RedTeamAgent" and r.status == "success"
    assert r.metrics["total_issues"] == len(verification) == len(r.claims)
    assert sum(r.metrics["by_result"].values()) == len(verification)
    assert {c.value for c in r.claims} <= {"SUPPORTED", "UNSUPPORTED", "FLAGGED"}


def test_red_team_with_no_issues_and_with_error():
    r = ra.normalize_verification([])
    assert r.summary == "0 verification issue(s)." and r.claims == []
    failed = ra.normalize_verification(None, error="verifier down")
    assert failed.status == "failed" and failed.errors == ["verifier down"]


def test_red_team_deduplicates_problems():
    v = VerificationResult("c", "e", "FLAGGED", 0.05, "low", "fix", "NewsAgent")
    r = ra.normalize_verification([v, v])
    assert r.uncertainties == ["[NewsAgent] FLAGGED: low"]


def test_synthesis_agent():
    response = {"query": "q", "company_ticker": "NVDA",
                "agent_results": {"FinancialAgent": _financial(), "RiskAgent": _risk()},
                "verification": [VerificationResult(
                    "NVDA revenue grew at a 17.5% CAGR from 2016-01-31 to 2017-01-29.", "e",
                    "FLAGGED", 1.0, "p", "c", "FinancialAgent")]}
    r = ra.normalize_synthesis(synthesize(response))
    _json_round_trip(r)
    assert r.agent_name == "SynthesisAgent" and r.status == "success"
    assert r.metrics["agent_count"] == 2 and r.metrics["verification_summary"]["total_issues"] == 1
    assert r.details["text"].startswith("Synthesis for NVDA")
    flagged = [c for c in r.claims if c.value]
    assert flagged and flagged[0].value == {"issues": ["FLAGGED"]}
    assert r.uncertainties and r.errors == []


def test_synthesis_failed_and_partial_statuses():
    failed = ra.normalize_synthesis(synthesize({"agent_results": {}}))
    assert failed.status == "failed" and failed.errors
    partial = ra.normalize_synthesis(synthesize({"agent_results": {
        "RiskAgent": LegacyAgentResult("RiskAgent", "t", status="partial")}}))
    assert partial.status == "partial"


# --- failure and fallback paths ---------------------------------------------------

def test_failed_agent_result():
    with patch("agents.risk_agent.identify_risks", side_effect=FileNotFoundError("vector store missing")):
        legacy = RiskAgent("NVDA").run("Risks")
    r = ra.normalize_result(legacy)
    assert r.status == "failed" and r.confidence is None
    assert r.errors == ["Agent execution failed: vector store missing"]
    assert r.summary == "RiskAgent failed." and r.uncertainties == []
    _json_round_trip(r)


def test_failure_warning_on_non_failed_status_is_an_error():
    legacy = LegacyAgentResult("X", "t", status="partial",
                               warnings=["Agent execution failed: x", "caveat"])
    r = ra.normalize_result(legacy)
    assert r.errors == ["Agent execution failed: x"] and r.uncertainties == ["caveat"]


def test_unknown_agent_and_status_use_generic_form():
    legacy = LegacyAgentResult("PositionAgent", "t", status="weird", findings=[
        Finding("claim", "ev", "id1", "form", "n/a", 0.4), Finding("claim2", "ev", "id1", "form", "n/a", 0.6)])
    r = ra.normalize_result(legacy)
    assert r.status == "partial" and r.confidence == 0.5
    assert len(r.source_references) == 1 and r.source_references[0].date is None


@pytest.mark.parametrize("agent", ["FinancialAgent", "MarketAgent", "ValuationAgent", "RiskAgent", "NewsAgent"])
def test_unrecognised_finding_formats_fall_back_to_plain_claims(agent):
    legacy = LegacyAgentResult(agent, "t", findings=[Finding("Some new claim", "ev", "NVDA_new_thing", "f", "2026-01-01", 0.7)])
    r = ra.normalize_result(legacy)
    assert len(r.claims) == 1 and r.claims[0].value is None
    assert r.calculations == [] and r.valuations == [] and r.risks == []


def test_parser_exception_falls_back_and_is_recorded(monkeypatch):
    def broken(_):
        raise RuntimeError("parser bug")
    monkeypatch.setitem(ra.NORMALIZERS, "FinancialAgent", broken)
    r = ra.normalize_result(_financial())
    assert r.claims and r.calculations == []
    assert "parser bug" in r.uncertainties[-1]


def test_failed_results_keep_generic_summary_for_every_agent():
    for name in ra.NORMALIZERS:
        r = ra.normalize_result(LegacyAgentResult(name, "t", status="failed", warnings=["boom"]))
        assert r.summary == f"{name} failed." and r.errors == ["boom"]


# --- wrappers / multiple results -----------------------------------------------

def test_run_standardized_wraps_existing_agent():
    with patch("agents.financial_agent.build_financial_summary", return_value=fake_summary()):
        r = ra.run_standardized(FinancialAgent, "NVDA", "task")
    assert isinstance(r, AgentResult) and r.agent_name == "FinancialAgent"


def test_standardize_response_and_multiple_results(tmp_path, monkeypatch):
    agent_results = {
        "ResearchAgent": _research(), "FinancialAgent": _financial(), "MarketAgent": _market(),
        "NewsAgent": _news(tmp_path, monkeypatch), "RiskAgent": _risk(), "ValuationAgent": _valuation(),
    }
    response = {"query": "q", "company_ticker": "NVDA", "agent_results": agent_results,
                "verification": run_verification(agent_results)}
    response["synthesis"] = synthesize(response)
    results = ra.standardize_response(response)
    names = [r.agent_name for r in results]
    assert names == list(agent_results) + ["RedTeamAgent", "SynthesisAgent"]  # all 8 agents
    text = results_to_json(results)
    doc = json.loads(text)
    assert doc["agent_count"] == 8
    assert results_from_json(text) == results


def test_standardize_response_without_verification_or_synthesis():
    results = ra.standardize_response({"agent_results": {"FinancialAgent": _financial()}})
    assert [r.agent_name for r in results] == ["FinancialAgent"]


@pytest.mark.integration
@pytest.mark.skipif(not get_db_path().exists(), reason="local knowledge base not built")
def test_real_agents_produce_valid_json():
    from core.orchestrator import run_query
    response = run_query("NVDA", "revenue growth valuation dcf competitors", synthesize_report=True)
    results = ra.standardize_response(response)
    assert results_from_json(results_to_json(results)) == results
    valuation = next(r for r in results if r.agent_name == "ValuationAgent")
    assert valuation.status == "failed" or valuation.valuations
