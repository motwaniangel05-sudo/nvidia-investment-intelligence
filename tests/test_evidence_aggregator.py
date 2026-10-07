"""Tests for the Evidence Aggregator (core/evidence_aggregator.py)."""

import json

import pytest

from agents.red_team_agent import VerificationResult
from core import dynamic_orchestrator as dyn
from core.agent_result import AgentResult, Calculation, Claim, Risk
from core.evidence_aggregator import (
    DEFAULT_TOKEN_BUDGET, Relevance, _referenced_ids, aggregate_evidence, estimate_tokens, source_quality,
)
from core.result_adapters import normalize_result, normalize_verification
from tests.test_dynamic_orchestrator import CONFIG, _registry
from tests.test_result_adapters import _financial, _market, _news, _research, _risk, _valuation

Q = "Should I buy NVIDIA at the current price?"


def all_results(tmp_path, monkeypatch):
    return [normalize_result(x) for x in (_research(), _financial(), _market(),
                                         _news(tmp_path, monkeypatch), _risk(), _valuation())]


def news_result(claims):
    return AgentResult("NewsAgent", "t", "success", claims=claims)


def news_claim(text, date, label="FACT", conf=0.6, n=0):
    return Claim(f"Classified as {label}", label, source=f"8-K:NVDA_8-K_{date}_{n:04d}", date=date,
                 evidence=text, confidence=conf)


def calc(agent, metric, value, source_id, period="p", unit="%"):
    return AgentResult(agent, "t", "success", calculations=[
        Calculation(metric, value, unit, period, None, f"SEC XBRL company facts ({source_id})")])


def all_numbers(ctx):
    return json.dumps(ctx.to_dict())


# --- multiple agents ---------------------------------------------------------------

def test_multiple_agents_fill_every_section(tmp_path, monkeypatch):
    results = all_results(tmp_path, monkeypatch)
    ctx = aggregate_evidence(Q, results, intent="investment_decision", company_ticker="NVDA")
    d = json.loads(ctx.to_json())
    for key in ("facts", "financial_metrics", "market_findings", "competitor_findings", "valuation",
                "risks", "news_events", "sources", "agent_status"):
        assert d[key], key
    assert [s["agent"] for s in d["agent_status"]] == [r.agent_name for r in results]
    assert d["intent"] == "investment_decision" and d["company_ticker"] == "NVDA"
    assert ctx.meta["token_estimate"] <= DEFAULT_TOKEN_BUDGET and not ctx.meta["budget_exceeded"]
    price = d["market_findings"][0]
    assert (price["metric"], price["value"], price["date"]) == ("Share price (close)", 228.86, "2026-09-28")


def test_numerical_evidence_is_never_dropped(tmp_path, monkeypatch):
    results = all_results(tmp_path, monkeypatch)
    ctx = aggregate_evidence(Q, results, token_budget=1)  # impossible budget
    assert ctx.meta["budget_exceeded"]
    flat = all_numbers(ctx)
    for r in results:
        for c in r.calculations:
            assert json.dumps(c.value) in flat, (r.agent_name, c.metric)
        for v in r.valuations:
            for n in (v.enterprise_value, v.multiple):
                if n is not None:
                    assert json.dumps(n) in flat
    fin = next(r for r in results if r.agent_name == "FinancialAgent")
    groups = {g["metric"]: g for g in ctx.financial_metrics}
    assert ctx.meta["section_agents"]["financial_metrics"] == "FinancialAgent"
    assert sum(len(g["values"]) for g in groups.values()) == len(fin.calculations)
    assert groups["Net Margin"]["period_prefix"] == "FY ended "
    assert groups["Net Margin"]["values"] == {"2016-01-31": 12.3, "2017-01-29": 24.1}


def test_text_sections_shrink_to_fit_the_budget(tmp_path, monkeypatch):
    claims = [news_claim(f"Item {i}: the board approved routine governance matter number {i} for the year.",
                         f"2020-01-{i % 28 + 1:02d}", n=i) for i in range(200)]
    roomy = aggregate_evidence(Q, [news_result(claims)], token_budget=100_000)
    tight = aggregate_evidence(Q, [news_result(claims)], token_budget=600)
    assert len(roomy.news_events) == 12 and roomy.meta["omitted"]["news_items"] == 188
    assert 5 <= len(tight.news_events) < 12 and tight.token_estimate <= 600


# --- news ranking ----------------------------------------------------------------

def test_news_is_ranked_by_relevance_and_capped():
    noise = [news_claim(f"NVIDIA filed governance exhibit {i} with the commission.", "2025-01-01", n=i)
             for i in range(300)]
    relevant = [news_claim("NVIDIA reported record quarterly revenue of $35.1 billion in its earnings release.",
                           "2024-11-20", n=900)]
    opinion = [news_claim("NVIDIA believes its earnings release shows strong demand.", "2024-11-20",
                          label="INTERPRETATION", conf=0.3, n=901)]
    ctx = aggregate_evidence("What happened after the latest NVIDIA earnings release?",
                             [news_result(noise + relevant + opinion)])
    assert ctx.news_events[0]["text"].startswith("NVIDIA reported record quarterly revenue")
    assert len(ctx.news_events) + len(ctx.interpretations) == 12
    assert ctx.interpretations[0]["quality"] == "LOW" and ctx.interpretations[0]["label"] == "INTERPRETATION"
    assert ctx.meta["omitted"]["news_items"] == 302 - 12


def test_recency_breaks_ties_for_latest_questions():
    claims = [news_claim("Quarterly dividend declared.", "2020-05-01", n=1),
              news_claim("Quarterly dividend approved.", "2026-05-01", n=2)]
    ctx = aggregate_evidence("latest dividend", [news_result(claims)])
    assert ctx.news_events[0]["date"] == "2026-05-01"


def test_relevance_ignores_terms_present_everywhere():
    rel = Relevance("NVIDIA revenue", ["NVIDIA revenue rose", "NVIDIA held a meeting", "NVIDIA filed"])
    assert rel.score("NVIDIA held a meeting") == 0.0
    assert rel.score("NVIDIA revenue rose") == 1.0
    assert Relevance("the and", ["x"]).score("anything") == 0.0


# --- duplicates ----------------------------------------------------------------------

def test_duplicate_news_is_merged_with_dates_kept():
    text = "The press release is attached as Exhibit 99.1."
    ctx = aggregate_evidence("q", [news_result([
        news_claim(text, "2024-01-01", n=1), news_claim(text.upper() + "  ", "2025-06-01", n=2, conf=0.9),
        news_claim(text, "2023-03-01", n=3)])])
    assert len(ctx.news_events) == 1
    item = ctx.news_events[0]
    assert item["occurrences"] == 3 and item["first_date"] == "2023-03-01" and item["date"] == "2025-06-01"
    assert item["source_id"] == "NVDA_8-K_2025-06-01_0002" and item["confidence"] == 0.9


def test_duplicate_competitor_metrics_merge_sources():
    market = AgentResult("MarketAgent", "t", "success", calculations=[
        Calculation("NVDA Revenue CAGR", 45.8, "%", "available fiscal history", None, f"SEC XBRL company facts (NVDA_vs_{c}_cagr)")
        for c in ("AMD", "INTC")])
    ctx = aggregate_evidence("q", [market])
    assert len(ctx.competitor_findings) == 1
    assert ctx.competitor_findings[0]["source_ids"] == ["NVDA_vs_AMD_cagr", "NVDA_vs_INTC_cagr"]
    assert ctx.conflicts == []


def test_identical_evidence_from_two_agents_is_kept_once():
    claim = Claim("Relevant evidence found regarding: x", source="10-K:NVDA_10-K_7", date="2026-02-25",
                  evidence="Data Center revenue was $115.2 billion.", confidence=0.4)
    ctx = aggregate_evidence("data center revenue", [
        AgentResult("ResearchAgent", "t", "success", claims=[claim]),
        AgentResult("PositionAgent", "t", "success", claims=[Claim("Data center", source=claim.source,
                    date=claim.date, evidence=claim.evidence, confidence=0.7)])])
    assert len(ctx.facts) == 1
    assert ctx.facts[0]["agents"] == ["ResearchAgent", "PositionAgent"] and ctx.facts[0]["confidence"] == 0.7
    assert "claim" not in ctx.facts[0]  # generic retrieval label dropped, evidence kept


# --- conflicts ---------------------------------------------------------------------

def test_conflicting_numbers_are_both_preserved():
    ctx = aggregate_evidence("q", [
        calc("FinancialAgent", "Revenue CAGR", 45.8, "NVDA_revenue_cagr", "2016-01-31 to 2026-01-25"),
        AgentResult("MarketAgent", "t", "success", calculations=[Calculation(
            "NVDA Revenue CAGR", 40.0, "%", "available fiscal history", None, "SEC XBRL company facts (NVDA_vs_AMD_cagr)")])])
    assert len(ctx.conflicts) == 1
    c = ctx.conflicts[0]
    assert c["type"] == "numeric_mismatch" and c["subject"] == "NVDA Revenue CAGR (full available history)"
    assert [(p["agent"], p["value"]) for p in c["positions"]] == [("FinancialAgent", 45.8), ("MarketAgent", 40.0)]
    assert "not resolved" in c["note"]


def test_rounding_differences_are_not_conflicts():
    ctx = aggregate_evidence("q", [calc("FinancialAgent", "Revenue CAGR", 45.8, "NVDA_revenue_cagr"),
                                   calc("PositionAgent", "Revenue CAGR", 45.84, "NVDA_x_cagr")])
    assert ctx.conflicts == []


def test_risk_severity_disagreement_is_a_conflict():
    ctx = aggregate_evidence("q", [AgentResult("RiskAgent", "t", "success", risks=[Risk("Supply Chain", "HIGH")]),
                                   AgentResult("PositionAgent", "t", "success", risks=[Risk("supply chain", "LOW")])])
    assert ctx.conflicts[0]["type"] == "severity_mismatch"
    assert {p["value"] for p in ctx.conflicts[0]["positions"]} == {"HIGH", "LOW"}


def test_user_price_mismatch_keeps_both_prices():
    val = normalize_result(_valuation())
    ctx = aggregate_evidence(Q, [val], company_ticker="NVDA", user_prices={"current_price": 500.0})
    c = ctx.conflicts[0]
    assert c["type"] == "user_price_mismatch" and c["subject"] == "NVDA current share price"
    assert [p["value"] for p in c["positions"]] == [500.0, 228.86] and c["positions"][1]["date"] == "2026-09-28"
    assert ctx.user_inputs["current_price"] == 500.0 and "never used as data" in ctx.user_inputs["note"]
    assert aggregate_evidence(Q, [val], user_prices={"current_price": 228.9}).conflicts == []


def test_conflicts_survive_an_impossible_budget():
    ctx = aggregate_evidence("q", [calc("FinancialAgent", "ROE", 10.0, "NVDA_roe_2026"),
                                   calc("PositionAgent", "ROE", 20.0, "NVDA_roe2_2026")], token_budget=1)
    assert len(ctx.conflicts) == 1 and ctx.meta["budget_exceeded"]


def test_red_team_flags_are_kept():
    v = normalize_verification([
        VerificationResult("NVDA CAGR 45.8% vs AMD", "e", "UNSUPPORTED", 1.0, "45.8 not in evidence", "fix", "MarketAgent"),
        VerificationResult("NVDA CAGR 45.8% vs INTC", "e", "UNSUPPORTED", 1.0, "12.1 not in evidence", "fix", "MarketAgent"),
        VerificationResult("Low confidence", "e", "FLAGGED", 0.1, "below threshold", "fix", "NewsAgent")])
    ctx = aggregate_evidence("q", [v])
    assert ctx.verification_flags == [
        {"agent": "MarketAgent", "result": "UNSUPPORTED", "count": 2,
         "claims": ["NVDA CAGR 45.8% vs AMD", "NVDA CAGR 45.8% vs INTC"], "problems": ["45.8 not in evidence"]},
        {"agent": "NewsAgent", "result": "FLAGGED", "count": 1, "claims": ["Low confidence"],
         "problems": ["below threshold"]}]
    assert ctx.missing_evidence == []


# --- missing evidence / failures ----------------------------------------------------

def test_missing_evidence_is_explicit_not_invented(tmp_path, monkeypatch):
    research = AgentResult("ResearchAgent", "t", "partial", uncertainties=["No relevant passages."])
    ctx = aggregate_evidence(Q, [research, normalize_result(_valuation()), normalize_result(_risk())])
    assert "ResearchAgent returned no evidence (partial)." in ctx.missing_evidence
    dcf = next(v for v in ctx.valuation if v["method"] == "DCF")
    assert "equity_value" not in dcf and "implied_share_price" not in dcf
    assert "not_computed" not in dcf  # identical for every scenario, so stated once:
    assert ctx.assumptions["ValuationAgent"]["dcf_shared"]["not_computed"] == [
        "equity_value", "implied_share_price", "sensitivity"]
    pe = next(v for v in ctx.valuation if v["method"] == "P/E")
    assert pe["not_computed"] == ["enterprise_value", "equity_value", "implied_share_price", "sensitivity"] \
        or "enterprise_value" not in pe
    assert ctx.risks[0]["not_assessed"] == ["probability", "impact"]
    assert ctx.uncertainties[0] == "[ResearchAgent] No relevant passages."


def test_user_price_without_stored_price_is_reported():
    ctx = aggregate_evidence(Q, [], user_prices={"current_price": 500.0})
    assert ctx.missing_evidence == ["No stored share price available to compare with the user-supplied current price."]
    assert ctx.conflicts == []


def test_no_results_gives_valid_empty_context():
    ctx = aggregate_evidence("q", [])
    d = json.loads(ctx.to_json())
    assert d["facts"] == [] and d["agent_status"] == [] and d["meta"]["omitted"] == {}


def test_failed_agent_is_recorded_but_its_content_ignored():
    failed = AgentResult("RiskAgent", "t", "failed", claims=[Claim("should be ignored")],
                         risks=[Risk("ignored")], errors=["vector store missing"])
    ctx = aggregate_evidence("q", [failed, normalize_result(_financial())])
    assert ctx.risks == [] and "should be ignored" not in all_numbers(ctx)
    assert ctx.agent_status[0] == {"agent": "RiskAgent", "status": "failed", "errors": ["vector store missing"]}
    assert "agent" not in ctx.financial_metrics[0]
    assert ctx.missing_evidence == ["RiskAgent failed: vector store missing"]
    assert ctx.financial_metrics


def test_real_failed_agent_through_adapter():
    from unittest.mock import patch
    from agents.risk_agent import RiskAgent
    with patch("agents.risk_agent.identify_risks", side_effect=FileNotFoundError("no store")):
        failed = normalize_result(RiskAgent("NVDA").run("Risks"))
    ctx = aggregate_evidence("q", [failed, AgentResult("Other", "t", "failed")])
    assert ctx.missing_evidence == ["RiskAgent failed: Agent execution failed: no store",
                                    "Other failed: unknown error"]


def test_synthesis_content_is_not_duplicated():
    ctx = aggregate_evidence("q", [AgentResult("SynthesisAgent", "t", "success", claims=[Claim("restated")])])
    assert ctx.facts == [] and ctx.missing_evidence == [] and ctx.agent_status[0]["agent"] == "SynthesisAgent"


def test_unparsed_claims_from_structured_agents_become_facts():
    r = AgentResult("FinancialAgent", "t", "success", claims=[Claim("New metric text", source="XBRL_metrics:NVDA_new")])
    ctx = aggregate_evidence("q", [r])
    assert ctx.facts[0]["claim"] == "New metric text" and ctx.facts[0]["quality"] == "HIGH"


# --- source preservation ------------------------------------------------------------

def test_sources_dates_and_quality_are_preserved(tmp_path, monkeypatch):
    ctx = aggregate_evidence(Q, all_results(tmp_path, monkeypatch))
    ids = set(_referenced_ids(ctx))  # every item cites its source id(s)
    assert sum(s["count"] for s in ctx.sources) == len(ids)
    assert all(s["quality"] in ("HIGH", "MEDIUM", "LOW") and s["form"] for s in ctx.sources)
    net_margin = next(g for g in ctx.financial_metrics if g["metric"] == "Net Margin")
    assert net_margin["source_id_pattern"] == "NVDA_net_margin_{period_end_date}"
    forms = {(s["form"], s["quality"]) for s in ctx.sources}
    assert {("SEC XBRL company facts", "HIGH"), ("8-K", "MEDIUM"), ("DCF model", "LOW"), ("config", "LOW")} <= forms
    assert net_margin["quality"] == "HIGH" and net_margin["confidence"] == 1.0
    cagr = next(g for g in ctx.financial_metrics if g["metric"] == "Revenue CAGR")
    assert cagr["source_ids"] == ["NVDA_revenue_cagr"]
    assert ctx.facts[0]["source_id"] == "NVDA_10-K_1" and ctx.facts[0]["date"] == "2026-02-25"
    assert ctx.facts[0]["quality"] == "HIGH"
    news = ctx.news_events[0]
    assert news["quality"] == "MEDIUM" and news["confidence"] == 0.8 and news["date"]
    assert news["source_id"] in ids
    by_method = {(v["method"], v.get("scenario")): v for v in ctx.valuation}
    assert by_method[("DCF", "conservative")]["quality"] == "LOW"
    assert by_method[("P/E", None)]["quality"] == "MEDIUM"
    assert {c["quality"] for c in ctx.competitor_findings if "competitor" in c} == {"LOW"}
    assert ctx.risks[0]["source_id"] in ids and ctx.risks[0]["quality"] == "HIGH"


@pytest.mark.parametrize("source,tier", [
    ("XBRL_metrics:NVDA_roe", "HIGH"), ("10-K/10-Q:chunk", "HIGH"), ("SEC XBRL company facts (x)", "HIGH"),
    ("8-K:NVDA_8-K_x", "MEDIUM"), ("market data + SEC XBRL company facts (x)", "MEDIUM"),
    ("market_data_and_XBRL:x", "MEDIUM"), ("DCF model (x)", "LOW"), ("config:AMD", "LOW"), (None, "LOW"),
    ("blog", "LOW"),
])
def test_source_quality_tiers(source, tier):
    assert source_quality(source) == tier


def test_estimate_tokens():
    assert estimate_tokens("abcd" * 10) == 10


# --- orchestrator integration ------------------------------------------------------

def test_orchestrator_builds_evidence_context(monkeypatch):
    monkeypatch.setattr(dyn, "run_verification", lambda results: [])
    q = "NVIDIA is trading at $500 today. Is it overvalued?"
    r = dyn.run_dynamic_query(q, config=CONFIG, registry=_registry())
    assert r.evidence_context.query == q
    assert r.evidence_context.user_inputs["current_price"] == 500.0
    assert r.to_dict()["evidence_context"]["intent"] == r.plan.intent


def test_orchestrator_survives_aggregator_error(monkeypatch):
    monkeypatch.setattr(dyn, "run_verification", lambda results: [])

    def broken(*a, **k):
        raise RuntimeError("bad")
    monkeypatch.setattr(dyn, "aggregate_evidence", broken)
    r = dyn.run_dynamic_query("Is NVIDIA overvalued?", config=CONFIG, registry=_registry())
    assert r.evidence_context is None and r.to_dict()["evidence_context"] is None


def test_cli_context(capsys, monkeypatch):
    monkeypatch.setattr(dyn, "run_verification", lambda results: [])
    assert dyn.main(["--context", "Is", "NVIDIA", "overvalued?"], registry=_registry()) == 0
    assert json.loads(capsys.readouterr().out)["query"] == "Is NVIDIA overvalued?"
    monkeypatch.setattr(dyn, "aggregate_evidence", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    assert dyn.main(["--context", "q"], registry=_registry()) == 0
    assert capsys.readouterr().out.strip() == "{}"


def test_dcf_components_are_folded_into_valuations():
    ctx = aggregate_evidence(Q, [normalize_result(_valuation())])
    dcf = next(v for v in ctx.valuation if v.get("scenario") == "conservative")
    assert dcf["components"] == {"PV of explicit-period FCF": 400_000_000_000.0,
                                 "PV of terminal value": 600_000_000_000.0}
    assert ctx.financial_metrics == []  # nothing left over, nothing duplicated
    shared = ctx.assumptions["ValuationAgent"]["dcf_shared"]
    assert shared["wacc"] == 0.11 and shared["terminal_growth"] == 0.03
    assert "wacc" not in dcf.get("assumptions", {}) and dcf["assumptions"]["revenue_growth_rate"] == 0.1


def test_metric_names_are_not_mistaken_for_tickers():
    ctx = aggregate_evidence("q", [calc("PositionAgent", "PV of cash", 1.0, "NVDA_pv", unit="USD")])
    assert (ctx.financial_metrics[0]["metric"], ctx.financial_metrics[0]["ticker"]) == ("PV of cash", "NVDA")


def test_two_values_for_one_period_are_both_kept():
    r = AgentResult("PositionAgent", "t", "success", calculations=[
        Calculation("ROE", 10.0, "%", "FY ended 2026", None, "SEC XBRL company facts (NVDA_a)"),
        Calculation("ROE", 12.0, "%", "FY ended 2026", None, "SEC XBRL company facts (NVDA_b)"),
        Calculation("ROE", 14.0, "%", "FY ended 2026", None, "SEC XBRL company facts (NVDA_c)")])
    ctx = aggregate_evidence("q", [r])
    assert ctx.financial_metrics[0]["period_prefix"] == "FY ended "
    assert ctx.financial_metrics[0]["values"] == {"2026": [10.0, 12.0, 14.0]}
    assert ctx.conflicts[0]["type"] == "numeric_mismatch"


def test_query_expansion_matches_related_words():
    rel = Relevance("latest earnings", ["Quarterly results were strong", "Board meeting held", "Other text"])
    assert rel.score("Quarterly results were strong") > rel.score("Board meeting held") == 0.0


def test_edge_inputs_are_skipped_not_invented():
    from core.agent_result import Valuation
    dcf = AgentResult("ValuationAgent", "t", "success", valuations=[
        Valuation("DCF", s, 1.0 * i, assumptions={"wacc": 0.1}, source=f"DCF model (NVDA_dcf_{s})")
        for i, s in enumerate(("a", "b"), 1)])
    ctx = aggregate_evidence("q", [dcf, calc("PositionAgent", "ROE", None, "NVDA_roe"),
                                   news_result([news_claim("", "2026-01-01")])])
    assert all("assumptions" not in v for v in ctx.valuation)
    assert ctx.assumptions["ValuationAgent"]["dcf_shared"]["wacc"] == 0.1
    assert ctx.news_events == [] and ctx.conflicts == []
