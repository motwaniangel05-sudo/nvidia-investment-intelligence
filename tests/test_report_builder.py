import sys

import pytest

from core import report_builder as rb

POS = {
    "source": "USER_PROVIDED (not verified market data)",
    "inputs": {"current_price": 500.0, "previous_price": 200.0, "cost_basis": 120.0},
    "calculations": [
        {"metric": "daily_change_percent", "value": 150.0, "unit": "%", "formula": "f"},
        {"metric": "gain_or_loss_vs_cost_percent", "value": 316.7, "unit": "%", "formula": "f"},
        {"metric": "position_state", "value": "GAIN", "unit": "", "formula": "f"},
    ],
    "flags": ["One-day move of 150.0% is unusually large."],
}

CTX = {
    "position_calculations": POS,
    "market_findings": [{"metric": "Share price (close)", "value": 228.86, "unit": "USD",
                         "date": "2026-09-28", "quality": "MEDIUM"}],
    "conflicts": [{"subject": "NVDA current share price", "note": "n",
                   "positions": [{"source": "user", "value": 500.0, "unit": "USD"},
                                 {"agent": "ValuationAgent", "value": 228.86, "unit": "USD"}]}],
    "financial_metrics": [
        {"metric": "Revenue Growth (YoY)", "unit": "%", "values": {"2025": 114.2, "2026": 65.5}, "quality": "HIGH"},
        {"metric": "Net Margin", "unit": "%", "values": {"2026": 55.6}, "quality": "HIGH"},
        {"metric": "Return on Equity", "unit": "%", "values": {"2026": 76.3}, "quality": "HIGH"},
        {"metric": "Other", "unit": "x", "values": {}, "quality": "LOW"},
    ],
    "competitor_findings": [
        {"metric": "AMD Revenue CAGR", "value": 26.1, "unit": "%", "period": "history"},
        {"metric": None, "value": None},
    ],
    "valuation": [
        {"method": "DCF", "scenario": "conservative", "enterprise_value": 1.0e12,
         "assumptions": {"revenue_growth_rate": 0.1}},
        {"method": "DCF", "scenario": "aggressive", "enterprise_value": 6.0e12, "assumptions": {}},
        {"method": "EV/EBIT", "enterprise_value": 5.0e12},
        {"method": "P/E", "multiple": 46.7},
        {"method": "Other"},
    ],
    "risks": [{"risk": "Financial", "severity": "HIGH"}, {"risk": "Supply Chain", "severity": "MEDIUM"}],
    "news_events": [{"date": "2026-09-03", "label": "FACT", "text": "Deal text"}],
    "agent_status": [{"agent": "FinancialAgent", "status": "success"}],
    "verification_flags": [
        {"agent": "MarketAgent", "result": "UNSUPPORTED", "count": 6},
        {"agent": "RiskAgent", "result": "FLAGGED", "count": 1},
    ],
    "uncertainties": ["[FinancialAgent] Free Cash Flow could not be computed"],
    "missing_evidence": ["equity value"],
    "sources": [{"form": "SEC XBRL", "quality": "HIGH", "count": 10}],
}


class FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def test_small_helpers():
    assert rb._latest({"b": 2, "a": 1, "c": 3}, 2) == [("b", 2), ("c", 3)]
    assert rb._latest(None) == []
    assert rb._failed(None) == []
    assert rb._failed([{"status": "failed"}, {"status": "success"}]) == [{"status": "failed"}]
    text = rb._conflict_text({"subject": "price", "positions": [
        {"source": "user", "value": 5, "unit": "USD"}, {"agent": "A", "value": 6}, {"value": 7}]})
    assert text == "price -> user: 5 USD vs A: 6 vs ?: 7"
    assert rb._conflict_text({}) == "conflict -> "


def test_data_lines():
    text = "\n".join(rb._data_lines(CTX))
    assert "USER-PROVIDED (not verified market data)" in text
    assert "current_price = 500.0" in text
    assert "- daily change percent: 150.0 %" in text
    assert "- position state: GAIN" in text
    assert "- WARNING: One-day move" in text
    assert "Share price (close): 228.86 USD (date 2026-09-28" in text
    assert "CONFLICT (not resolved):" in text
    assert rb._data_lines({}) == ["No data available."]


def test_financial_and_market_lines():
    fin = "\n".join(rb._financial_lines(CTX))
    assert "Revenue Growth (YoY) (%) latest: 2025: 114.2, 2026: 65.5 [source quality HIGH]" in fin
    assert rb._financial_lines({}) == ["No financial metrics available."]
    assert rb._market_lines(CTX) == ["- AMD Revenue CAGR: 26.1 % (history)"]
    assert rb._market_lines({}) == ["No competitor findings available."]


def test_valuation_lines():
    assert rb._market_ev(CTX) == 5.0e12
    assert rb._market_ev({}) is None
    text = "\n".join(rb._valuation_lines(CTX))
    assert "DCF (conservative): enterprise value about 1,000 billion USD" in text
    assert "BELOW the market-implied enterprise value (0.20x)" in text
    assert "ABOVE the market-implied enterprise value (1.20x)" in text
    assert "EV/EBIT: market-implied enterprise value about 5,000 billion USD" in text
    assert "- P/E: 46.7x" in text
    assert "- Other: see agent output" in text
    assert "STORED price" in text
    no_market = "\n".join(rb._valuation_lines({"valuation": [
        {"method": "DCF", "scenario": "s", "enterprise_value": 1.0e12, "assumptions": {}},
        {"method": "DCF"}]}))
    assert "BELOW" not in no_market and "ABOVE" not in no_market
    assert "- DCF: see agent output" in no_market
    assert rb._valuation_lines({}) == ["No valuation available."]


def test_other_section_lines():
    risks = rb._risk_lines(CTX)
    assert risks[0] == "- Financial: severity HIGH"
    assert "not a measure of impact" in risks[-1]
    assert rb._risk_lines({}) == ["No risk findings available."]

    news = rb._news_lines(CTX)
    assert news[0] == "- [2026-09-03] (FACT) Deal text"
    assert "keywords" in news[-1]
    assert rb._news_lines({}) == ["No recent events available."]

    ver = "\n".join(rb._verification_lines(CTX))
    assert "- FinancialAgent: success" in ver
    assert "6 claim(s) from MarketAgent marked UNSUPPORTED" in ver
    assert "1 claim(s) from RiskAgent marked FLAGGED" in ver
    assert ver.count("Known limitation") == 1
    assert rb._verification_lines({}) == ["No verification data available."]

    runs = [{"agent": "risk", "status": "failed", "error": "boom"}]
    unc = "\n".join(rb._uncertainty_lines(CTX, runs))
    assert "Free Cash Flow" in unc
    assert "- Missing: equity value" in unc
    assert "Agent failed: risk (boom)" in unc
    assert rb._uncertainty_lines({}, []) == ["No uncertainties recorded."]

    assert rb._source_lines(CTX) == ["- SEC XBRL (quality HIGH, 10 item(s))"]
    assert rb._source_lines({}) == ["No sources recorded."]


def test_compute_confidence():
    assert rb.compute_confidence({}, [], True, []) == (0.8, ["Base score 0.8 (maximum allowed is 0.9)."])
    score, reasons = rb.compute_confidence({"conflicts": [1]}, [], True, [])
    assert score == 0.7
    assert any("conflict" in r for r in reasons)
    score, reasons = rb.compute_confidence(CTX, [{"status": "failed"}], False, ["99"])
    assert score == 0.1
    assert len(reasons) == 7


def test_position_sentence():
    assert rb._position_sentence({}) == ""
    text = rb._position_sentence(CTX)
    assert "GAIN of 316.7%" in text
    assert "typed one-day move is 150.0%" in text
    only_move = {"position_calculations": {"calculations": [
        {"metric": "daily_change_percent", "value": 5, "unit": "%"}]}}
    assert rb._position_sentence(only_move) == "The typed one-day move is 5%."
    nothing = {"position_calculations": {"calculations": [{"metric": "other", "value": 1, "unit": ""}]}}
    assert rb._position_sentence(nothing) == ""


def test_valuation_point():
    assert rb._valuation_point({}) == ""
    assert rb._valuation_point({"valuation": [{"method": "EV/EBIT", "enterprise_value": 5e12}]}) == ""
    text = rb._valuation_point(CTX)
    assert text.startswith("Valuation: 1 of 2 DCF scenarios are below")
    assert "range 0.20x to 1.20x" in text


def test_block():
    assert rb._block("T", ["a", "b"]) == "## T\na\nb\n"


def test_ask_summary(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen["json"] = json
        return FakeResp({"message": {"content": '{"summary": "Short."}'}})

    monkeypatch.setattr(rb.requests, "post", fake_post)
    assert rb.ask_summary("- point", model="m", base_url="http://h:1") == "Short."
    assert seen["json"]["think"] is False
    assert seen["json"]["model"] == "m"
    assert "POINTS:" in seen["json"]["messages"][1]["content"]


def test_ask_summary_invalid(monkeypatch):
    monkeypatch.setattr(rb.requests, "post",
                        lambda url, json=None, timeout=None: FakeResp({"message": {"content": "nope"}}))
    with pytest.raises(ValueError):
        rb.ask_summary("- point")
    monkeypatch.setattr(rb.requests, "post",
                        lambda url, json=None, timeout=None: FakeResp({"message": {"content": '{"summary": ""}'}}))
    with pytest.raises(ValueError):
        rb.ask_summary("- point")


def test_clean_summary():
    text = "Revenue is 65.5%. Price is 999 today. This looks cheap. Fine point. Another. Extra."
    cleaned, bad = rb._clean_summary(text, "growth 65.5")
    assert cleaned == "Revenue is 65.5%. Fine point. Another."
    assert bad == ["999"]
    assert rb._clean_summary("", "") == ("", [])


def test_bull_points():
    pts = rb._bull_points(CTX)
    assert pts[0].startswith("Revenue Growth (YoY): 65.5%")
    assert len(pts) == 4
    assert rb._bull_points({}) == []


def test_bear_points():
    pts = rb._bear_points(CTX)
    joined = "\n".join(pts)
    assert pts[0].startswith("Valuation: 1 of 2 DCF scenarios")
    assert "HIGH (by how many filing years mention them, not by impact): Financial." in joined
    assert "One-day move of 150.0%" in joined
    assert "Conflict, not resolved: NVDA current share price" in joined
    assert "Red-Team marked 6 claim(s) as unsupported" in joined
    assert "Free cash flow could not be computed" in joined
    assert rb._bear_points({}) == []


def test_assessment():
    text = rb._assessment(CTX, [{"status": "failed"}])
    assert "price conflict" in text
    assert "typed prices that are unverified" in text
    assert "DCF models" in text
    assert "1 agent(s) that failed" in text
    assert "Open points" not in rb._assessment({}, [])


def test_build_report_full():
    seen = {}

    def ask(points_text):
        seen["points"] = points_text
        return "Short summary."

    out = rb.build_report("My question?", CTX, runs=[], ask=ask)
    md = out["markdown"]
    for heading in ("Executive Conclusion", "AI Draft Summary (written by Qwen, may contain mistakes)",
                    "What The Data Says", "Financial Analysis", "Market Analysis", "Valuation",
                    "Risks", "Recent Events", "Verification",
                    "Bull Case (supporting points from the evidence)",
                    "Bear Case (concerns from the evidence)", "Key Uncertainties",
                    "Final Assessment", "Evidence"):
        assert "## " + heading in md
    assert "Short summary." in md
    assert "GAIN of 316.7%" in md
    assert rb.DISCLAIMER in md
    assert out["qwen_ok"] is True
    assert out["confidence"] == 0.5
    assert out["removed_numbers"] == []
    assert "- Revenue Growth" in seen["points"]


def test_build_report_qwen_down():
    def ask(points_text):
        raise RuntimeError("no ollama")

    out = rb.build_report("q", CTX, ask=ask)
    assert out["qwen_ok"] is False
    assert "Not available: Qwen could not be reached" in out["markdown"]


def test_build_report_summary_removed():
    out = rb.build_report("q", CTX, ask=lambda points_text: "Price is 999 today.")
    assert "Removed: the model's summary" in out["markdown"]
    assert out["removed_numbers"] == ["999"]


def test_build_report_empty_context():
    called = []
    out = rb.build_report("q", {}, ask=lambda points_text: called.append(points_text))
    assert called == []
    assert "No points to summarize." in out["markdown"]
    assert "This report lists the evidence found" in out["markdown"]
    assert "None found in the evidence." in out["markdown"]
    assert out["confidence"] == 0.8


def test_main_usage(monkeypatch, capsys):
    assert rb.main([]) == 1
    assert "Usage" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["x"])
    assert rb.main() == 1


def test_main_runs(monkeypatch, capsys):
    from core import dynamic_pipeline as dp

    monkeypatch.setattr(dp, "run", lambda query, synthesizer=None: {
        "context": CTX, "agent_runs": [], "errors": []})

    def boom(url, json=None, timeout=None):
        raise ConnectionError("no ollama")

    monkeypatch.setattr(rb.requests, "post", boom)
    assert rb.main(["my question"]) == 0
    printed = capsys.readouterr().out
    assert "# Investment Intelligence Report" in printed
    assert "Total time" in printed
