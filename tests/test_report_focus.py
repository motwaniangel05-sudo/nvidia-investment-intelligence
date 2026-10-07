from core import report_builder as rb

RISKS = [{"risk": "Financial", "severity": "HIGH"}, {"risk": "Supply Chain", "severity": "MEDIUM"}]
VALUATION = [
    {"method": "DCF", "scenario": "conservative", "enterprise_value": 1.0e12, "assumptions": {}},
    {"method": "EV/EBIT", "enterprise_value": 5.0e12},
    {"method": "P/E", "multiple": 46.7},
]
COMPETITORS = [
    {"metric": "AMD Revenue CAGR", "value": 26.1, "unit": "%", "period": "history"},
    {"metric": None, "value": None},
]
NEWS = [{"date": "2026-09-03", "text": "Deal text"}]
FIN = [{"metric": "Net Margin", "unit": "%", "values": {"2026": 55.6}}]


def test_risk_focus():
    pts = rb._focus_points({"intent": "risk", "risks": RISKS})
    assert len(pts) == 2
    assert pts[0].startswith("Risk category Financial: severity HIGH")


def test_valuation_focus():
    pts = rb._focus_points({"intent": "valuation", "valuation": VALUATION})
    assert pts[0].startswith("Valuation: 1 of 1 DCF scenarios")
    assert "DCF results depend on fixed assumptions." in pts
    assert "P/E: 46.7x, based on the stored share price." in pts


def test_valuation_without_data_falls_back():
    assert rb._focus_points({"intent": "valuation"}) == []


def test_competitor_and_market_focus():
    for intent in ("competitor", "market"):
        pts = rb._focus_points({"intent": intent, "competitor_findings": COMPETITORS})
        assert pts == ["AMD Revenue CAGR: 26.1 % (history)."]


def test_news_focus():
    pts = rb._focus_points({"intent": "news", "news_events": NEWS})
    assert pts[0] == "[2026-09-03] Deal text"
    assert "may not be relevant" in pts[-1]


def test_financial_and_research_focus():
    for intent in ("financial", "research"):
        pts = rb._focus_points({"intent": intent, "financial_metrics": FIN})
        assert pts[0].startswith("Net Margin: 55.6%")


def test_default_focus_uses_bull_and_bear_points():
    ctx = {"financial_metrics": FIN, "risks": RISKS}
    assert rb._focus_points(ctx) == rb._bull_points(ctx) + rb._bear_points(ctx)
    assert rb._focus_points({"intent": "risk"}) == []


def test_question_is_passed_to_qwen():
    seen = {}

    def ask(points_text):
        seen["text"] = points_text
        return "Short."

    rb.build_report("What are the risks?", {"intent": "risk", "risks": RISKS}, ask=ask)
    assert seen["text"].startswith("QUESTION: What are the risks?\n- Risk category Financial")


def test_summary_drops_ratio_words_and_market_share():
    text = "NVIDIA has a three-to-one market share advantage. Fine point. It is twice as large."
    cleaned, bad = rb._clean_summary(text, "")
    assert cleaned == "Fine point."
    assert bad == []
