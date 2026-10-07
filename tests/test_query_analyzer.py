from core.config_loader import load_config
from core.query_analyzer import (
    analyze_query, choose_primary_intent, company_aliases, detect_intents,
    extract_tickers, extract_user_prices,
)
from core.task_schema import Intent

CONFIG = load_config()


def test_financial_health_question():
    a = analyze_query("How financially healthy is NVIDIA?", CONFIG)
    assert a.primary_intent == Intent.FINANCIAL
    assert a.target_ticker == "NVDA"
    assert "financially" in a.matched_terms["financial"]


def test_buy_question_is_investment_decision():
    a = analyze_query("Should I buy NVIDIA at the current price?", CONFIG)
    assert a.primary_intent == Intent.INVESTMENT_DECISION
    assert Intent.VALUATION in a.intents


def test_competitor_question_extracts_competitor():
    a = analyze_query("Compare NVIDIA with AMD.", CONFIG)
    assert a.primary_intent == Intent.COMPETITOR
    assert a.target_ticker == "NVDA"
    assert a.competitor_tickers == ["AMD"]


def test_naming_a_competitor_implies_comparison():
    a = analyze_query("NVIDIA and Broadcom revenue", CONFIG)
    assert Intent.COMPETITOR in a.intents
    assert a.competitor_tickers == ["AVGO"]
    assert a.matched_terms["competitor"] == ["mentions AVGO"]


def test_news_question():
    a = analyze_query("What happened to NVIDIA after the latest earnings report?", CONFIG)
    assert a.primary_intent == Intent.NEWS
    assert Intent.FINANCIAL in a.intents


def test_valuation_question():
    a = analyze_query("Is NVIDIA overvalued? What is its fair value?", CONFIG)
    assert a.primary_intent == Intent.VALUATION


def test_verification_and_general_intents():
    assert analyze_query("Please fact-check this", CONFIG).primary_intent == Intent.VERIFICATION
    a = analyze_query("hello there", CONFIG)
    assert a.primary_intent == Intent.GENERAL
    assert a.intents == []


def test_empty_query_is_general():
    assert analyze_query("", CONFIG).primary_intent == Intent.GENERAL


def test_target_is_mentioned_company_when_configured_company_absent():
    a = analyze_query("How healthy is AMD?", CONFIG)
    assert a.target_ticker == "AMD"
    assert a.competitor_tickers == []


def test_short_keywords_do_not_match_inside_words():
    # "roa" must not fire inside "broad", "eps" not inside "steps"
    assert Intent.FINANCIAL not in detect_intents("broad steps")


def test_primary_intent_priority():
    intents = {Intent.FINANCIAL: ["revenue"], Intent.INVESTMENT_DECISION: ["buy"]}
    assert choose_primary_intent(intents) == Intent.INVESTMENT_DECISION
    assert choose_primary_intent({}) == Intent.GENERAL


def test_company_aliases_include_names_and_tickers():
    aliases = company_aliases(CONFIG)
    assert aliases["nvidia"] == "NVDA"
    assert aliases["intel"] == "INTC"
    assert aliases["broadcom"] == "AVGO"
    assert aliases["amd"] == "AMD"


def test_extract_tickers_in_order_without_duplicates():
    assert extract_tickers("Intel vs NVIDIA vs NVDA vs intel", CONFIG) == ["INTC", "NVDA"]


def test_extract_user_prices_from_example_question():
    q = ("NVIDIA is trading at $500 today, yesterday it was $200, and my average "
         "purchase price is $120. Should I buy more NVIDIA or sell my current holdings?")
    assert extract_user_prices(q) == {
        "current_price": 500.0, "previous_price": 200.0, "cost_basis": 120.0,
    }


def test_extract_user_prices_handles_commas_and_ignores_unlabelled():
    assert extract_user_prices("I bought at $1,250.50") == {"cost_basis": 1250.5}
    assert extract_user_prices("my price target is $900") == {"target_price": 900.0}
    assert extract_user_prices("what about $300?") == {}
