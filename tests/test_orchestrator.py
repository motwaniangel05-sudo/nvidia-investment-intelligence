from unittest.mock import MagicMock, patch
from core import orchestrator


def _all_mocked():
    mocks = {name: MagicMock() for name in orchestrator.AGENT_CLASSES}
    for m in mocks.values():
        m.return_value.run.return_value = "mock result"
    return mocks


def test_classify_query_financial_only():
    assert orchestrator.classify_query("revenue growth margin") == ["FinancialAgent"]


def test_classify_query_multiple_agents():
    matched = orchestrator.classify_query("company overview revenue growth margin")
    assert set(matched) == {"ResearchAgent", "FinancialAgent"}


def test_classify_query_fallback_when_no_keyword_matches():
    matched = orchestrator.classify_query("zzzz qqqq xxxx")
    assert matched == ["ResearchAgent", "FinancialAgent"]


def test_run_query_runs_only_financial_agent():
    mocks = _all_mocked()
    with patch.dict(orchestrator.AGENT_CLASSES, mocks):
        orchestrator.run_query(
            company_ticker="NVDA",
            query="revenue growth margin",
            run_verification_step=False,
        )

    mocks["FinancialAgent"].return_value.run.assert_called_once()
    for name, m in mocks.items():
        if name != "FinancialAgent":
            m.return_value.run.assert_not_called()


def test_run_query_runs_multiple_matching_agents():
    mocks = _all_mocked()
    with patch.dict(orchestrator.AGENT_CLASSES, mocks):
        orchestrator.run_query(
            company_ticker="NVDA",
            query="company overview revenue growth margin",
            run_verification_step=False,
        )

    mocks["FinancialAgent"].return_value.run.assert_called_once()
    mocks["ResearchAgent"].return_value.run.assert_called_once()
    for name in ("MarketAgent", "NewsAgent", "RiskAgent", "ValuationAgent"):
        mocks[name].return_value.run.assert_not_called()


def test_run_query_runs_verification_when_enabled():
    mocks = _all_mocked()
    fake_issues = ["issue-1", "issue-2"]

    with patch.dict(orchestrator.AGENT_CLASSES, mocks), \
         patch.object(orchestrator, "run_verification", return_value=fake_issues) as mock_verify:
        response = orchestrator.run_query(
            company_ticker="NVDA",
            query="revenue growth margin",
            run_verification_step=True,
        )

    mock_verify.assert_called_once()
    passed_results = mock_verify.call_args.args[0]
    assert list(passed_results.keys()) == ["FinancialAgent"]
    assert passed_results["FinancialAgent"] == "mock result"
    assert response["verification"] == fake_issues


def test_run_query_skips_verification_when_disabled():
    mocks = _all_mocked()

    with patch.dict(orchestrator.AGENT_CLASSES, mocks), \
         patch.object(orchestrator, "run_verification") as mock_verify:
        response = orchestrator.run_query(
            company_ticker="NVDA",
            query="revenue growth margin",
            run_verification_step=False,
        )

    mock_verify.assert_not_called()
    assert response["verification"] == []


def test_run_query_skips_verification_when_no_agents_ran():
    with patch.object(orchestrator, "classify_query", return_value=[]), \
         patch.object(orchestrator, "run_verification") as mock_verify:
        response = orchestrator.run_query(
            company_ticker="NVDA",
            query="anything",
            run_verification_step=True,
        )

    mock_verify.assert_not_called()
    assert response["agent_results"] == {}
    assert response["verification"] == []


def test_run_query_response_structure_and_agent_wiring():
    mocks = _all_mocked()

    with patch.dict(orchestrator.AGENT_CLASSES, mocks):
        response = orchestrator.run_query(
            company_ticker="NVDA",
            query="revenue growth margin",
            run_verification_step=False,
        )

    assert set(response.keys()) == {
        "query", "company_ticker", "agents_activated",
        "agent_results", "verification",
    }
    assert response["query"] == "revenue growth margin"
    assert response["company_ticker"] == "NVDA"
    assert response["agents_activated"] == ["FinancialAgent"]
    assert response["agent_results"] == {"FinancialAgent": "mock result"}

    mocks["FinancialAgent"].assert_called_once_with(company_ticker="NVDA")
    mocks["FinancialAgent"].return_value.run.assert_called_once_with("revenue growth margin")


def test_keyword_agents_and_class_agents_match():
    assert set(orchestrator.AGENT_KEYWORDS) == set(orchestrator.AGENT_CLASSES)


def test_every_agent_has_at_least_one_keyword():
    for name, keywords in orchestrator.AGENT_KEYWORDS.items():
        assert keywords, f"{name} has no keywords and can never be matched"


def test_keywords_are_lowercase():
    # classify_query lowercases the query, so an uppercase keyword never matches.
    for name, keywords in orchestrator.AGENT_KEYWORDS.items():
        for kw in keywords:
            assert kw == kw.lower(), f"{name} keyword {kw!r} is not lowercase"


def test_short_keywords_do_not_match_inside_longer_words():
    assert not orchestrator._keyword_matches("roa", "broad market")
    assert not orchestrator._keyword_matches("roa", "product roadmap")
    assert not orchestrator._keyword_matches("roe", "heroes")


def test_short_keywords_still_match_as_whole_tokens():
    assert orchestrator._keyword_matches("roa", "what is the roa?")
    assert orchestrator._keyword_matches("roe", "roe and margins")
    assert orchestrator._keyword_matches("p/e", "current p/e ratio")
    assert orchestrator._keyword_matches("8-k", "latest 8-k filing")
    assert orchestrator._keyword_matches("vs", "nvda vs amd")


def test_long_keywords_remain_substring_matches():
    assert orchestrator._keyword_matches("earn", "quarterly earnings")


# ---- optional synthesis step ----
from agents.synthesis_agent import SynthesisReport
from core.schemas import AgentResult


def test_synthesis_is_off_by_default():
    mocks = _all_mocked()
    with patch.dict(orchestrator.AGENT_CLASSES, mocks), \
         patch.object(orchestrator, "synthesize") as mock_synth:
        response = orchestrator.run_query(
            company_ticker="NVDA", query="revenue growth margin",
            run_verification_step=False,
        )

    mock_synth.assert_not_called()
    assert "synthesis" not in response


def test_synthesis_runs_on_the_full_response_when_enabled():
    mocks = _all_mocked()
    keys_seen = []

    def fake_synthesize(resp):
        keys_seen.append(set(resp))
        return "REPORT"

    with patch.dict(orchestrator.AGENT_CLASSES, mocks), \
         patch.object(orchestrator, "synthesize", side_effect=fake_synthesize):
        response = orchestrator.run_query(
            company_ticker="NVDA", query="revenue growth margin",
            run_verification_step=False, synthesize_report=True,
        )

    assert keys_seen == [{"query", "company_ticker", "agents_activated",
                          "agent_results", "verification"}]
    assert response["synthesis"] == "REPORT"


def test_real_synthesis_report_is_returned_when_enabled():
    mocks = _all_mocked()
    mocks["FinancialAgent"].return_value.run.return_value = AgentResult(
        agent_name="FinancialAgent", task="t",
    )
    with patch.dict(orchestrator.AGENT_CLASSES, mocks):
        response = orchestrator.run_query(
            company_ticker="NVDA", query="revenue growth margin",
            run_verification_step=False, synthesize_report=True,
        )

    report = response["synthesis"]
    assert isinstance(report, SynthesisReport)
    assert report.overall_status == "complete"
    assert list(report.agents) == ["FinancialAgent"]
