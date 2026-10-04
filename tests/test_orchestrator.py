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
