import json
import logging

import pytest

from core import dynamic_orchestrator as dyn
from core.config_loader import load_config
from core.schemas import AgentResult, Finding
from core.task_schema import ExecutionPlan, AgentTask

CONFIG = load_config()


class FakeAgent:
    """Stands in for an existing BaseAgent subclass; records calls."""
    calls = []
    agent_name = "FakeAgent"
    status = "success"

    def __init__(self, company_ticker):
        self.company_ticker = company_ticker

    def run(self, task):
        FakeAgent.calls.append((self.agent_name, self.company_ticker, task))
        return AgentResult(
            agent_name=self.agent_name, task=task, status=self.status,
            findings=[Finding(f"{self.agent_name} claim", "evidence", "id", "form", "2026-01-25", 1.0)],
        )


def _fake_class(name, status="success"):
    return type(name, (FakeAgent,), {"agent_name": name, "status": status})


def _registry(failing=None, raising=None):
    reg = {}
    for key, adapter in dyn.AGENT_REGISTRY.items():
        name = adapter.agent_name
        if key == raising:
            cls = type(name, (), {"agent_name": name, "__init__": _boom})
        else:
            cls = _fake_class(name, "failed" if key == failing else "success")
        reg[key] = dyn.AgentAdapter(key, cls)
    return reg


def _boom(self, company_ticker):
    raise RuntimeError("constructor exploded")


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    FakeAgent.calls = []
    monkeypatch.setattr(dyn, "run_verification", lambda results: ["issue"] * len(results))


def _ran(response):
    return [r.agent_name for r in response.runs]


def test_financial_health_question_runs_only_selected_agents():
    r = dyn.run_dynamic_query("How financially healthy is NVIDIA?", config=CONFIG, registry=_registry())
    assert _ran(r) == ["ResearchAgent", "FinancialAgent", "RiskAgent", "RedTeamAgent"]
    assert set(r.agent_results) == {"ResearchAgent", "FinancialAgent", "RiskAgent"}
    assert r.verification == ["issue"] * 3


def test_buy_sell_question():
    r = dyn.run_dynamic_query("Should I buy NVIDIA at the current price?", config=CONFIG, registry=_registry())
    assert set(_ran(r)) == {"FinancialAgent", "MarketAgent", "ValuationAgent",
                            "RiskAgent", "NewsAgent", "RedTeamAgent"}
    assert "ResearchAgent" not in r.agent_results


def test_competitor_question():
    r = dyn.run_dynamic_query("Compare NVIDIA with AMD.", config=CONFIG, registry=_registry())
    assert set(r.agent_results) == {"ResearchAgent", "FinancialAgent", "MarketAgent", "RiskAgent"}
    assert r.plan.analysis.competitor_tickers == ["AMD"]


def test_news_question():
    r = dyn.run_dynamic_query("What happened to NVIDIA after the latest earnings report?",
                              config=CONFIG, registry=_registry())
    assert set(r.agent_results) == {"NewsAgent", "FinancialAgent", "MarketAgent", "ResearchAgent"}


def test_valuation_question():
    r = dyn.run_dynamic_query("Is NVIDIA overvalued? What is its fair value?",
                              config=CONFIG, registry=_registry())
    assert set(r.agent_results) == {"ValuationAgent", "FinancialAgent", "MarketAgent"}


def test_agents_receive_query_and_ticker():
    dyn.run_dynamic_query("How healthy is AMD?", config=CONFIG, registry=_registry())
    assert {c[1] for c in FakeAgent.calls} == {"AMD"}
    assert {c[2] for c in FakeAgent.calls} == {"How healthy is AMD?"}


def test_ticker_override():
    dyn.run_dynamic_query("How healthy is NVIDIA?", company_ticker=" intc ", config=CONFIG, registry=_registry())
    assert {c[1] for c in FakeAgent.calls} == {"INTC"}


def test_logging_shows_query_plan_and_agent_lifecycle(caplog):
    with caplog.at_level(logging.INFO):
        dyn.run_dynamic_query("Is NVIDIA overvalued?", config=CONFIG, registry=_registry(failing="market"))
    text = caplog.text
    assert "[ORCHESTRATOR] Query: Is NVIDIA overvalued?" in text
    assert "[PLANNER] Selected: financial, market, valuation, red_team" in text
    assert "[AGENT] FinancialAgent started" in text
    assert "[AGENT] FinancialAgent completed" in text
    assert "[AGENT] MarketAgent error" in text
    assert "[AGENT] RedTeamAgent completed" in text


def test_failed_agent_does_not_stop_others():
    r = dyn.run_dynamic_query("Is NVIDIA overvalued?", config=CONFIG, registry=_registry(failing="market"))
    runs = {x.agent_name: x for x in r.runs}
    assert runs["MarketAgent"].status == "failed"
    assert runs["MarketAgent"].error
    assert runs["ValuationAgent"].status == "success"


def test_adapter_exception_becomes_failed_result():
    r = dyn.run_dynamic_query("Is NVIDIA overvalued?", config=CONFIG, registry=_registry(raising="valuation"))
    result = r.agent_results["ValuationAgent"]
    assert result.status == "failed"
    assert "constructor exploded" in result.warnings[0]
    assert {x.agent_name: x.error for x in r.runs}["ValuationAgent"] == "constructor exploded"


def test_missing_adapter_is_recorded(caplog):
    reg = _registry()
    del reg["valuation"]
    with caplog.at_level(logging.ERROR):
        r = dyn.run_dynamic_query("Is NVIDIA overvalued?", config=CONFIG, registry=reg)
    assert {x.agent: x.error for x in r.runs}["valuation"] == "no adapter registered"
    assert "[AGENT] valuation error: no adapter registered" in caplog.text


def test_red_team_failure_is_logged_not_raised(monkeypatch, caplog):
    def broken(_):
        raise ValueError("verifier down")
    monkeypatch.setattr(dyn, "run_verification", broken)
    with caplog.at_level(logging.ERROR):
        r = dyn.run_dynamic_query("Is NVIDIA overvalued?", config=CONFIG, registry=_registry())
    assert r.verification == []
    assert r.runs[-1].agent_name == "RedTeamAgent" and r.runs[-1].status == "failed"
    assert "[AGENT] RedTeamAgent error: verifier down" in caplog.text


def test_no_verification_skips_red_team():
    r = dyn.run_dynamic_query("Is NVIDIA overvalued?", config=CONFIG, registry=_registry(),
                              run_verification_step=False)
    assert "RedTeamAgent" not in _ran(r)
    assert r.verification == []


def test_synthesis_uses_existing_synthesis_agent(monkeypatch):
    monkeypatch.setattr(dyn, "run_verification", lambda results: [])
    r = dyn.run_dynamic_query("Is NVIDIA overvalued?", config=CONFIG, registry=_registry(),
                              synthesize_report=True)
    assert r.synthesis.overall_status == "complete"
    assert set(r.synthesis.agents) == set(r.agent_results)


def test_legacy_dict_matches_run_query_shape():
    r = dyn.run_dynamic_query("Is NVIDIA overvalued?", config=CONFIG, registry=_registry())
    d = r.to_legacy_dict()
    assert set(d) == {"query", "company_ticker", "agents_activated", "agent_results", "verification"}
    assert d["agents_activated"] == list(r.agent_results)


def test_execute_plan_skips_red_team_with_no_results():
    plan = ExecutionPlan(query="q", intent="general", company_ticker="NVDA",
                         tasks=[AgentTask("red_team", "verify", "test")])
    r = dyn.execute_plan(plan, registry={})
    assert r.runs == [] and r.verification == []


def test_default_registry_wraps_existing_agents():
    names = {k: a.agent_name for k, a in dyn.AGENT_REGISTRY.items()}
    assert names == {
        "research": "ResearchAgent", "financial": "FinancialAgent", "market": "MarketAgent",
        "news": "NewsAgent", "risk": "RiskAgent", "valuation": "ValuationAgent",
    }


def test_format_response():
    r = dyn.run_dynamic_query("Is NVIDIA overvalued?", config=CONFIG, registry=_registry(failing="market"))
    text = dyn.format_response(r)
    assert "Intent: valuation | Ticker: NVDA" in text
    assert "valuation: evaluate current valuation" in text
    assert "MarketAgent: failed" in text
    assert "Red-Team issues: 3" in text


def test_cli_plan_only(capsys):
    assert dyn.main(["--plan-only", "--ticker", "amd", "Compare", "NVIDIA", "with", "AMD"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["intent"] == "competitor"
    assert out["company_ticker"] == "AMD"
    assert out["agents"][-1] == "red_team"


def test_cli_run(capsys):
    assert dyn.main(["--no-verify", "Is", "NVIDIA", "overvalued?"], registry=_registry()) == 0
    out = capsys.readouterr().out
    assert "Query: Is NVIDIA overvalued?" in out
    assert "Red-Team" not in out
