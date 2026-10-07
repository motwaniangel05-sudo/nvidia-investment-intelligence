import sys
import time
from types import SimpleNamespace as NS

import pytest

from core import dynamic_pipeline as dp


class FakeResponse:
    def __init__(self, plan):
        self.plan = plan
        self.runs = []
        self.agent_results = {}
        self.standard_results = {}
        self.evidence_context = None

    def to_dict(self):
        return {"results": dict(self.standard_results)}


class FakeContext:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return self.data


def make_plan(agents=None, prices=None):
    agents = list(agents or ["financial", "valuation", dp.do.RED_TEAM])
    tasks = [NS(agent=a, reasoning_task="task " + a) for a in agents]
    return NS(query="Should I buy?", company_ticker="NVDA", intent="investment_decision",
              tasks=tasks, agents=agents, analysis=NS(user_prices=prices))


def make_registry(*names):
    return {n: NS(key=n, agent_name=n.capitalize() + "Agent") for n in names}


def good_synth(query, context):
    return {"answer": "ok", "source": "qwen"}


@pytest.fixture
def env(monkeypatch):
    state = {"plan": make_plan(), "red_team_calls": 0, "agg_args": None,
             "fail_agents": set(), "slow_agents": set(), "agg_error": None}

    def fake_plan_query(query, config=None, include_verification=True):
        return state["plan"]

    def fake_run_agent(adapter, context):
        if adapter.key in state["slow_agents"]:
            time.sleep(0.4)
        if adapter.key in state["fail_agents"]:
            raise RuntimeError("agent crashed")
        run = dp.do.AgentRun(adapter.key, adapter.agent_name, "success", 0.1)
        return NS(name=adapter.agent_name), run

    def fake_aggregate(query, results, **kwargs):
        if state["agg_error"]:
            raise RuntimeError(state["agg_error"])
        state["agg_args"] = (query, list(results), kwargs)
        return FakeContext({"query": query, "sources": [{"form": "SEC"}],
                            "conflicts": ["c"], "verification_flags": ["v"]})

    def fake_red_team(response):
        state["red_team_calls"] += 1

    monkeypatch.setattr(dp.do, "plan_query", fake_plan_query)
    monkeypatch.setattr(dp.do, "DynamicResponse", FakeResponse)
    monkeypatch.setattr(dp.do, "_run_agent", fake_run_agent)
    monkeypatch.setattr(dp.do, "normalize_result", lambda r: {"normalized": r.name})
    monkeypatch.setattr(dp.do, "aggregate_evidence", fake_aggregate)
    monkeypatch.setattr(dp.do, "_run_red_team", fake_red_team)
    return state


def test_run_full_flow(env):
    env["plan"] = make_plan(prices={"current_price": 500.0, "cost_basis": 120.0})
    events = []
    out = dp.run("Should I buy?", registry=make_registry("financial", "valuation"),
                 synthesizer=good_synth, progress=lambda stage, info: events.append(stage))
    assert out["selected_agents"] == env["plan"].agents
    assert sorted(r["agent"] for r in out["agent_runs"]) == ["financial", "valuation"]
    assert out["errors"] == []
    assert out["final_response"]["answer"] == "ok"
    assert out["sources"] == [{"form": "SEC"}]
    assert out["verification"] == ["v"]
    assert out["context"]["position_calculations"]["inputs"]["current_price"] == 500.0
    assert env["red_team_calls"] == 1
    assert env["agg_args"][0] == "Should I buy?"
    assert len(env["agg_args"][1]) == 2
    assert events == ["plan", "agents", "evidence"]
    assert out["metadata"]["parallel_agents"] is True
    assert out["metadata"]["agent_timeout"] == dp.AGENT_TIMEOUT


def test_ticker_override(env):
    dp.run("q", company_ticker=" amd ", registry=make_registry("financial", "valuation"),
           synthesizer=good_synth)
    assert env["plan"].company_ticker == "AMD"
    assert env["agg_args"][2]["company_ticker"] == "AMD"


def test_missing_adapter(env):
    out = dp.run("q", registry=make_registry("financial"), synthesizer=good_synth)
    assert "valuation: no adapter registered" in out["errors"]


def test_agent_exception(env):
    env["fail_agents"] = {"valuation"}
    out = dp.run("q", registry=make_registry("financial", "valuation"), synthesizer=good_synth)
    assert "valuation: agent crashed" in out["errors"]
    assert env["red_team_calls"] == 1


def test_agent_timeout(env):
    env["slow_agents"] = {"valuation"}
    out = dp.run("q", registry=make_registry("financial", "valuation"),
                 synthesizer=good_synth, timeout=0.05)
    assert "valuation: timed out after 0.05s" in out["errors"]
    assert env["red_team_calls"] == 1


def test_no_red_team_when_all_agents_fail(env):
    env["fail_agents"] = {"financial", "valuation"}
    out = dp.run("q", registry=make_registry("financial", "valuation"), synthesizer=good_synth)
    assert env["red_team_calls"] == 0
    assert len(out["errors"]) == 2


def test_plan_with_only_red_team(env):
    env["plan"] = make_plan(agents=[dp.do.RED_TEAM])
    out = dp.run("q", registry={}, synthesizer=good_synth)
    assert out["agent_runs"] == []
    assert env["red_team_calls"] == 0
    assert out["final_response"]["answer"] == "ok"


def test_aggregator_failure(env):
    env["agg_error"] = "bad"
    seen = {}

    def synth(query, context):
        seen["context"] = context
        return {"answer": "ok", "source": "qwen"}

    out = dp.run("Should I buy?", registry=make_registry("financial", "valuation"), synthesizer=synth)
    assert "aggregator: bad" in out["errors"]
    assert seen["context"] == {"query": "Should I buy?"}
    assert out["sources"] == []
    assert out["verification"] is None


def test_position_calculator_failure(env, monkeypatch):
    env["plan"] = make_plan(prices={"current_price": 1.0})

    def boom(prices):
        raise RuntimeError("boom")

    monkeypatch.setattr(dp, "calculate_position", boom)
    out = dp.run("q", registry=make_registry("financial", "valuation"), synthesizer=good_synth)
    assert "position calculator: boom" in out["errors"]


def test_synthesizer_raises(env):
    def synth(query, context):
        raise ValueError("nope")

    out = dp.run("q", registry=make_registry("financial", "valuation"), synthesizer=synth)
    assert out["final_response"] == {"answer": "Synthesis failed.", "source": "none", "confidence": 0.0}
    assert "synthesis: nope" in out["errors"]


def test_fallback_is_reported(env):
    out = dp.run("q", registry=make_registry("financial", "valuation"),
                 synthesizer=lambda q, c: {"source": "fallback", "answer": "x"})
    assert "qwen: unavailable or invalid output, fallback answer used" in out["errors"]


def test_notify():
    dp._notify(None, "plan", {})
    seen = []
    dp._notify(lambda stage, info: seen.append((stage, info)), "plan", 1)
    assert seen == [("plan", 1)]

    def boom(stage, info):
        raise RuntimeError("x")

    dp._notify(boom, "plan", 1)


def test_main_usage(monkeypatch, capsys):
    assert dp.main([]) == 1
    assert "Usage" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["x"])
    assert dp.main() == 1


def test_main_prints_result(monkeypatch, capsys):
    out = {
        "agent_runs": [
            {"agent": "financial", "status": "success", "duration_seconds": 0.1, "error": ""},
            {"agent": "risk", "status": "failed", "duration_seconds": 0.0, "error": "boom"},
        ],
        "final_response": {"source": "qwen", "answer": "Hello", "recommendation": "Rec",
                           "risks": ["r1"], "uncertainties": ["u1"], "assumptions": [],
                           "confidence": 0.4},
        "errors": ["risk: boom"],
        "metadata": {"total_seconds": 1},
    }
    monkeypatch.setattr(dp, "run", lambda query: out)
    assert dp.main(["question"]) == 0
    printed = capsys.readouterr().out
    for text in ("=== SELECTED AGENTS ===", "financial: success", "risk: failed",
                 "FINAL ANSWER (qwen)", "Hello", "Recommendation: Rec", "RISKS:", "- r1",
                 "Confidence: 0.4", "risk: boom"):
        assert text in printed
