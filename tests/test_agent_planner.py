from core.agent_planner import PLAYBOOKS, plan_from_analysis, plan_query
from core.config_loader import load_config
from core.task_schema import Intent, PRODUCER_AGENTS, QueryAnalysis

CONFIG = load_config()


def _agents(query, **kw):
    return plan_query(query, config=CONFIG, **kw).agents


def test_financial_health_plan():
    assert set(_agents("How financially healthy is NVIDIA?")) == {
        "financial", "research", "risk", "red_team"}


def test_buy_sell_plan():
    assert set(_agents("Should I buy NVIDIA at the current price?")) == {
        "financial", "market", "valuation", "risk", "news", "red_team"}


def test_competitor_plan():
    plan = plan_query("Compare NVIDIA with AMD.", config=CONFIG)
    assert set(plan.agents) == {"research", "financial", "market", "risk", "red_team"}
    market = next(t for t in plan.tasks if t.agent == "market")
    assert market.reasoning_task == "compare NVDA against AMD"


def test_news_plan():
    assert set(_agents("What happened to NVIDIA after the latest earnings report?")) == {
        "news", "financial", "market", "research", "red_team"}


def test_valuation_plan():
    assert set(_agents("Is NVIDIA overvalued? What is its fair value?")) == {
        "valuation", "financial", "market", "red_team"}


def test_planner_never_selects_every_agent_for_these_questions():
    for q in ("How financially healthy is NVIDIA?", "Compare NVIDIA with AMD.",
              "Is NVIDIA overvalued?", "What are NVIDIA's business segments?"):
        assert set(_agents(q, include_verification=False)) != set(PRODUCER_AGENTS)


def test_red_team_runs_last_and_can_be_disabled():
    agents = _agents("How financially healthy is NVIDIA?")
    assert agents[-1] == "red_team"
    assert "red_team" not in _agents("How financially healthy is NVIDIA?", include_verification=False)


def test_secondary_intent_adds_only_its_agent():
    # news is primary (news playbook); the secondary "risk" intent adds only RiskAgent.
    plan = plan_query("What are the risks in the latest news?", config=CONFIG, include_verification=False)
    assert plan.intent == "news"
    assert plan.agents == ["research", "financial", "market", "news", "risk"]
    risk = next(t for t in plan.tasks if t.agent == "risk")
    assert risk.reason.startswith("intent 'risk' detected")


def test_user_price_noted_in_valuation_task():
    plan = plan_query("NVIDIA is trading at $500 today. Should I buy?", config=CONFIG)
    valuation = next(t for t in plan.tasks if t.agent == "valuation")
    assert "user-supplied price" in valuation.reasoning_task
    assert plan.analysis.user_prices == {"current_price": 500.0}


def test_every_task_has_a_reason_and_plan_serializes():
    plan = plan_query("Should I sell NVIDIA?", config=CONFIG)
    assert all(t.reason for t in plan.tasks)
    d = plan.to_dict()
    assert d["intent"] == "investment_decision"
    assert d["agents"] == plan.agents
    assert d["reasoning_tasks"] == plan.reasoning_tasks
    assert d["analysis"]["primary_intent"] == "investment_decision"


def test_every_intent_has_a_playbook():
    assert set(PLAYBOOKS) == set(Intent)


def test_empty_plan_has_no_red_team():
    analysis = QueryAnalysis(query="q", primary_intent=Intent.GENERAL)
    original = PLAYBOOKS[Intent.GENERAL]
    PLAYBOOKS[Intent.GENERAL] = []
    try:
        plan = plan_from_analysis(analysis)
    finally:
        PLAYBOOKS[Intent.GENERAL] = original
    assert plan.agents == []
    assert plan.to_dict()["analysis"]["primary_intent"] == "general"
