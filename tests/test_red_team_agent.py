from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents import red_team_agent as rt

CHECKS = [
    "check_low_confidence",
    "check_unsupported_numeric_claims",
    "check_missing_disclaimers",
    "check_historical_cagr_outlier",
    "check_revenue_consistency",
]


def _finding(confidence, claim="claim", evidence="evidence"):
    return SimpleNamespace(claim=claim, evidence_text=evidence, confidence=confidence)


def _result(*findings):
    return SimpleNamespace(findings=list(findings))


@pytest.fixture
def checks():
    """Replace every check with a mock returning [] so run_verification's wiring is tested alone."""
    with ExitStack() as stack:
        yield {n: stack.enter_context(patch.object(rt, n, return_value=[])) for n in CHECKS}


# ---- check_low_confidence (real logic) ----

def test_low_confidence_flags_finding_below_threshold():
    out = rt.check_low_confidence([_finding(0.05, claim="weak claim")], "NewsAgent")
    assert len(out) == 1
    assert out[0].claim == "weak claim"
    assert out[0].verification_result == "FLAGGED"
    assert out[0].confidence == 0.05
    assert out[0].source_agent == "NewsAgent"


def test_low_confidence_does_not_flag_at_or_above_threshold():
    findings = [_finding(rt.LOW_CONFIDENCE_THRESHOLD), _finding(0.5), _finding(0.99)]
    assert rt.check_low_confidence(findings, "NewsAgent") == []


def test_low_confidence_truncates_evidence_to_200_chars():
    out = rt.check_low_confidence([_finding(0.0, evidence="x" * 500)], "NewsAgent")
    assert len(out[0].evidence) == 200


def test_low_confidence_empty_input():
    assert rt.check_low_confidence([], "NewsAgent") == []


# ---- run_verification (wiring) ----

def test_run_verification_empty_input(checks):
    assert rt.run_verification({}) == []
    for name in CHECKS:
        checks[name].assert_not_called()


def test_run_verification_runs_every_per_agent_check_for_each_agent(checks):
    news = _result(_finding(0.5))
    risk = _result(_finding(0.5))
    rt.run_verification({"NewsAgent": news, "RiskAgent": risk})

    for name in CHECKS[:4]:
        assert checks[name].call_count == 2
        checks[name].assert_any_call(news.findings, "NewsAgent")
        checks[name].assert_any_call(risk.findings, "RiskAgent")
    checks["check_revenue_consistency"].assert_not_called()


def test_run_verification_aggregates_results_in_agent_order(checks):
    checks["check_low_confidence"].side_effect = lambda findings, name: [f"lc-{name}"]
    checks["check_missing_disclaimers"].side_effect = lambda findings, name: [f"disc-{name}"]

    out = rt.run_verification({"A": _result(), "B": _result()})
    assert out == ["lc-A", "disc-A", "lc-B", "disc-B"]


def test_revenue_consistency_runs_only_when_both_agents_present(checks):
    rt.run_verification({"FinancialAgent": _result()})
    rt.run_verification({"ValuationAgent": _result()})
    checks["check_revenue_consistency"].assert_not_called()


def test_revenue_consistency_receives_both_agents_findings(checks):
    fin = _result(_finding(0.5, claim="fin"))
    val = _result(_finding(0.5, claim="val"))
    checks["check_revenue_consistency"].return_value = ["rev-issue"]

    out = rt.run_verification({"FinancialAgent": fin, "ValuationAgent": val})

    checks["check_revenue_consistency"].assert_called_once_with(fin.findings, val.findings)
    assert out[-1] == "rev-issue"
