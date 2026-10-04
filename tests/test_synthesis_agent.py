"""Tests for SynthesisAgent: mechanical combination of agent results."""

from agents.red_team_agent import VerificationResult, run_verification
from agents.synthesis_agent import SynthesisReport, synthesize
from core.schemas import AgentResult, Finding


def _f(claim, confidence=1.0):
    return Finding(
        claim=claim, evidence_text="e", source_chunk_id="c",
        source_form="XBRL_metrics", source_filing_date="2024-01-28",
        confidence=confidence,
    )


def _result(name, findings=(), status="success", warnings=()):
    return AgentResult(
        agent_name=name, task="t", findings=list(findings),
        status=status, warnings=list(warnings),
    )


def _v(claim, agent, result="FLAGGED"):
    return VerificationResult(
        claim=claim, evidence="e", verification_result=result,
        confidence=0.5, problem="p", correction="c", source_agent=agent,
    )


def _resp(agent_results, verification=()):
    return {
        "query": "q", "company_ticker": "NVDA",
        "agents_activated": list(agent_results),
        "agent_results": agent_results, "verification": list(verification),
    }


# ---- overall status ----

def test_status_complete_when_every_agent_succeeded():
    r = synthesize(_resp({"A": _result("A"), "B": _result("B")}))
    assert r.overall_status == "complete"


def test_status_partial_when_any_agent_is_partial():
    r = synthesize(_resp({"A": _result("A"), "B": _result("B", status="partial")}))
    assert r.overall_status == "partial"


def test_status_partial_when_some_but_not_all_failed():
    r = synthesize(_resp({"A": _result("A"), "B": _result("B", status="failed")}))
    assert r.overall_status == "partial"


def test_status_failed_when_every_agent_failed():
    r = synthesize(_resp({"A": _result("A", status="failed")}))
    assert r.overall_status == "failed"


def test_empty_response_does_not_crash_and_is_failed():
    r = synthesize({})
    assert isinstance(r, SynthesisReport)
    assert r.overall_status == "failed"
    assert r.agents == {}
    assert r.verification_summary == {"total_issues": 0, "by_result": {}}
    assert r.caveats == []


# ---- headline findings ----

def test_headlines_are_top_n_by_confidence():
    findings = [_f("low", 0.2), _f("high", 1.0), _f("mid", 0.5), _f("second", 0.9)]
    r = synthesize(_resp({"A": _result("A", findings)}), max_findings_per_agent=2)
    claims = [h["claim"] for h in r.agents["A"]["headline_findings"]]
    assert claims == ["high", "second"]
    assert r.agents["A"]["finding_count"] == 4


def test_default_is_three_headlines_per_agent():
    findings = [_f(f"c{i}", 1.0 - i / 10) for i in range(5)]
    r = synthesize(_resp({"A": _result("A", findings)}))
    assert len(r.agents["A"]["headline_findings"]) == 3


def test_agent_with_no_findings_has_empty_headlines():
    r = synthesize(_resp({"A": _result("A", status="partial")}))
    assert r.agents["A"]["headline_findings"] == []
    assert r.agents["A"]["finding_count"] == 0


# ---- verification flags ----

def test_flagged_finding_is_marked_and_counted():
    findings = [_f("bad claim"), _f("good claim", 0.9)]
    r = synthesize(_resp({"A": _result("A", findings)}, [_v("bad claim", "A", "UNSUPPORTED")]))
    heads = {h["claim"]: h for h in r.agents["A"]["headline_findings"]}
    assert heads["bad claim"]["issues"] == ["UNSUPPORTED"]
    assert heads["good claim"]["issues"] == []
    assert r.agents["A"]["flagged_finding_count"] == 1


def test_flag_only_applies_to_the_agent_that_raised_it():
    results = {"A": _result("A", [_f("same claim")]), "B": _result("B", [_f("same claim")])}
    r = synthesize(_resp(results, [_v("same claim", "B")]))
    assert r.agents["A"]["flagged_finding_count"] == 0
    assert r.agents["B"]["flagged_finding_count"] == 1


def test_supported_verification_is_not_treated_as_a_problem():
    r = synthesize(_resp({"A": _result("A", [_f("x")])}, [_v("x", "A", "SUPPORTED")]))
    assert r.agents["A"]["flagged_finding_count"] == 0
    assert r.agents["A"]["headline_findings"][0]["issues"] == []


def test_verification_summary_counts_by_result():
    ver = [_v("a", "A", "FLAGGED"), _v("b", "A", "FLAGGED"), _v("c", "A", "UNSUPPORTED")]
    r = synthesize(_resp({"A": _result("A")}, ver))
    assert r.verification_summary == {
        "total_issues": 3, "by_result": {"FLAGGED": 2, "UNSUPPORTED": 1},
    }


# ---- caveats ----

def test_caveats_are_prefixed_with_agent_name():
    r = synthesize(_resp({"A": _result("A", warnings=["watch out"])}))
    assert r.caveats == ["[A] watch out"]


def test_duplicate_warning_within_one_agent_appears_once():
    r = synthesize(_resp({"A": _result("A", warnings=["same", "same"])}))
    assert r.caveats == ["[A] same"]


def test_same_warning_from_two_agents_is_kept_for_both():
    results = {"A": _result("A", warnings=["same"]), "B": _result("B", warnings=["same"])}
    r = synthesize(_resp(results))
    assert r.caveats == ["[A] same", "[B] same"]


def test_redteam_caveat_added_only_when_verification_issues_exist():
    clean = synthesize(_resp({"A": _result("A")}))
    assert not any("RedTeam" in c for c in clean.caveats)

    flagged = synthesize(_resp({"A": _result("A")}, [_v("x", "A")]))
    assert any("[RedTeam] 1 issue(s)" in c for c in flagged.caveats)


def test_agent_warnings_are_copied_not_shared():
    original = _result("A", warnings=["w"])
    r = synthesize(_resp({"A": original}))
    r.agents["A"]["warnings"].append("mutated")
    assert original.warnings == ["w"]


# ---- text and metadata ----

def test_report_carries_ticker_and_query():
    r = synthesize(_resp({"A": _result("A")}))
    assert r.company_ticker == "NVDA"
    assert r.query == "q"


def test_text_shows_status_agents_flags_and_caveats():
    results = {"A": _result("A", [_f("bad claim")], warnings=["careful"])}
    r = synthesize(_resp(results, [_v("bad claim", "A")]))
    assert "Overall status: complete" in r.text
    assert "A (success, 1 findings, 1 flagged)" in r.text
    assert "- bad claim (confidence 1.0) [FLAGGED]" in r.text
    assert "Verification: 1 issue(s)" in r.text
    assert "* [A] careful" in r.text


def test_text_omits_caveats_section_when_there_are_none():
    r = synthesize(_resp({"A": _result("A")}))
    assert "Caveats:" not in r.text


# ---- works with the real Red-Team output ----

def test_works_with_real_run_verification_output():
    weak = _f("weak finding", confidence=0.05)
    results = {"NewsAgent": _result("NewsAgent", [weak])}
    r = synthesize(_resp(results, run_verification(results)))
    assert r.agents["NewsAgent"]["flagged_finding_count"] == 1
    assert r.verification_summary["by_result"] == {"FLAGGED": 1}


# ---- ties in confidence: newest finding first ----

def _dated(claim, confidence, date):
    return Finding(
        claim=claim, evidence_text="e", source_chunk_id="c",
        source_form="XBRL_metrics", source_filing_date=date, confidence=confidence,
    )


def _heads(findings, n=3):
    r = synthesize(_resp({"A": _result("A", findings)}), max_findings_per_agent=n)
    return [h["claim"] for h in r.agents["A"]["headline_findings"]]


def test_equal_confidence_prefers_most_recent_filing_date():
    findings = [
        _dated("old", 1.0, "2017-01-29"),
        _dated("newest", 1.0, "2026-01-25"),
        _dated("middle", 1.0, "2020-01-26"),
    ]
    assert _heads(findings, 2) == ["newest", "middle"]


def test_missing_date_ranks_below_real_dates():
    findings = [_dated("no date", 1.0, "n/a"), _dated("dated", 1.0, "2016-01-31")]
    assert _heads(findings, 1) == ["dated"]


def test_confidence_still_beats_recency():
    findings = [_dated("new but weaker", 0.5, "2026-01-25"), _dated("old but stronger", 1.0, "2016-01-31")]
    assert _heads(findings, 1) == ["old but stronger"]
