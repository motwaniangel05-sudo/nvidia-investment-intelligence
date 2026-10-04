"""Tests for RedTeamAgent: mechanical verification checks on other agents' findings."""

from core.schemas import Finding
from agents.red_team_agent import (
    check_historical_cagr_outlier,
    check_low_confidence,
    check_missing_disclaimers,
    check_revenue_consistency,
    check_unsupported_numeric_claims,
)


def make_finding(claim, evidence="", confidence=1.0):
    return Finding(
        claim=claim, evidence_text=evidence, source_chunk_id="c1",
        source_form="test", source_filing_date="2024-01-01", confidence=confidence,
    )


def test_check_low_confidence_flags_below_threshold():
    findings = [make_finding("Some claim", confidence=0.05)]
    results = check_low_confidence(findings, "TestAgent")
    assert len(results) == 1
    assert results[0].verification_result == "FLAGGED"


def test_check_low_confidence_does_not_flag_above_threshold():
    findings = [make_finding("Some claim", confidence=0.9)]
    results = check_low_confidence(findings, "TestAgent")
    assert len(results) == 0


def test_check_unsupported_numeric_claims_catches_missing_number():
    findings = [make_finding(
        "Revenue grew 45.8% this year",
        evidence="The company reported strong performance overall.",
    )]
    results = check_unsupported_numeric_claims(findings, "TestAgent")
    assert len(results) == 1
    assert results[0].verification_result == "UNSUPPORTED"


def test_check_unsupported_numeric_claims_passes_when_number_present():
    findings = [make_finding(
        "Revenue grew 45.8% this year",
        evidence="Revenue growth was 45.8% year over year according to the filing.",
    )]
    results = check_unsupported_numeric_claims(findings, "TestAgent")
    assert len(results) == 0


def test_check_missing_disclaimers_catches_fcf_without_capex_mention():
    findings = [make_finding(
        "FCF proxy for fiscal year was $20 billion",
        evidence="Operating cash flow was strong.",  # no "capex" mentioned
    )]
    results = check_missing_disclaimers(findings, "TestAgent")
    assert len(results) == 1
    assert "CapEx" in results[0].problem


def test_check_missing_disclaimers_passes_with_capex_mentioned():
    findings = [make_finding(
        "FCF proxy for fiscal year was $20 billion",
        evidence="This uses Operating Cash Flow since CapEx data is unavailable.",
    )]
    results = check_missing_disclaimers(findings, "TestAgent")
    assert len(results) == 0


def test_check_missing_disclaimers_catches_ev_ebit_without_ebitda_mention():
    findings = [make_finding(
        "EV/EBIT ratio: 43.0x",
        evidence="Enterprise value divided by operating income.",  # no "ebitda"
    )]
    results = check_missing_disclaimers(findings, "TestAgent")
    assert len(results) == 1
    assert "EBITDA" in results[0].problem


def test_check_historical_cagr_outlier_flags_unflagged_extreme_value():
    findings = [
        make_finding("DCF scenario 'conservative': Enterprise Value = $1,000,000,000,000 (growth rate used: 0.1)."),
        make_finding("DCF scenario 'moderate': Enterprise Value = $1,500,000,000,000 (growth rate used: 0.2)."),
        make_finding("DCF scenario 'extreme': Enterprise Value = $5,000,000,000,000 (growth rate used: 0.5)."),
    ]
    results = check_historical_cagr_outlier(findings, "TestAgent")
    assert len(results) == 1
    assert "extreme" in results[0].claim


def test_check_historical_cagr_outlier_does_not_double_flag_already_flagged():
    findings = [
        make_finding("DCF scenario 'conservative': Enterprise Value = $1,000,000,000,000 (growth rate used: 0.1)."),
        make_finding("DCF scenario 'moderate': Enterprise Value = $1,500,000,000,000 (growth rate used: 0.2)."),
        make_finding(
            "DCF scenario 'historical_cagr' (historical-CAGR extrapolation -- "
            "likely unrealistic as a base case): Enterprise Value = $5,000,000,000,000 (growth rate used: None)."
        ),
    ]
    results = check_historical_cagr_outlier(findings, "TestAgent")
    assert len(results) == 0  # already self-flagged, don't duplicate


def test_check_revenue_consistency_flags_none_growth_rate_display():
    fin_findings = [make_finding("NVDA revenue grew at a 45.8% CAGR from 2016-01-31 to 2026-01-25.")]
    val_findings = [make_finding(
        "DCF scenario 'historical_cagr': Enterprise Value = $4,805,266,116,296 (growth rate used: None)."
    )]
    results = check_revenue_consistency(fin_findings, val_findings)
    assert len(results) == 1
    assert "45.8" in results[0].correction


def test_check_revenue_consistency_no_findings_when_no_cagr_claim():
    fin_findings = [make_finding("Some unrelated financial claim.")]
    val_findings = [make_finding("DCF scenario 'historical_cagr': growth rate used: None.")]
    results = check_revenue_consistency(fin_findings, val_findings)
    assert len(results) == 0
