from agents import red_team_agent as rt
from core.schemas import Finding


def _f(claim, evidence="", confidence=0.5):
    return Finding(
        claim=claim,
        evidence_text=evidence,
        source_chunk_id="chunk-1",
        source_form="10-K",
        source_filing_date="2024-01-01",
        confidence=confidence,
    )


# ---- check_unsupported_numeric_claims ----

def test_number_present_in_evidence_is_not_flagged():
    f = _f("Revenue was $4,805 million", "total revenue of 4,805 million")
    assert rt.check_unsupported_numeric_claims([f], "FinancialAgent") == []


def test_number_missing_from_evidence_is_flagged_unsupported():
    f = _f("Revenue was $4,805 million", "no figures here")
    out = rt.check_unsupported_numeric_claims([f], "FinancialAgent")
    assert len(out) == 1
    assert out[0].verification_result == "UNSUPPORTED"
    assert out[0].source_agent == "FinancialAgent"
    assert out[0].claim == f.claim


def test_claim_without_numbers_is_skipped():
    assert rt.check_unsupported_numeric_claims([_f("Margins improved", "x")], "A") == []


def test_single_digit_numbers_are_ignored():
    # _clean_numbers drops 1-character numbers, so this claim has no checkable numbers
    assert rt.check_unsupported_numeric_claims([_f("Company has 3 segments", "x")], "A") == []


def test_trailing_punctuation_is_stripped_before_comparing():
    f = _f("Revenue was 4,805.", "total 4,805 million")
    assert rt.check_unsupported_numeric_claims([f], "A") == []


def test_any_overlapping_number_counts_as_support():
    # Documented behavior ("SOME number"): the shared year hides the unsupported 4,805
    f = _f("Revenue was 4,805 in 2023", "2023 annual filing")
    assert rt.check_unsupported_numeric_claims([f], "A") == []


def test_percent_sign_must_match_between_claim_and_evidence():
    # Current behavior: "45.8%" and "45.8" are different tokens, so this is flagged
    f = _f("Revenue grew 45.8%", "grew 45.8 percent")
    assert len(rt.check_unsupported_numeric_claims([f], "A")) == 1


def test_unsupported_evidence_truncated_to_200_chars():
    f = _f("Revenue was 4,805", "y" * 500)
    out = rt.check_unsupported_numeric_claims([f], "A")
    assert len(out[0].evidence) == 200


# ---- check_missing_disclaimers ----

def test_fcf_without_capex_disclaimer_is_flagged():
    out = rt.check_missing_disclaimers([_f("NVDA FCF margin was strong", "cash flow rose")], "FinancialAgent")
    assert len(out) == 1
    assert out[0].verification_result == "FLAGGED"
    assert "CAPEX" in out[0].correction


def test_fcf_with_capex_in_evidence_is_not_flagged():
    f = _f("NVDA FCF margin was strong", "FCF proxy; CapEx data unavailable")
    assert rt.check_missing_disclaimers([f], "A") == []


def test_ev_ebit_without_ebitda_disclaimer_is_flagged():
    out = rt.check_missing_disclaimers([_f("EV/EBIT is 30x", "multiple computed")], "ValuationAgent")
    assert len(out) == 1
    assert "EBITDA" in out[0].correction


def test_ev_ebit_with_ebitda_in_claim_is_not_flagged():
    f = _f("EV/EBIT is 30x (EBITDA substituted for EBIT)", "x")
    assert rt.check_missing_disclaimers([f], "A") == []


def test_finding_missing_both_disclaimers_gets_two_results():
    out = rt.check_missing_disclaimers([_f("FCF and EV/EBIT both look fine", "x")], "A")
    assert len(out) == 2


def test_unrelated_finding_gets_no_disclaimer_check():
    assert rt.check_missing_disclaimers([_f("Revenue grew", "x")], "A") == []


# ---- check_historical_cagr_outlier ----

def _dcf(ev, extra=""):
    return _f(f"DCF scenario base: Enterprise Value = ${ev:,}{extra}")


def test_outlier_scenario_more_than_2x_median_is_flagged():
    out = rt.check_historical_cagr_outlier([_dcf(100), _dcf(110), _dcf(500)], "ValuationAgent")
    assert len(out) == 1
    assert "500" in out[0].claim
    assert out[0].verification_result == "FLAGGED"
    assert out[0].source_agent == "ValuationAgent"


def test_outlier_already_marked_unrealistic_is_not_flagged():
    findings = [_dcf(100), _dcf(110), _dcf(500, " (Likely Unrealistic growth)")]
    assert rt.check_historical_cagr_outlier(findings, "A") == []


def test_outlier_check_needs_at_least_two_scenarios():
    assert rt.check_historical_cagr_outlier([_dcf(100)], "A") == []
    assert rt.check_historical_cagr_outlier([], "A") == []


def test_outlier_check_ignores_non_dcf_findings():
    findings = [_f("Enterprise Value = $999,999"), _dcf(100), _dcf(110)]
    assert rt.check_historical_cagr_outlier(findings, "A") == []


def test_two_scenarios_can_never_be_flagged():
    # Current behavior: with 2 values the "median" is the larger one, so nothing exceeds 2x it
    assert rt.check_historical_cagr_outlier([_dcf(100), _dcf(1000)], "A") == []


# ---- check_revenue_consistency ----

FIN_CAGR = "NVDA revenue grew at a 45.8% CAGR from FY2020 to FY2024"
VAL_NONE = "DCF scenario historical_cagr: growth rate used: None, Enterprise Value = $1,000"


def test_hidden_growth_rate_in_valuation_claim_is_flagged():
    out = rt.check_revenue_consistency([_f(FIN_CAGR)], [_f(VAL_NONE)])
    assert len(out) == 1
    assert out[0].verification_result == "FLAGGED"
    assert out[0].source_agent == "ValuationAgent"
    assert "45.8%" in out[0].evidence
    assert "45.8" in out[0].correction


def test_displayed_growth_rate_is_not_flagged():
    val = _f("DCF scenario historical_cagr: growth rate used: 45.8, Enterprise Value = $1,000")
    assert rt.check_revenue_consistency([_f(FIN_CAGR)], [val]) == []


def test_no_financial_cagr_means_no_results():
    assert rt.check_revenue_consistency([_f("Margins improved")], [_f(VAL_NONE)]) == []


def test_non_historical_cagr_valuation_findings_are_ignored():
    val = _f("DCF scenario bull: growth rate used: None")
    assert rt.check_revenue_consistency([_f(FIN_CAGR)], [val]) == []


def test_first_financial_cagr_wins():
    fin = [_f(FIN_CAGR), _f("NVDA revenue grew at a 10.0% CAGR from FY2015 to FY2019")]
    out = rt.check_revenue_consistency(fin, [_f(VAL_NONE)])
    assert "45.8%" in out[0].evidence


# ---- run_verification, unmocked end to end ----

def test_run_verification_end_to_end_with_real_checks():
    from core.schemas import AgentResult

    bad = _f("Revenue was $4,805 million", "no figures here", confidence=0.05)
    result = AgentResult(agent_name="FinancialAgent", task="t", findings=[bad])

    out = rt.run_verification({"FinancialAgent": result})

    assert [v.verification_result for v in out] == ["FLAGGED", "UNSUPPORTED"]
    assert all(v.source_agent == "FinancialAgent" for v in out)
