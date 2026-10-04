"""Tests for RiskAgent: Finding construction from risk_tools output."""

from unittest.mock import patch

from agents.risk_agent import RiskAgent


def fake_risks():
    return [
        {
            "category": "competitive",
            "severity": "HIGH",
            "severity_methodology": "Methodology text here.",
            "distinct_years_mentioned": 8,
            "uncertainty_corroboration": "4/8 chunks were UNCERTAINTY.",
            "evidence": [
                {"chunk_id": "c1", "text": "Competitive risk evidence text.",
                 "score": 0.3, "filing_date": "2024-01-28"},
            ],
        },
        {
            "category": "supply_chain",
            "severity": "MEDIUM",
            "severity_methodology": "Methodology text here.",
            "distinct_years_mentioned": 4,
            "uncertainty_corroboration": "8/8 chunks were UNCERTAINTY.",
            "evidence": [
                {"chunk_id": "c2", "text": "Supply chain evidence text.",
                 "score": 0.25, "filing_date": "2023-01-29"},
            ],
        },
    ]


def test_risk_agent_produces_one_finding_per_category():
    agent = RiskAgent("NVDA")
    with patch("agents.risk_agent.identify_risks", return_value=fake_risks()):
        result = agent.run("Assess risks")

    assert len(result.findings) == 2
    assert result.status == "success"


def test_risk_agent_severity_distribution_in_metrics():
    agent = RiskAgent("NVDA")
    with patch("agents.risk_agent.identify_risks", return_value=fake_risks()):
        result = agent.run("Assess risks")

    assert result.metrics["severity_distribution"]["HIGH"] == 1
    assert result.metrics["severity_distribution"]["MEDIUM"] == 1
    assert result.metrics["severity_distribution"]["LOW"] == 0


def test_risk_agent_empty_risks_returns_partial():
    agent = RiskAgent("NVDA")
    with patch("agents.risk_agent.identify_risks", return_value=[]):
        result = agent.run("Assess risks")

    assert result.status == "partial"
    assert len(result.warnings) >= 1


def test_risk_agent_always_includes_methodology_disclaimer():
    agent = RiskAgent("NVDA")
    with patch("agents.risk_agent.identify_risks", return_value=fake_risks()):
        result = agent.run("Assess risks")

    assert any("frequency-based proxy" in w for w in result.warnings)


def test_risk_agent_finding_claim_includes_category_and_severity():
    agent = RiskAgent("NVDA")
    with patch("agents.risk_agent.identify_risks", return_value=fake_risks()):
        result = agent.run("Assess risks")

    claims = [f.claim for f in result.findings]
    assert any("Competitive" in c and "HIGH" in c for c in claims)
    assert any("Supply Chain" in c and "MEDIUM" in c for c in claims)


def test_risk_agent_handles_risk_with_no_evidence():
    risks_no_evidence = [{
        "category": "financial",
        "severity": "LOW",
        "severity_methodology": "Methodology text.",
        "distinct_years_mentioned": 1,
        "uncertainty_corroboration": "0/0 chunks.",
        "evidence": [],
    }]
    agent = RiskAgent("NVDA")
    with patch("agents.risk_agent.identify_risks", return_value=risks_no_evidence):
        result = agent.run("Assess risks")

    assert len(result.findings) == 1
    assert result.findings[0].source_filing_date == "n/a"



# ---- retrieve, finding metadata, evidence selection, severity counts ----

def _run_risk(risks):
    with patch("agents.risk_agent.identify_risks", return_value=risks):
        return RiskAgent("NVDA").run("Assess risks")


def _risk(category="financial", severity="LOW", evidence=None, years=1):
    return {
        "category": category,
        "severity": severity,
        "severity_methodology": "Methodology text.",
        "distinct_years_mentioned": years,
        "uncertainty_corroboration": "0/0 chunks.",
        "evidence": evidence if evidence is not None else [],
    }


def test_retrieve_is_unused_and_returns_empty_list():
    assert RiskAgent("NVDA").retrieve("anything") == []


def test_finding_metadata_and_text_for_first_risk():
    result = _run_risk(fake_risks())
    f = result.findings[0]

    assert f.claim == "Competitive risk: severity=HIGH (8 distinct fiscal years mentioned)."
    assert f.source_chunk_id == "c1"
    assert f.source_form == "10-K/10-Q"
    assert f.source_filing_date == "2024-01-28"
    assert f.confidence == 1.0
    assert "Methodology text here." in f.evidence_text
    assert "4/8 chunks were UNCERTAINTY." in f.evidence_text
    assert "Example evidence: Competitive risk evidence text." in f.evidence_text


def test_underscored_category_is_title_cased_in_claim():
    result = _run_risk([_risk(category="customer_concentration")])
    assert result.findings[0].claim.startswith("Customer Concentration risk:")


def test_only_first_evidence_item_is_used():
    evidence = [
        {"chunk_id": "top", "text": "best text", "score": 0.9, "filing_date": "2024-01-28"},
        {"chunk_id": "second", "text": "other text", "score": 0.5, "filing_date": "2023-01-29"},
    ]
    f = _run_risk([_risk(evidence=evidence)]).findings[0]

    assert f.source_chunk_id == "top"
    assert "best text" in f.evidence_text
    assert "other text" not in f.evidence_text


def test_risk_without_evidence_gets_fallback_chunk_id_and_na_text():
    f = _run_risk([_risk(category="financial", evidence=[])]).findings[0]

    assert f.source_chunk_id == "NVDA_financial_no_evidence"
    assert f.source_filing_date == "n/a"
    assert "Example evidence: N/A" in f.evidence_text


def test_severity_distribution_counts_all_three_levels():
    risks = [
        _risk("competitive", "HIGH"), _risk("regulatory", "HIGH"),
        _risk("technology", "MEDIUM"),
        _risk("financial", "LOW"), _risk("execution", "LOW"), _risk("geopolitical", "LOW"),
    ]
    result = _run_risk(risks)

    assert result.metrics["severity_distribution"] == {"HIGH": 2, "MEDIUM": 1, "LOW": 3}
    assert result.metrics["risk_categories_analyzed"] == 6


def test_empty_risks_result_has_no_findings_and_specific_warning():
    result = _run_risk([])

    assert result.findings == []
    assert any("No risk categories produced usable evidence" in w for w in result.warnings)
