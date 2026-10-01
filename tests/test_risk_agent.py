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
