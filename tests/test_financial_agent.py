"""Tests for FinancialAgent: finding construction from computed metrics."""

from unittest.mock import patch

from agents.financial_agent import FinancialAgent


def fake_summary(include_fcf=False):
    summary = {
        "company_ticker": "NVDA",
        "revenue": {"2016-01-31": 5010000000.0, "2017-01-29": 6910000000.0},
        "revenue_yoy_growth": {"2017-01-29": 37.9},
        "revenue_cagr": 17.5,
        "net_income": {"2016-01-31": 614000000.0, "2017-01-29": 1666000000.0},
        "net_income_yoy_growth": {"2017-01-29": 171.3},
        "margins": {
            "gross_margin": {"2016-01-31": 56.1},
            "operating_margin": {"2016-01-31": 15.0},
            "net_margin": {"2016-01-31": 12.3, "2017-01-29": 24.1},
        },
        "free_cash_flow": {"2016-01-31": 900000000.0} if include_fcf else {},
        "returns": {
            "roe": {"2016-01-31": 13.7, "2017-01-29": 28.9},
            "roa": {"2016-01-31": 8.3},
        },
    }
    return summary


def test_financial_agent_does_not_use_rag():
    agent = FinancialAgent("NVDA")
    assert agent.retrieve("anything") == []


def test_financial_agent_produces_cagr_finding():
    agent = FinancialAgent("NVDA")
    with patch("agents.financial_agent.build_financial_summary", return_value=fake_summary()):
        result = agent.run("Analyze NVIDIA's financial health")

    cagr_findings = [f for f in result.findings if "CAGR" in f.claim]
    assert len(cagr_findings) == 1
    assert "17.5%" in cagr_findings[0].claim
    assert cagr_findings[0].confidence == 1.0


def test_financial_agent_produces_yoy_growth_findings():
    agent = FinancialAgent("NVDA")
    with patch("agents.financial_agent.build_financial_summary", return_value=fake_summary()):
        result = agent.run("Analyze NVIDIA's financial health")

    yoy_findings = [f for f in result.findings if "YoY" in f.claim]
    assert len(yoy_findings) == 1
    assert "37.9%" in yoy_findings[0].claim


def test_financial_agent_produces_margin_findings():
    agent = FinancialAgent("NVDA")
    with patch("agents.financial_agent.build_financial_summary", return_value=fake_summary()):
        result = agent.run("Analyze NVIDIA's financial health")

    margin_findings = [f for f in result.findings if "net margin" in f.claim]
    assert len(margin_findings) == 2  # one per period in fake data


def test_financial_agent_produces_roe_findings():
    agent = FinancialAgent("NVDA")
    with patch("agents.financial_agent.build_financial_summary", return_value=fake_summary()):
        result = agent.run("Analyze NVIDIA's financial health")

    roe_findings = [f for f in result.findings if "ROE" in f.claim]
    assert len(roe_findings) == 2


def test_financial_agent_warns_when_fcf_missing():
    agent = FinancialAgent("NVDA")
    with patch("agents.financial_agent.build_financial_summary", return_value=fake_summary(include_fcf=False)):
        result = agent.run("Analyze NVIDIA's financial health")

    assert any("Free Cash Flow" in w for w in result.warnings)


def test_financial_agent_no_warning_when_fcf_present():
    agent = FinancialAgent("NVDA")
    with patch("agents.financial_agent.build_financial_summary", return_value=fake_summary(include_fcf=True)):
        result = agent.run("Analyze NVIDIA's financial health")

    assert not any("Free Cash Flow" in w for w in result.warnings)


def test_financial_agent_metrics_include_cagr_and_year_count():
    agent = FinancialAgent("NVDA")
    with patch("agents.financial_agent.build_financial_summary", return_value=fake_summary()):
        result = agent.run("Analyze NVIDIA's financial health")

    assert result.metrics["years_of_revenue_data"] == 2
    assert result.metrics["revenue_cagr_pct"] == 17.5


def test_financial_agent_status_success_when_findings_exist():
    agent = FinancialAgent("NVDA")
    with patch("agents.financial_agent.build_financial_summary", return_value=fake_summary()):
        result = agent.run("Analyze NVIDIA's financial health")
    assert result.status == "success"
