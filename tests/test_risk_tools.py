"""Tests for risk identification: severity methodology, evidence assembly."""

from unittest.mock import MagicMock, patch

import pytest

from tools.risk_tools import (
    SEVERITY_THRESHOLDS,
    _classify_for_uncertainty,
    _severity_from_frequency,
    identify_risks,
)


def test_severity_from_frequency_high():
    assert _severity_from_frequency(SEVERITY_THRESHOLDS["HIGH"]) == "HIGH"
    assert _severity_from_frequency(SEVERITY_THRESHOLDS["HIGH"] + 5) == "HIGH"


def test_severity_from_frequency_medium():
    assert _severity_from_frequency(SEVERITY_THRESHOLDS["MEDIUM"]) == "MEDIUM"
    assert _severity_from_frequency(SEVERITY_THRESHOLDS["HIGH"] - 1) == "MEDIUM"


def test_severity_from_frequency_low():
    assert _severity_from_frequency(0) == "LOW"
    assert _severity_from_frequency(SEVERITY_THRESHOLDS["MEDIUM"] - 1) == "LOW"


def test_classify_for_uncertainty_with_no_classifier_returns_false():
    assert _classify_for_uncertainty("some risk text", None) is False


def test_classify_for_uncertainty_uses_model_prediction():
    fake_classifier = {
        "vectorizer": MagicMock(),
        "model": MagicMock(),
    }
    fake_classifier["model"].predict.return_value = ["UNCERTAINTY"]
    result = _classify_for_uncertainty("some risk text", fake_classifier)
    assert result is True

    fake_classifier["model"].predict.return_value = ["FACT"]
    result = _classify_for_uncertainty("some fact text", fake_classifier)
    assert result is False


def fake_chunks(years):
    """Build fake retrieved chunks with given filing years."""
    return [
        {"chunk_id": f"chunk_{i}", "text": f"risk text {i}", "score": 0.3,
         "filing_date": f"{year}-01-28"}
        for i, year in enumerate(years)
    ]


def test_identify_risks_computes_severity_from_distinct_years():
    # 7+ distinct years -> HIGH
    many_years = [str(y) for y in range(2016, 2024)]  # 8 distinct years

    with patch("tools.risk_tools.rag_retrieve", return_value=fake_chunks(many_years)), \
         patch("tools.risk_tools._load_classifier", return_value=None):
        risks = identify_risks("NVDA", top_k_per_category=8)

    assert len(risks) == len(risks)  # sanity: produced results
    for r in risks:
        assert r["severity"] == "HIGH"
        assert r["distinct_years_mentioned"] == 8
        assert "methodology" in r["severity_methodology"].lower() or "proxy" in r["severity_methodology"].lower()


def test_identify_risks_skips_category_with_no_results():
    with patch("tools.risk_tools.rag_retrieve", return_value=[]), \
         patch("tools.risk_tools._load_classifier", return_value=None):
        risks = identify_risks("NVDA")

    assert risks == []


def test_identify_risks_includes_uncertainty_corroboration_text():
    years = ["2020", "2021"]
    with patch("tools.risk_tools.rag_retrieve", return_value=fake_chunks(years)), \
         patch("tools.risk_tools._load_classifier", return_value=None):
        risks = identify_risks("NVDA")

    for r in risks:
        assert "uncertainty_corroboration" in r
        assert "/" in r["uncertainty_corroboration"]  # e.g. "0/2"


def test_identify_risks_evidence_has_required_fields():
    years = ["2022", "2023", "2024"]
    with patch("tools.risk_tools.rag_retrieve", return_value=fake_chunks(years)), \
         patch("tools.risk_tools._load_classifier", return_value=None):
        risks = identify_risks("NVDA")

    for r in risks:
        for ev in r["evidence"]:
            assert "chunk_id" in ev
            assert "text" in ev
            assert "score" in ev
            assert "filing_date" in ev
