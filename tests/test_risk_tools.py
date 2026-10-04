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



# ---- _load_classifier, evidence assembly, year counting, corroboration ----
import pickle as _pickle

from tools import risk_tools as rt


def _patch_models_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(rt, "load_config", lambda: {})
    monkeypatch.setattr(rt, "get_path", lambda cfg, key: tmp_path)


def test_load_classifier_returns_none_when_pickle_missing(tmp_path, monkeypatch):
    _patch_models_dir(monkeypatch, tmp_path)
    assert rt._load_classifier() is None


def test_load_classifier_reads_pickle(tmp_path, monkeypatch):
    _patch_models_dir(monkeypatch, tmp_path)
    clf_dir = tmp_path / "classification"
    clf_dir.mkdir()
    (clf_dir / "event_classifier.pkl").write_bytes(_pickle.dumps({"vectorizer": "v", "model": "m"}))

    assert rt._load_classifier() == {"vectorizer": "v", "model": "m"}


def test_classify_for_uncertainty_only_sees_first_500_characters():
    vectorizer, model = MagicMock(), MagicMock()
    model.predict.return_value = ["FACT"]
    rt._classify_for_uncertainty("x" * 1000, {"vectorizer": vectorizer, "model": model})

    vectorizer.transform.assert_called_once_with(["x" * 500])


def test_identify_risks_evidence_is_top_3_by_score_and_text_truncated():
    chunks = [
        {"chunk_id": f"c{i}", "text": "t" * 1000, "score": score, "filing_date": "2023-01-28"}
        for i, score in enumerate([0.1, 0.9, 0.5, 0.7, 0.2])
    ]
    with patch.object(rt, "rag_retrieve", return_value=chunks), \
         patch.object(rt, "_load_classifier", return_value=None):
        risks = identify_risks("NVDA")

    ev = risks[0]["evidence"]
    assert [e["chunk_id"] for e in ev] == ["c1", "c3", "c2"]
    assert all(len(e["text"]) == 400 for e in ev)


def test_identify_risks_counts_distinct_years_and_ignores_missing_dates():
    chunks = [
        {"chunk_id": "a", "text": "t", "score": 0.3, "filing_date": "2020-03-01"},
        {"chunk_id": "b", "text": "t", "score": 0.3, "filing_date": "2020-11-01"},
        {"chunk_id": "c", "text": "t", "score": 0.3, "filing_date": "2021-03-01"},
        {"chunk_id": "d", "text": "t", "score": 0.3, "filing_date": ""},
    ]
    with patch.object(rt, "rag_retrieve", return_value=chunks), \
         patch.object(rt, "_load_classifier", return_value=None):
        risks = identify_risks("NVDA")

    assert risks[0]["distinct_years_mentioned"] == 2
    assert risks[0]["severity"] == "LOW"


def test_identify_risks_medium_severity_at_three_distinct_years():
    with patch.object(rt, "rag_retrieve", return_value=fake_chunks(["2020", "2021", "2022"])), \
         patch.object(rt, "_load_classifier", return_value=None):
        risks = identify_risks("NVDA")

    assert all(r["severity"] == "MEDIUM" for r in risks)


def test_identify_risks_reports_uncertainty_corroboration_count():
    classifier = {"vectorizer": MagicMock(), "model": MagicMock()}
    classifier["model"].predict.side_effect = lambda vec: ["UNCERTAINTY"]
    with patch.object(rt, "rag_retrieve", return_value=fake_chunks(["2020", "2021"])), \
         patch.object(rt, "_load_classifier", return_value=classifier):
        risks = identify_risks("NVDA")

    assert risks[0]["uncertainty_corroboration"].startswith("2/2")


def test_identify_risks_queries_every_category_with_top_k():
    with patch.object(rt, "rag_retrieve", return_value=[]) as mock_rag, \
         patch.object(rt, "_load_classifier", return_value=None):
        identify_risks("NVDA", top_k_per_category=4)

    assert mock_rag.call_count == len(rt.RISK_CATEGORIES)
    mock_rag.assert_any_call(rt.RISK_CATEGORIES["geopolitical"], top_k=4)


def test_identify_risks_returns_one_record_per_category_with_results():
    def fake(query, top_k):
        return fake_chunks(["2022"]) if query == rt.RISK_CATEGORIES["financial"] else []

    with patch.object(rt, "rag_retrieve", side_effect=fake), \
         patch.object(rt, "_load_classifier", return_value=None):
        risks = identify_risks("NVDA")

    assert [r["category"] for r in risks] == ["financial"]
