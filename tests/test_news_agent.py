"""Tests for NewsAgent: classification integration, reliability flagging."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agents.news_agent import NewsAgent


def fake_chunks_df():
    return pd.DataFrame([
        {
            "chunk_id": "NVDA_8-K_2024-01-01_0000",
            "filing_date": "2024-01-01",
            "text": "Revenue was 500 million dollars in the quarter. We believe our strategy will succeed.",
        },
    ])


def test_news_agent_flags_claim_predictions_as_low_reliability(tmp_path, monkeypatch):
    agent = NewsAgent("NVDA")

    fake_config = {"paths": {"data_processed": str(tmp_path)}}
    monkeypatch.setattr("agents.news_agent.load_config", lambda: fake_config)
    monkeypatch.setattr(
        "agents.news_agent.get_path",
        lambda cfg, key: tmp_path,
    )

    chunks_path = tmp_path / "chunks_NVDA_8K.csv"
    fake_chunks_df().to_csv(chunks_path, index=False)

    # Mock classify to always return CLAIM
    agent._classify_sentence = MagicMock(return_value={"label": "CLAIM", "confidence": 0.9})

    result = agent.run("test")

    assert len(result.findings) > 0
    assert all("LOW RELIABILITY" in f.claim for f in result.findings)


def test_news_agent_does_not_flag_fact_predictions(tmp_path, monkeypatch):
    agent = NewsAgent("NVDA")

    fake_config = {"paths": {"data_processed": str(tmp_path)}}
    monkeypatch.setattr("agents.news_agent.load_config", lambda: fake_config)
    monkeypatch.setattr("agents.news_agent.get_path", lambda cfg, key: tmp_path)

    chunks_path = tmp_path / "chunks_NVDA_8K.csv"
    fake_chunks_df().to_csv(chunks_path, index=False)

    agent._classify_sentence = MagicMock(return_value={"label": "FACT", "confidence": 0.9})

    result = agent.run("test")

    assert len(result.findings) > 0
    assert all("LOW RELIABILITY" not in f.claim for f in result.findings)


def test_news_agent_missing_chunks_file_returns_failed(tmp_path, monkeypatch):
    agent = NewsAgent("NVDA")

    fake_config = {"paths": {"data_processed": str(tmp_path)}}
    monkeypatch.setattr("agents.news_agent.load_config", lambda: fake_config)
    monkeypatch.setattr("agents.news_agent.get_path", lambda cfg, key: tmp_path)
    # Don't create the chunks file

    result = agent.run("test")

    assert result.status == "failed"
    assert len(result.warnings) >= 1


def test_news_agent_always_includes_unreliability_warning(tmp_path, monkeypatch):
    agent = NewsAgent("NVDA")

    fake_config = {"paths": {"data_processed": str(tmp_path)}}
    monkeypatch.setattr("agents.news_agent.load_config", lambda: fake_config)
    monkeypatch.setattr("agents.news_agent.get_path", lambda cfg, key: tmp_path)

    chunks_path = tmp_path / "chunks_NVDA_8K.csv"
    fake_chunks_df().to_csv(chunks_path, index=False)

    agent._classify_sentence = MagicMock(return_value={"label": "FACT", "confidence": 0.9})

    result = agent.run("test")

    assert any("CLAIM classification is unreliable" in w for w in result.warnings)


def test_news_agent_tallies_label_counts(tmp_path, monkeypatch):
    agent = NewsAgent("NVDA")

    fake_config = {"paths": {"data_processed": str(tmp_path)}}
    monkeypatch.setattr("agents.news_agent.load_config", lambda: fake_config)
    monkeypatch.setattr("agents.news_agent.get_path", lambda cfg, key: tmp_path)

    chunks_path = tmp_path / "chunks_NVDA_8K.csv"
    fake_chunks_df().to_csv(chunks_path, index=False)

    agent._classify_sentence = MagicMock(return_value={"label": "FACT", "confidence": 0.9})

    result = agent.run("test")

    assert result.metrics["label_counts"].get("FACT", 0) >= 1
    assert result.metrics["sentences_classified"] == len(result.findings)


# ---- retrieve, classifier loading, sentence classification, filtering ----
import pickle as _pickle


def _news_setup(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.news_agent.load_config", lambda: {"paths": {}})
    monkeypatch.setattr("agents.news_agent.get_path", lambda cfg, key: tmp_path)
    fake_chunks_df().to_csv(tmp_path / "chunks_NVDA_8K.csv", index=False)


def test_retrieve_is_unused_and_returns_empty_list():
    assert NewsAgent("NVDA").retrieve("anything") == []


def test_load_classifier_raises_helpful_error_when_pickle_missing(tmp_path, monkeypatch):
    _news_setup(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="train_classifier"):
        NewsAgent("NVDA")._load_classifier()


def test_load_classifier_reads_pickle_and_caches_result(tmp_path, monkeypatch):
    _news_setup(tmp_path, monkeypatch)
    clf_dir = tmp_path / "classification"
    clf_dir.mkdir()
    pkl = clf_dir / "event_classifier.pkl"
    pkl.write_bytes(_pickle.dumps({"vectorizer": "v", "model": "m"}))

    agent = NewsAgent("NVDA")
    first = agent._load_classifier()
    assert first == {"vectorizer": "v", "model": "m"}

    pkl.unlink()  # a second call must not touch the disk
    assert agent._load_classifier() is first


def test_classify_sentence_returns_label_and_max_probability():
    agent = NewsAgent("NVDA")
    vectorizer, model = MagicMock(), MagicMock()
    model.predict.return_value = ["UNCERTAINTY"]
    model.predict_proba.return_value = [[0.1, 0.7, 0.2]]
    agent._classifier = {"vectorizer": vectorizer, "model": model}

    out = agent._classify_sentence("Results may vary.")

    vectorizer.transform.assert_called_once_with(["Results may vary."])
    assert out["label"] == "UNCERTAINTY"
    assert out["confidence"] == pytest.approx(0.7)
    assert isinstance(out["confidence"], float)


def test_analyze_skips_sentences_that_are_not_labelable(tmp_path, monkeypatch):
    _news_setup(tmp_path, monkeypatch)
    monkeypatch.setattr("agents.news_agent.split_into_sentences", lambda text: ["keep me", "drop me"])
    monkeypatch.setattr("agents.news_agent.is_labelable_sentence", lambda s: s == "keep me")

    agent = NewsAgent("NVDA")
    agent._classify_sentence = MagicMock(return_value={"label": "FACT", "confidence": 0.9})
    result = agent.run("test")

    agent._classify_sentence.assert_called_once_with("keep me")
    assert [f.evidence_text for f in result.findings] == ["keep me"]


def test_analyze_returns_partial_when_nothing_is_labelable(tmp_path, monkeypatch):
    _news_setup(tmp_path, monkeypatch)
    monkeypatch.setattr("agents.news_agent.is_labelable_sentence", lambda s: False)

    agent = NewsAgent("NVDA")
    agent._classify_sentence = MagicMock()
    result = agent.run("test")

    agent._classify_sentence.assert_not_called()
    assert result.status == "partial"
    assert result.findings == []
    assert result.metrics["sentences_classified"] == 0
    assert result.metrics["label_counts"] == {}
    assert any("CLAIM classification is unreliable" in w for w in result.warnings)


def test_analyze_finding_carries_chunk_metadata_and_rounded_confidence(tmp_path, monkeypatch):
    _news_setup(tmp_path, monkeypatch)
    monkeypatch.setattr("agents.news_agent.split_into_sentences", lambda text: ["one sentence"])
    monkeypatch.setattr("agents.news_agent.is_labelable_sentence", lambda s: True)

    agent = NewsAgent("NVDA")
    agent._classify_sentence = MagicMock(return_value={"label": "FACT", "confidence": 0.123456})
    result = agent.run("test")

    f = result.findings[0]
    assert f.claim == "Classified as FACT"
    assert f.evidence_text == "one sentence"
    assert f.source_chunk_id == "NVDA_8-K_2024-01-01_0000"
    assert f.source_form == "8-K"
    assert f.source_filing_date == "2024-01-01"
    assert f.confidence == 0.1235
    assert result.status == "success"


def test_analyze_counts_each_label_separately(tmp_path, monkeypatch):
    _news_setup(tmp_path, monkeypatch)
    monkeypatch.setattr("agents.news_agent.split_into_sentences", lambda text: ["a", "b", "c"])
    monkeypatch.setattr("agents.news_agent.is_labelable_sentence", lambda s: True)

    agent = NewsAgent("NVDA")
    agent._classify_sentence = MagicMock(side_effect=[
        {"label": "FACT", "confidence": 0.9},
        {"label": "CLAIM", "confidence": 0.6},
        {"label": "FACT", "confidence": 0.8},
    ])
    result = agent.run("test")

    assert result.metrics["label_counts"] == {"FACT": 2, "CLAIM": 1}
    assert [("LOW RELIABILITY" in f.claim) for f in result.findings] == [False, True, False]
