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
