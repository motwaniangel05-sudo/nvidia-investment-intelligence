"""Tests for loading CSV data into the knowledge base."""

import math

import pandas as pd
import pytest

from core.memory import (
    _accession_from_source_file,
    _none_if_nan,
    load_chunks,
    load_documents,
    load_metrics,
    load_prices,
)
from core.schemas import get_connection, initialize_schema


def make_config(tmp_path):
    financial_dir = tmp_path / "financial"
    processed_dir = tmp_path / "processed"
    market_dir = tmp_path / "market"
    for d in (financial_dir, processed_dir, market_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "company": {"ticker": "NVDA"},
        "competitors": [{"ticker": "AMD", "name": "AMD", "reason": "test"}],
        "paths": {
            "data_financial": str(financial_dir),
            "data_processed": str(processed_dir),
            "data_market": str(market_dir),
        },
    }, financial_dir, processed_dir, market_dir


def patch_get_path(monkeypatch):
    def fake_get_path(cfg, key):
        from pathlib import Path
        return Path(cfg["paths"][key])
    monkeypatch.setattr("core.memory.get_path", fake_get_path)


def test_none_if_nan_converts_nan_to_none():
    assert _none_if_nan(float("nan")) is None
    assert _none_if_nan(5) == 5
    assert _none_if_nan("text") == "text"


def test_accession_from_source_file_parses_correctly():
    name = "NVDA_10-K_0001045810-16-000205_nvda-2016x10k.htm"
    assert _accession_from_source_file(name) == "0001045810-16-000205"


def test_accession_from_source_file_handles_malformed_name():
    assert _accession_from_source_file("weird_name.htm") == ""


def test_load_documents_inserts_rows(tmp_path, monkeypatch):
    config, financial_dir, _, _ = make_config(tmp_path)
    patch_get_path(monkeypatch)

    pd.DataFrame([
        {"form": "10-K", "filingDate": "2016-03-17", "reportDate": "2016-01-31",
         "accessionNumber": "0001-16-000205", "primaryDocument": "a.htm",
         "url": "http://example.com/a.htm"},
    ]).to_csv(financial_dir / "filings_index_NVDA.csv", index=False)

    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    conn = get_connection(db_path)

    count = load_documents(conn, config)
    assert count == 1

    row = conn.execute("SELECT * FROM documents").fetchone()
    assert row["document_id"] == "NVDA_10-K_0001-16-000205"
    assert row["company_ticker"] == "NVDA"
    conn.close()


def test_load_documents_missing_file_returns_zero(tmp_path, monkeypatch):
    config, _, _, _ = make_config(tmp_path)
    patch_get_path(monkeypatch)

    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    conn = get_connection(db_path)

    count = load_documents(conn, config)
    assert count == 0
    conn.close()


def test_load_chunks_links_to_correct_document(tmp_path, monkeypatch):
    config, financial_dir, processed_dir, _ = make_config(tmp_path)
    patch_get_path(monkeypatch)

    # Documents must exist first (chunks has a foreign key to documents)
    pd.DataFrame([
        {"form": "10-K", "filingDate": "2016-03-17", "reportDate": "2016-01-31",
         "accessionNumber": "0001-16-000205", "primaryDocument": "a.htm",
         "url": "http://example.com/a.htm"},
    ]).to_csv(financial_dir / "filings_index_NVDA.csv", index=False)

    pd.DataFrame([
        {"chunk_id": "NVDA_10-K_2016-03-17_0000", "company_ticker": "NVDA",
         "form": "10-K", "filing_date": "2016-03-17",
         "source_file": "NVDA_10-K_0001-16-000205_a.htm",
         "chunk_index": 0, "word_count": 500, "text": "Sample chunk text."},
    ]).to_csv(processed_dir / "chunks_NVDA.csv", index=False)

    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    conn = get_connection(db_path)

    load_documents(conn, config)
    count = load_chunks(conn, config)
    assert count == 1

    row = conn.execute("SELECT * FROM chunks").fetchone()
    assert row["document_id"] == "NVDA_10-K_0001-16-000205"
    conn.close()


def test_load_metrics_converts_nan_to_null(tmp_path, monkeypatch):
    config, financial_dir, _, _ = make_config(tmp_path)
    patch_get_path(monkeypatch)

    pd.DataFrame([
        {"concept": "Revenues", "unit": "USD", "value": 5000000000,
         "start": None, "end": "2016-01-31", "form": "10-K", "fy": 2016,
         "fp": "FY", "filed": "2016-03-17", "accn": "0001-16-000205"},
    ]).to_csv(financial_dir / "xbrl_facts_NVDA.csv", index=False)

    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    conn = get_connection(db_path)

    count = load_metrics(conn, config)
    assert count == 1

    row = conn.execute("SELECT * FROM metrics").fetchone()
    assert row["period_start"] is None  # not the string "nan"
    assert row["value"] == 5000000000
    conn.close()


def test_load_metrics_is_idempotent_per_ticker(tmp_path, monkeypatch):
    config, financial_dir, _, _ = make_config(tmp_path)
    patch_get_path(monkeypatch)

    pd.DataFrame([
        {"concept": "Revenues", "unit": "USD", "value": 5000000000,
         "start": None, "end": "2016-01-31", "form": "10-K", "fy": 2016,
         "fp": "FY", "filed": "2016-03-17", "accn": "0001-16-000205"},
    ]).to_csv(financial_dir / "xbrl_facts_NVDA.csv", index=False)

    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    conn = get_connection(db_path)

    load_metrics(conn, config)
    load_metrics(conn, config)  # run twice

    count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    assert count == 1  # not 2 -- old rows cleared before reinsert
    conn.close()


def test_load_prices_tags_target_and_competitor_roles(tmp_path, monkeypatch):
    config, _, _, market_dir = make_config(tmp_path)
    patch_get_path(monkeypatch)

    pd.DataFrame([
        {"date": "2016-01-04", "open": 1, "high": 1.2, "low": 0.9,
         "close": 1.1, "adj_close": 1.05, "volume": 1000},
    ]).to_csv(market_dir / "prices_NVDA.csv", index=False)

    pd.DataFrame([
        {"date": "2016-01-04", "open": 2, "high": 2.2, "low": 1.9,
         "close": 2.1, "adj_close": 2.05, "volume": 2000},
    ]).to_csv(market_dir / "prices_AMD.csv", index=False)

    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    conn = get_connection(db_path)

    count = load_prices(conn, config)
    assert count == 2

    nvda_row = conn.execute(
        "SELECT * FROM prices WHERE company_ticker = 'NVDA'"
    ).fetchone()
    amd_row = conn.execute(
        "SELECT * FROM prices WHERE company_ticker = 'AMD'"
    ).fetchone()
    assert nvda_row["role"] == "target"
    assert amd_row["role"] == "competitor"
    conn.close()


def test_load_prices_missing_ticker_file_skips_gracefully(tmp_path, monkeypatch):
    config, _, _, market_dir = make_config(tmp_path)
    patch_get_path(monkeypatch)

    # Only NVDA file exists; AMD is configured but has no CSV
    pd.DataFrame([
        {"date": "2016-01-04", "open": 1, "high": 1.2, "low": 0.9,
         "close": 1.1, "adj_close": 1.05, "volume": 1000},
    ]).to_csv(market_dir / "prices_NVDA.csv", index=False)

    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    conn = get_connection(db_path)

    count = load_prices(conn, config)
    assert count == 1  # only NVDA loaded, AMD skipped without crashing
    conn.close()
