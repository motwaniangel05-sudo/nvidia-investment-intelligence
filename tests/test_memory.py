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



# ---- missing files, competitor CIK handling, filtering, build_knowledge_base ----
from unittest.mock import MagicMock, patch

from core import memory as mem


def _metrics_row(concept="Revenues", value=100):
    return {"concept": concept, "unit": "USD", "value": value,
            "start": None, "end": "2016-01-31", "form": "10-K", "fy": 2016,
            "fp": "FY", "filed": "2016-03-17", "accn": "0001-16-000205"}


def _fresh_conn(tmp_path):
    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    return get_connection(db_path)


def test_load_chunks_missing_file_returns_zero(tmp_path, monkeypatch):
    config, _, _, _ = make_config(tmp_path)
    patch_get_path(monkeypatch)
    conn = _fresh_conn(tmp_path)

    assert load_chunks(conn, config) == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    conn.close()


def test_load_metrics_missing_file_returns_zero(tmp_path, monkeypatch):
    config, _, _, _ = make_config(tmp_path)
    patch_get_path(monkeypatch)
    conn = _fresh_conn(tmp_path)

    assert load_metrics(conn, config) == 0
    conn.close()


def test_load_metrics_loads_competitor_with_cik_and_skips_one_without(tmp_path, monkeypatch):
    config, financial_dir, _, _ = make_config(tmp_path)
    config["competitors"] = [
        {"ticker": "AMD", "cik": "0000002488"},
        {"ticker": "INTC"},  # no CIK configured: must be skipped even if a file exists
    ]
    patch_get_path(monkeypatch)

    for t in ("NVDA", "AMD", "INTC"):
        pd.DataFrame([_metrics_row()]).to_csv(financial_dir / f"xbrl_facts_{t}.csv", index=False)

    conn = _fresh_conn(tmp_path)
    assert load_metrics(conn, config) == 2

    tickers = {r[0] for r in conn.execute("SELECT DISTINCT company_ticker FROM metrics")}
    assert tickers == {"NVDA", "AMD"}
    conn.close()


def test_load_metrics_reload_only_clears_its_own_ticker(tmp_path, monkeypatch):
    config, financial_dir, _, _ = make_config(tmp_path)
    config["competitors"] = [{"ticker": "AMD", "cik": "0000002488"}]
    patch_get_path(monkeypatch)

    pd.DataFrame([_metrics_row()]).to_csv(financial_dir / "xbrl_facts_NVDA.csv", index=False)
    pd.DataFrame([_metrics_row(), _metrics_row("NetIncomeLoss", 5)]).to_csv(
        financial_dir / "xbrl_facts_AMD.csv", index=False)

    conn = _fresh_conn(tmp_path)
    load_metrics(conn, config)
    load_metrics(conn, config)

    counts = dict(conn.execute("SELECT company_ticker, COUNT(*) FROM metrics GROUP BY company_ticker").fetchall())
    assert counts == {"NVDA": 1, "AMD": 2}
    conn.close()


def test_load_documents_keeps_only_10k_and_10q(tmp_path, monkeypatch):
    config, financial_dir, _, _ = make_config(tmp_path)
    patch_get_path(monkeypatch)

    def row(form, accn):
        return {"form": form, "filingDate": "2016-03-17", "reportDate": "2016-01-31",
                "accessionNumber": accn, "primaryDocument": "a.htm",
                "url": "http://example.com/a.htm"}

    pd.DataFrame([row("10-K", "A1"), row("10-Q", "A2"), row("8-K", "A3")]).to_csv(
        financial_dir / "filings_index_NVDA.csv", index=False)

    conn = _fresh_conn(tmp_path)
    assert load_documents(conn, config) == 2

    forms = {r[0] for r in conn.execute("SELECT form FROM documents")}
    assert forms == {"10-K", "10-Q"}
    conn.close()


def test_load_prices_is_idempotent(tmp_path, monkeypatch):
    config, _, _, market_dir = make_config(tmp_path)
    patch_get_path(monkeypatch)

    pd.DataFrame([
        {"date": "2016-01-04", "open": 1, "high": 1.2, "low": 0.9,
         "close": 1.1, "adj_close": 1.05, "volume": 1000},
    ]).to_csv(market_dir / "prices_NVDA.csv", index=False)

    conn = _fresh_conn(tmp_path)
    load_prices(conn, config)
    load_prices(conn, config)

    assert conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 1
    conn.close()


def test_build_knowledge_base_returns_counts_and_closes_connection():
    conn = MagicMock()
    with patch.object(mem, "load_config", return_value={"cfg": 1}), \
         patch.object(mem, "initialize_schema") as mock_init, \
         patch.object(mem, "get_connection", return_value=conn) as mock_get, \
         patch.object(mem, "load_documents", return_value=1) as ld, \
         patch.object(mem, "load_chunks", return_value=2), \
         patch.object(mem, "load_metrics", return_value=3), \
         patch.object(mem, "load_prices", return_value=4):
        out = mem.build_knowledge_base(db_path="some.db")

    assert out == {"documents": 1, "chunks": 2, "metrics": 3, "prices": 4}
    mock_init.assert_called_once_with("some.db")
    mock_get.assert_called_once_with("some.db")
    ld.assert_called_once_with(conn, {"cfg": 1})
    conn.close.assert_called_once()


def test_build_knowledge_base_closes_connection_when_a_loader_raises():
    conn = MagicMock()
    with patch.object(mem, "load_config", return_value={}), \
         patch.object(mem, "initialize_schema"), \
         patch.object(mem, "get_connection", return_value=conn), \
         patch.object(mem, "load_documents", return_value=0), \
         patch.object(mem, "load_chunks", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            mem.build_knowledge_base()

    conn.close.assert_called_once()
