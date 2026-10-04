"""Tests for the knowledge base schema: table creation, constraints, indexes."""

import sqlite3

import pytest

from core.schemas import get_connection, initialize_schema, table_counts


def test_all_tables_are_created(tmp_path):
    db_path = tmp_path / "test.db"
    initialize_schema(db_path)

    conn = get_connection(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cursor.fetchall()}
    conn.close()

    assert {"documents", "chunks", "metrics", "prices"}.issubset(tables)


def test_foreign_keys_are_enforced(tmp_path):
    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    conn = get_connection(db_path)

    # Try inserting a chunk whose document_id does not exist in documents
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO chunks (chunk_id, document_id, chunk_index, word_count, text) "
            "VALUES (?, ?, ?, ?, ?)",
            ("fake_chunk", "nonexistent_document", 0, 10, "some text"),
        )
        conn.commit()
    conn.close()


def test_valid_chunk_insert_succeeds_with_parent_document(tmp_path):
    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    conn = get_connection(db_path)

    conn.execute(
        "INSERT INTO documents (document_id, company_ticker, form, filing_date, "
        "report_date, accession_number, source_url, local_file) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("NVDA_10-K_test123", "NVDA", "10-K", "2016-03-17",
         "2016-01-31", "test123", "http://example.com", "test.htm"),
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, document_id, chunk_index, word_count, text) "
        "VALUES (?, ?, ?, ?, ?)",
        ("chunk_0", "NVDA_10-K_test123", 0, 100, "sample chunk text"),
    )
    conn.commit()

    cursor = conn.execute("SELECT COUNT(*) FROM chunks")
    assert cursor.fetchone()[0] == 1
    conn.close()


def test_indexes_exist(tmp_path):
    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    conn = get_connection(db_path)

    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = {row["name"] for row in cursor.fetchall()}
    conn.close()

    expected = {
        "idx_chunks_document_id",
        "idx_metrics_ticker_concept",
        "idx_metrics_fy_form",
        "idx_prices_ticker_date",
    }
    assert expected.issubset(indexes)


def test_table_counts_reflects_inserted_rows(tmp_path):
    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    conn = get_connection(db_path)

    conn.execute(
        "INSERT INTO metrics (company_ticker, concept, unit, value, period_end) "
        "VALUES (?, ?, ?, ?, ?)",
        ("NVDA", "Revenues", "USD", 5000000000, "2016-01-31"),
    )
    conn.execute(
        "INSERT INTO prices (company_ticker, role, date, close) VALUES (?, ?, ?, ?)",
        ("NVDA", "target", "2016-01-04", 0.81),
    )
    conn.commit()
    conn.close()

    counts = table_counts(db_path)
    assert counts["metrics"] == 1
    assert counts["prices"] == 1
    assert counts["documents"] == 0
    assert counts["chunks"] == 0


def test_initialize_schema_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    initialize_schema(db_path)  # run twice, should not error

    conn = get_connection(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row["name"] for row in cursor.fetchall()]
    conn.close()

    # No duplicate table names
    assert len(tables) == len(set(tables))


def test_metrics_does_not_require_document_fk(tmp_path):
    """
    Confirms design decision: a metric can be inserted with an
    accession_number that doesn't correspond to any row in documents,
    since metrics are not foreign-keyed to documents (see Phase 4 notes).
    """
    db_path = tmp_path / "test.db"
    initialize_schema(db_path)
    conn = get_connection(db_path)

    # Should NOT raise, unlike the chunks FK test above
    conn.execute(
        "INSERT INTO metrics (company_ticker, concept, unit, value, "
        "period_end, accession_number) VALUES (?, ?, ?, ?, ?, ?)",
        ("NVDA", "Revenues", "USD", 5000000000, "2016-01-31", "no_such_accession"),
    )
    conn.commit()
    conn.close()



# ---- get_db_path ----
from unittest.mock import patch as _patch

from core import schemas as _schemas


def test_get_db_path_uses_the_config_it_is_given():
    cfg = {"paths": {"knowledge_db": "x.db"}}
    with _patch.object(_schemas, "get_path", return_value="db-path") as mock_get, \
         _patch.object(_schemas, "load_config") as mock_load:
        assert _schemas.get_db_path(cfg) == "db-path"

    mock_get.assert_called_once_with(cfg, "knowledge_db")
    mock_load.assert_not_called()


def test_get_db_path_loads_config_when_none_given():
    loaded = {"loaded": True}
    with _patch.object(_schemas, "get_path", return_value="db-path") as mock_get, \
         _patch.object(_schemas, "load_config", return_value=loaded) as mock_load:
        assert _schemas.get_db_path() == "db-path"

    mock_load.assert_called_once_with()
    mock_get.assert_called_once_with(loaded, "knowledge_db")
