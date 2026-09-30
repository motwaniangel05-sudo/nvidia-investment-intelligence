"""
SQLite schema definitions for the project's knowledge base.

Four tables this phase: documents, chunks, metrics, prices. See Phase 4
design notes for why metrics/prices are not foreign-keyed to documents.

Later phases (News/Event Agent, Risk Agent, agent orchestration) will add
further tables (events, claims, agent_results, verification_results) once
their actual shape is known from real requirements, rather than guessed
at now.
"""

import sqlite3
from pathlib import Path

from core.config_loader import get_path, load_config
from core.logger import get_logger

log = get_logger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id       TEXT PRIMARY KEY,
    company_ticker    TEXT NOT NULL,
    form              TEXT NOT NULL,
    filing_date       TEXT NOT NULL,
    report_date       TEXT,
    accession_number  TEXT NOT NULL,
    source_url        TEXT,
    local_file        TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL,
    word_count    INTEGER,
    text          TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    company_ticker    TEXT NOT NULL,
    concept           TEXT NOT NULL,
    unit              TEXT,
    value             REAL,
    period_start      TEXT,
    period_end        TEXT NOT NULL,
    form              TEXT,
    fiscal_year       INTEGER,
    fiscal_period     TEXT,
    filed_date        TEXT,
    accession_number  TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    price_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    company_ticker  TEXT NOT NULL,
    role            TEXT NOT NULL,
    date            TEXT NOT NULL,
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL,
    adj_close       REAL,
    volume          INTEGER
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_metrics_ticker_concept
    ON metrics(company_ticker, concept);

CREATE INDEX IF NOT EXISTS idx_metrics_fy_form
    ON metrics(fiscal_year, form);

CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
    ON prices(company_ticker, date);
"""


def get_db_path(config: dict = None) -> Path:
    """Return the configured path to the knowledge base SQLite file."""
    config = config or load_config()
    return get_path(config, "knowledge_db")


def get_connection(db_path: Path = None) -> sqlite3.Connection:
    """
    Open a connection to the knowledge base. Enables foreign key
    enforcement (off by default in SQLite) and row access by column name.
    """
    db_path = db_path or get_db_path()
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_schema(db_path: Path = None) -> None:
    """Create all tables and indexes if they don't already exist."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        log.info("Knowledge base schema initialized at %s", db_path or get_db_path())
    finally:
        conn.close()


def table_counts(db_path: Path = None) -> dict:
    """Return {table_name: row_count} for all four tables. Useful for
    quick sanity checks after loading data (Part 3)."""
    conn = get_connection(db_path)
    try:
        counts = {}
        for table in ("documents", "chunks", "metrics", "prices"):
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        return counts
    finally:
        conn.close()


if __name__ == "__main__":
    initialize_schema()
    print("Schema initialized. Current row counts:")
    for table, count in table_counts().items():
        print(f"  {table}: {count}")


# ============================================================
# Agent result structures (Phase 7+). These are NOT database
# tables -- agent outputs are transient/in-memory for now. If a
# later phase needs to persist them (e.g. for caching or audit),
# a dedicated table will be added then, based on real requirements.
# ============================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Finding:
    """One evidence-backed claim produced by an agent."""
    claim: str
    evidence_text: str
    source_chunk_id: str
    source_form: str
    source_filing_date: str
    confidence: float  # 0.0 to 1.0, based on retrieval score


@dataclass
class AgentResult:
    """Standardized output every agent returns (per master prompt Section 10)."""
    agent_name: str
    task: str
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    status: str = "success"  # "success" | "partial" | "failed"
