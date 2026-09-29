"""
Loads existing CSV outputs (Phase 2/3) into the SQLite knowledge base
(Phase 4 schema), and provides simple read helpers other modules will use.

Safe to rerun: documents/chunks use INSERT OR REPLACE (natural primary
keys); metrics/prices clear only the rows for the ticker being loaded,
then reinsert, since those tables use auto-increment IDs with no natural
uniqueness to key off of.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from core.config_loader import get_path, load_config
from core.logger import get_logger
from core.schemas import get_connection, initialize_schema

log = get_logger(__name__)


def load_documents(conn, config: dict) -> int:
    ticker = config["company"]["ticker"]
    path = get_path(config, "data_financial") / f"filings_index_{ticker}.csv"
    if not path.exists():
        log.warning("No filings index found at %s; skipping documents load", path)
        return 0

    df = pd.read_csv(path)
    df = df[df["form"].isin(["10-K", "10-Q"])]

    rows = 0
    for _, r in df.iterrows():
        document_id = f"{ticker}_{r['form']}_{r['accessionNumber']}"
        local_file = f"{ticker}_{r['form']}_{r['accessionNumber']}_{r['primaryDocument']}"
        conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(document_id, company_ticker, form, filing_date, report_date, "
            "accession_number, source_url, local_file) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (document_id, ticker, r["form"], r["filingDate"], r["reportDate"],
             r["accessionNumber"], r["url"], local_file),
        )
        rows += 1
    conn.commit()
    return rows


def load_chunks(conn, config: dict) -> int:
    ticker = config["company"]["ticker"]
    path = get_path(config, "data_processed") / f"chunks_{ticker}.csv"
    if not path.exists():
        log.warning("No chunks file found at %s; skipping chunks load", path)
        return 0

    df = pd.read_csv(path)

    rows = 0
    for _, r in df.iterrows():
        document_id = f"{r['company_ticker']}_{r['form']}_{_accession_from_source_file(r['source_file'])}"
        conn.execute(
            "INSERT OR REPLACE INTO chunks "
            "(chunk_id, document_id, chunk_index, word_count, text) "
            "VALUES (?, ?, ?, ?, ?)",
            (r["chunk_id"], document_id, r["chunk_index"], r["word_count"], r["text"]),
        )
        rows += 1
    conn.commit()
    return rows


def _accession_from_source_file(source_file: str) -> str:
    """
    Extract the accession number from a local filename like:
    NVDA_10-K_0001045810-16-000205_nvda-2016x10k.htm
    -> 0001045810-16-000205
    This lets us reconstruct the same document_id used in load_documents()
    without re-reading the filings index for every chunk row.
    """
    parts = source_file.split("_")
    return parts[2] if len(parts) > 2 else ""


def load_metrics(conn, config: dict) -> int:
    ticker = config["company"]["ticker"]
    path = get_path(config, "data_financial") / f"xbrl_facts_{ticker}.csv"
    if not path.exists():
        log.warning("No XBRL facts file found at %s; skipping metrics load", path)
        return 0

    df = pd.read_csv(path)

    # Clear only this ticker's existing metrics before reloading
    conn.execute("DELETE FROM metrics WHERE company_ticker = ?", (ticker,))

    rows = 0
    for _, r in df.iterrows():
        conn.execute(
            "INSERT INTO metrics (company_ticker, concept, unit, value, "
            "period_start, period_end, form, fiscal_year, fiscal_period, "
            "filed_date, accession_number) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ticker, r["concept"], r["unit"], r["value"],
             _none_if_nan(r.get("start")), r["end"], r.get("form"),
             _none_if_nan(r.get("fy")), r.get("fp"), r.get("filed"), r.get("accn")),
        )
        rows += 1
    conn.commit()
    return rows


def load_prices(conn, config: dict) -> int:
    tickers = [config["company"]["ticker"]] + [c["ticker"] for c in config.get("competitors", [])]
    roles = {config["company"]["ticker"]: "target"}
    for c in config.get("competitors", []):
        roles[c["ticker"]] = "competitor"

    market_dir = get_path(config, "data_market")
    total_rows = 0

    for ticker in tickers:
        path = market_dir / f"prices_{ticker}.csv"
        if not path.exists():
            log.warning("No price file found at %s; skipping", path)
            continue

        df = pd.read_csv(path)
        conn.execute("DELETE FROM prices WHERE company_ticker = ?", (ticker,))

        for _, r in df.iterrows():
            conn.execute(
                "INSERT INTO prices (company_ticker, role, date, open, high, "
                "low, close, adj_close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ticker, roles[ticker], r["date"], r["open"], r["high"],
                 r["low"], r["close"], r["adj_close"], r["volume"]),
            )
            total_rows += 1
    conn.commit()
    return total_rows


def _none_if_nan(value):
    """CSV NaN values need to become SQL NULL, not the literal string 'nan'."""
    if pd.isna(value):
        return None
    return value


def build_knowledge_base(db_path: Optional[Path] = None) -> dict:
    """Main entry: initialize schema, load all 4 tables, return row counts."""
    config = load_config()
    initialize_schema(db_path)
    conn = get_connection(db_path)

    try:
        counts = {
            "documents": load_documents(conn, config),
            "chunks": load_chunks(conn, config),
            "metrics": load_metrics(conn, config),
            "prices": load_prices(conn, config),
        }
    finally:
        conn.close()

    log.info("Knowledge base build complete: %s", counts)
    return counts


if __name__ == "__main__":
    result = build_knowledge_base()
    print("Rows loaded:")
    for table, count in result.items():
        print(f"  {table}: {count}")
