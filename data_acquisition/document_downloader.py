"""
Download the actual 10-K/10-Q filing documents (HTML) referenced in the
filings index, and record each in the manifest for provenance tracking.

These full documents feed Phase 3 (text extraction/chunking) and Phase 5
(RAG retrieval) -- unlike xbrl_facts.py, which gives us structured numbers,
this gives us the narrative text (MD&A, risk factors, footnotes).
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from core.config_loader import get_path, load_config
from core.logger import get_logger
from data_acquisition.manifest import Manifest
from data_acquisition.sec_client import SecClient

log = get_logger(__name__)

# Only these form types are downloaded here; 8-Ks are handled separately
# by the future News/Event agent (Phase 10), since they are short event
# announcements rather than substantial narrative documents.
DOWNLOADABLE_FORMS = ["10-K", "10-Q"]


def build_local_filename(ticker: str, form: str, accession_number: str, primary_document: str) -> str:
    """
    Build a predictable local filename, e.g.:
    NVDA_10-K_0001045810-16-000205_nvda-2016x10k.htm
    Including the accession number guarantees uniqueness even if two
    filings happen to share a primary document name.
    """
    safe_form = form.replace("/", "-")
    return f"{ticker}_{safe_form}_{accession_number}_{primary_document}"


def already_downloaded(manifest: Manifest, source_url: str) -> bool:
    """Check whether we already have a manifest record for this exact URL."""
    return manifest.find_by_url(source_url) is not None


def download_filing_documents() -> pd.DataFrame:
    """
    Main entry: read the filings index, download each 10-K/10-Q document
    not already downloaded, save it, and record it in the manifest.

    Returns a DataFrame summarizing what was downloaded/skipped/failed.
    """
    config = load_config()
    ticker = config["company"]["ticker"]

    index_path = get_path(config, "data_financial") / f"filings_index_{ticker}.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Filings index not found at {index_path}. "
            "Run `python -m data_acquisition.filings_index` first."
        )

    filings = pd.read_csv(index_path)
    filings = filings[filings["form"].isin(DOWNLOADABLE_FORMS)].copy()

    if filings.empty:
        log.warning("No 10-K/10-Q filings found in the index to download.")
        return pd.DataFrame()

    client = SecClient(config)
    manifest = Manifest()

    docs_dir = get_path(config, "data_documents")
    docs_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, str]] = []

    for _, row in filings.iterrows():
        source_url = row["url"]
        form = row["form"]
        accession = row["accessionNumber"]
        primary_doc = row["primaryDocument"]
        filing_date = row["filingDate"]

        local_name = build_local_filename(ticker, form, accession, primary_doc)
        local_path = docs_dir / local_name

        if already_downloaded(manifest, source_url):
            log.info("Already downloaded, skipping: %s", local_name)
            results.append({
                "form": form, "filingDate": filing_date,
                "accessionNumber": accession, "local_file": local_name,
                "status": "skipped_already_downloaded",
            })
            continue

        try:
            response = client.get(source_url)
        except Exception as e:
            log.error("Failed to download %s: %s", source_url, e)
            results.append({
                "form": form, "filingDate": filing_date,
                "accessionNumber": accession, "local_file": local_name,
                "status": f"failed: {e}",
            })
            continue

        local_path.write_bytes(response.content)
        manifest.add(
            local_path,
            source_url=source_url,
            source_name="SEC_EDGAR",
            doc_type=form,
            company_ticker=ticker,
        )
        log.info("Downloaded: %s (%d bytes)", local_name, len(response.content))
        results.append({
            "form": form, "filingDate": filing_date,
            "accessionNumber": accession, "local_file": local_name,
            "status": "downloaded",
        })

    summary = pd.DataFrame(results)
    return summary


if __name__ == "__main__":
    result = download_filing_documents()
    if result.empty:
        print("No documents processed.")
    else:
        print(result["status"].value_counts().to_string())
        print()
        failed = result[result["status"].str.startswith("failed")]
        if not failed.empty:
            print("FAILED downloads:")
            print(failed.to_string(index=False))
