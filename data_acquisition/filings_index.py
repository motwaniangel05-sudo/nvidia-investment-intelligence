"""
Build a clean catalog of a company's SEC filings.

The SEC "submissions" JSON stores filings as parallel arrays (one array per
field). We convert that into a tidy table, filter to the form types and years
we care about, and generate the URL of each filing's main document.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from core.config_loader import PROJECT_ROOT, get_path, load_config
from core.logger import get_logger
from data_acquisition.manifest import Manifest
from data_acquisition.sec_client import SecClient

log = get_logger(__name__)

ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}"


def submissions_to_dataframe(submissions: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert SEC submissions JSON into a DataFrame with one row per filing.

    Recent filings live in submissions['filings']['recent']. Older filings may
    be split into extra files listed in submissions['filings']['files'];
    those are returned by fetch_all_filings() below.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent or "accessionNumber" not in recent:
        return pd.DataFrame()
    return pd.DataFrame(recent)


def build_filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    cik_int = int(cik)  # URL uses CIK without leading zeros
    accession_nodash = accession_number.replace("-", "")
    return ARCHIVES_URL.format(
        cik_int=cik_int,
        accession_nodash=accession_nodash,
        primary_doc=primary_document,
    )


def fetch_all_filings(client: SecClient, cik: str, submissions: Dict[str, Any]) -> pd.DataFrame:
    """
    Combine 'recent' filings with older filing files if the company has many.
    NVIDIA files a lot, so older years are often in these extra files.
    """
    frames: List[pd.DataFrame] = [submissions_to_dataframe(submissions)]

    for extra in submissions.get("filings", {}).get("files", []):
        name = extra.get("name")
        if not name:
            continue
        url = f"https://data.sec.gov/submissions/{name}"
        log.info("Fetching older filings file: %s", name)
        older = client.get_json(url)
        if older and "accessionNumber" in older:
            frames.append(pd.DataFrame(older))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def filter_filings(df: pd.DataFrame, forms: List[str], start_year: int, end_year: int) -> pd.DataFrame:
    """Keep chosen form types within the year range; add year + URL-ready columns."""
    if df.empty:
        return df

    df = df.copy()
    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
    df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce")

    df = df[df["form"].isin(forms)]
    df = df[
        (df["filingDate"].dt.year >= start_year)
        & (df["filingDate"].dt.year <= end_year)
    ]
    df = df.drop_duplicates(subset=["accessionNumber"])
    df = df.sort_values("filingDate").reset_index(drop=True)
    return df


def build_filings_index() -> pd.DataFrame:
    """Main entry: fetch catalog, filter, save raw JSON + CSV catalog, log to manifest."""
    config = load_config()
    company = config["company"]
    ticker = company["ticker"]
    cik = company["cik"]
    period = config["data_period"]
    forms = config["acquisition"]["filing_forms"]

    client = SecClient(config)
    manifest = Manifest()

    submissions = client.get_submissions(cik)

    # Verify the CIK really belongs to who we think it does
    sec_name = submissions.get("name", "")
    log.info("SEC reports this CIK belongs to: %s", sec_name)
    expected = company["short_name"].lower()
    if expected not in sec_name.lower():
        raise ValueError(
            f"CIK {cik} belongs to '{sec_name}', which does not match "
            f"'{company['short_name']}'. Check config.yaml company.cik."
        )

    # Save the raw response (evidence of what SEC told us, and when)
    raw_dir = get_path(config, "data_raw") / "sec"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"submissions_{ticker}.json"
    raw_file.write_text(json.dumps(submissions, indent=2), encoding="utf-8")
    manifest.add(
        raw_file,
        source_url=f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json",
        source_name="SEC_EDGAR",
        doc_type="submissions",
        company_ticker=ticker,
    )

    df_all = fetch_all_filings(client, cik, submissions)
    df = filter_filings(df_all, forms, period["start_year"], period["end_year"])

    if df.empty:
        log.warning("No filings found for the given forms and years.")
        return df

    df["url"] = df.apply(
        lambda r: build_filing_url(cik, r["accessionNumber"], r["primaryDocument"]),
        axis=1,
    )

    keep = ["form", "filingDate", "reportDate", "accessionNumber",
            "primaryDocument", "url"]
    df = df[keep]

    out_dir = get_path(config, "data_financial")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"filings_index_{ticker}.csv"
    df.to_csv(out_file, index=False)
    log.info("Saved filings catalog: %s (%d filings)", out_file, len(df))
    return df


if __name__ == "__main__":
    result = build_filings_index()
    if result.empty:
        print("No filings found.")
    else:
        print(result.groupby("form").size().to_string())
        print()
        print(result[result["form"] == "10-K"][["filingDate", "reportDate", "url"]].to_string(index=False))
