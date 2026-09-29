"""
Phase 2 wrap-up: read everything we've collected and report on completeness.

This does not fetch anything new -- it only inspects what is already on
disk/in the manifest and flags gaps, so we know exactly what we can rely on
before building Phase 3 (document processing) on top of it.
"""

from typing import List

import pandas as pd

from core.config_loader import get_path, load_config
from core.logger import get_logger
from data_acquisition.manifest import Manifest
from data_acquisition.xbrl_facts import CONCEPTS_OF_INTEREST

log = get_logger(__name__)


def check_filings_index(config: dict) -> List[str]:
    """Return a list of issue strings (empty list = no issues found)."""
    issues = []
    ticker = config["company"]["ticker"]
    path = get_path(config, "data_financial") / f"filings_index_{ticker}.csv"

    if not path.exists():
        return [f"filings_index_{ticker}.csv is missing entirely"]

    df = pd.read_csv(path, parse_dates=["filingDate"])
    years = range(config["data_period"]["start_year"], config["data_period"]["end_year"] + 1)

    tenk_years = set(df[df["form"] == "10-K"]["filingDate"].dt.year)
    for y in years:
        # A 10-K reporting on fiscal year Y is typically filed in Y or Y+1
        if y not in tenk_years and (y + 1) not in tenk_years:
            issues.append(f"No 10-K found filed in {y} or {y+1}")

    tenq_count = len(df[df["form"] == "10-Q"])
    expected_min = len(list(years)) * 2  # be lenient; some years partial
    if tenq_count < expected_min:
        issues.append(
            f"Only {tenq_count} 10-Q filings found; expected roughly "
            f"{len(list(years)) * 3} for {len(list(years))} years"
        )

    return issues


def check_xbrl_facts(config: dict) -> List[str]:
    issues = []
    ticker = config["company"]["ticker"]
    path = get_path(config, "data_financial") / f"xbrl_facts_{ticker}.csv"

    if not path.exists():
        return [f"xbrl_facts_{ticker}.csv is missing entirely"]

    df = pd.read_csv(path)
    found_concepts = set(df["concept"].unique())
    missing = [c for c in CONCEPTS_OF_INTEREST if c not in found_concepts]
    if missing:
        issues.append(f"XBRL concepts with zero data: {missing}")

    if df.empty:
        issues.append("XBRL facts file exists but has zero rows")

    return issues


def check_documents(config: dict) -> List[str]:
    """Verify every 10-K/10-Q in the filings index has a downloaded file."""
    issues = []
    ticker = config["company"]["ticker"]
    index_path = get_path(config, "data_financial") / f"filings_index_{ticker}.csv"
    docs_dir = get_path(config, "data_documents")

    if not index_path.exists():
        return ["Cannot check documents: filings index is missing"]

    df = pd.read_csv(index_path)
    df = df[df["form"].isin(["10-K", "10-Q"])]

    existing_files = {f.name for f in docs_dir.glob("*.htm")}
    missing_docs = []
    for _, row in df.iterrows():
        expected_name = f"{ticker}_{row['form']}_{row['accessionNumber']}_{row['primaryDocument']}"
        if expected_name not in existing_files:
            missing_docs.append(expected_name)

    if missing_docs:
        issues.append(
            f"{len(missing_docs)} filing document(s) referenced in the index "
            f"but not downloaded: {missing_docs[:3]}{'...' if len(missing_docs) > 3 else ''}"
        )

    return issues


def check_market_data(config: dict) -> List[str]:
    issues = []
    tickers = [config["company"]["ticker"]] + [c["ticker"] for c in config.get("competitors", [])]
    market_dir = get_path(config, "data_market")

    date_ranges = {}
    for ticker in tickers:
        path = market_dir / f"prices_{ticker}.csv"
        if not path.exists():
            issues.append(f"prices_{ticker}.csv is missing entirely")
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            issues.append(f"prices_{ticker}.csv exists but has zero rows")
            continue
        date_ranges[ticker] = (df["date"].min(), df["date"].max(), len(df))

    if date_ranges:
        # Flag tickers whose date range is inconsistent with the others
        first_dates = {v[0] for v in date_ranges.values()}
        if len(first_dates) > 1:
            issues.append(f"Tickers have mismatched start dates: {date_ranges}")

    return issues


def check_manifest_integrity(config: dict) -> List[str]:
    issues = []
    manifest = Manifest()
    rows = manifest.read_all()

    if not rows:
        return ["Manifest is empty"]

    for row in rows:
        if len(row.get("sha256", "")) != 64:
            issues.append(f"Invalid hash for {row.get('file_path')}")
        from pathlib import Path
        from core.config_loader import PROJECT_ROOT
        full_path = PROJECT_ROOT / row["file_path"]
        if not full_path.exists():
            issues.append(f"Manifest references missing file: {row['file_path']}")

    return issues


def run_validation_report() -> bool:
    """
    Run all checks, print a report, and return True if everything passed
    (no issues found across all checks).
    """
    config = load_config()

    checks = {
        "Filings Index": check_filings_index(config),
        "XBRL Financial Facts": check_xbrl_facts(config),
        "Filing Documents": check_documents(config),
        "Market Data": check_market_data(config),
        "Manifest Integrity": check_manifest_integrity(config),
    }

    print("=" * 70)
    print("PHASE 2 DATA VALIDATION REPORT")
    print("=" * 70)

    all_passed = True
    for section, issues in checks.items():
        status = "PASS" if not issues else f"ISSUES ({len(issues)})"
        print(f"\n[{status}] {section}")
        if issues:
            all_passed = False
            for issue in issues:
                print(f"  - {issue}")

    print("\n" + "=" * 70)
    print("OVERALL:", "ALL CHECKS PASSED" if all_passed else "ISSUES FOUND — review above")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    import sys
    passed = run_validation_report()
    sys.exit(0 if passed else 1)
