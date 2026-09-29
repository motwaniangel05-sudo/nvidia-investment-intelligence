"""
Fetch structured financial facts from SEC's XBRL "company facts" API.

This gives us audited, machine-readable numbers (revenue, net income, EPS,
etc.) as originally tagged in each filing, instead of us having to parse
prose or HTML tables to find them.
"""

import json
from typing import Any, Dict, List

import pandas as pd

from core.config_loader import get_path, load_config
from core.logger import get_logger
from data_acquisition.manifest import Manifest
from data_acquisition.sec_client import SecClient

log = get_logger(__name__)

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Concepts we care about for this project (us-gaap taxonomy names).
# Not every concept exists for every company/period; missing ones are
# simply absent from the output, never invented.
CONCEPTS_OF_INTEREST = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "ResearchAndDevelopmentExpense",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "LongTermDebtNoncurrent",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
]


def fetch_company_facts(client: SecClient, cik: str) -> Dict[str, Any]:
    """Download the full companyfacts JSON for a company."""
    cik_padded = str(cik).zfill(10)
    url = COMPANYFACTS_URL.format(cik=cik_padded)
    log.info("Fetching XBRL company facts: %s", url)
    return client.get_json(url)


def flatten_facts(facts_json: Dict[str, Any], concepts: List[str] = None) -> pd.DataFrame:
    """
    Flatten the nested companyfacts JSON into one row per reported value.

    If `concepts` is given, only those us-gaap concepts are kept. Otherwise
    every us-gaap concept is included (useful for exploration).
    """
    rows = []
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})

    for concept, concept_data in us_gaap.items():
        if concepts is not None and concept not in concepts:
            continue
        for unit, entries in concept_data.get("units", {}).items():
            for entry in entries:
                rows.append({
                    "concept": concept,
                    "unit": unit,
                    "value": entry.get("val"),
                    "start": entry.get("start"),
                    "end": entry.get("end"),
                    "form": entry.get("form"),
                    "fy": entry.get("fy"),
                    "fp": entry.get("fp"),
                    "filed": entry.get("filed"),
                    "accn": entry.get("accn"),
                    "frame": entry.get("frame"),
                })

    if not rows:
        return pd.DataFrame(columns=[
            "concept", "unit", "value", "start", "end", "form",
            "fy", "fp", "filed", "accn", "frame",
        ])

    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.drop_duplicates()
    df = df.sort_values(["concept", "end"]).reset_index(drop=True)
    return df


def filter_by_year(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Keep only facts whose period end falls within our data period."""
    if df.empty:
        return df
    mask = (df["end"].dt.year >= start_year) & (df["end"].dt.year <= end_year)
    return df[mask].reset_index(drop=True)


def build_xbrl_facts() -> pd.DataFrame:
    """Main entry: fetch, flatten, filter, save, record in manifest."""
    config = load_config()
    company = config["company"]
    ticker = company["ticker"]
    cik = company["cik"]
    period = config["data_period"]

    client = SecClient(config)
    manifest = Manifest()

    facts_json = fetch_company_facts(client, cik)

    sec_name = facts_json.get("entityName", "")
    log.info("XBRL facts entity name: %s", sec_name)
    expected = company["short_name"].lower()
    if expected not in sec_name.lower():
        raise ValueError(
            f"CIK {cik} XBRL facts belong to '{sec_name}', which does not "
            f"match '{company['short_name']}'. Check config.yaml company.cik."
        )

    # Save raw JSON as evidence of exactly what SEC returned
    raw_dir = get_path(config, "data_raw") / "sec"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"companyfacts_{ticker}.json"
    raw_file.write_text(json.dumps(facts_json, indent=2), encoding="utf-8")
    manifest.add(
        raw_file,
        source_url=COMPANYFACTS_URL.format(cik=str(cik).zfill(10)),
        source_name="SEC_EDGAR_XBRL",
        doc_type="companyfacts",
        company_ticker=ticker,
    )

    df = flatten_facts(facts_json, concepts=CONCEPTS_OF_INTEREST)
    df = filter_by_year(df, period["start_year"], period["end_year"])

    available_concepts = sorted(df["concept"].unique()) if not df.empty else []
    missing_concepts = [c for c in CONCEPTS_OF_INTEREST if c not in available_concepts]
    if missing_concepts:
        log.warning(
            "These requested concepts were not found for this company: %s",
            missing_concepts,
        )

    out_dir = get_path(config, "data_financial")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"xbrl_facts_{ticker}.csv"
    df.to_csv(out_file, index=False)
    log.info("Saved XBRL facts: %s (%d rows, %d concepts)",
              out_file, len(df), len(available_concepts))
    return df


if __name__ == "__main__":
    result = build_xbrl_facts()
    if result.empty:
        print("No XBRL facts found.")
    else:
        print("Concepts found:", sorted(result["concept"].unique()))
        print()
        annual = result[(result["form"] == "10-K") & (result["concept"] == "Revenues")]
        print("Annual Revenues (10-K only):")
        print(annual[["end", "value", "fy"]].to_string(index=False))
