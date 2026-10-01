"""
Risk identification and scoring methodology.

SEVERITY METHODOLOGY (stated explicitly, per project requirement to never
assign scores without explaining methodology):
Severity is proxied by FREQUENCY OF MENTION -- how many distinct fiscal
years' worth of filings discuss a given risk category. A risk NVIDIA
re-discloses across many years is treated as more persistently material
than one mentioned rarely. This is a defensible but NOT definitive
heuristic: a risk could be severe but newly-emerged (low frequency, high
real severity) or routinely disclosed but low-impact (high frequency,
low real severity). This limitation is disclosed in every output.

UNCERTAINTY CORROBORATION: Phase 10's trained classifier is applied to
each retrieved risk chunk; if it predicts UNCERTAINTY, this is reported
as corroborating signal (the chunk uses hedged/risk language), not as
independent proof of severity.
"""

import pickle
from collections import defaultdict
from typing import Dict, List

from core.config_loader import get_path, load_config
from core.logger import get_logger
from core.schemas import get_connection
from rag.retriever import retrieve as rag_retrieve

log = get_logger(__name__)

# Each category maps to a specific, keyword-rich search query (per Phase 5
# finding: specific queries dramatically outperform vague ones for TF-IDF).
RISK_CATEGORIES = {
    "competitive": "competition competitors market share pricing pressure",
    "regulatory": "export restrictions regulations government license compliance",
    "geopolitical": "China Taiwan trade tensions geopolitical sanctions",
    "supply_chain": "supply chain manufacturing capacity foundry semiconductor shortage",
    "customer_concentration": "customer concentration significant customers dependent",
    "technology": "technology obsolescence innovation research development failure",
    "financial": "debt liquidity capital interest rate currency",
    "execution": "integration acquisition execution management key personnel",
}

SEVERITY_THRESHOLDS = {"HIGH": 7, "MEDIUM": 3, "LOW": 1}  # min distinct years mentioned


def _severity_from_frequency(distinct_years: int) -> str:
    """Map distinct-year mention count to a severity label per thresholds above."""
    if distinct_years >= SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    if distinct_years >= SEVERITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def _load_classifier():
    config = load_config()
    path = get_path(config, "models") / "classification" / "event_classifier.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _classify_for_uncertainty(text: str, classifier: dict) -> bool:
    """Returns True if the classifier predicts UNCERTAINTY for this text."""
    if classifier is None:
        return False
    vec = classifier["vectorizer"].transform([text[:500]])
    pred = classifier["model"].predict(vec)[0]
    return pred == "UNCERTAINTY"


def identify_risks(company_ticker: str, top_k_per_category: int = 8) -> List[Dict]:
    """
    For each risk category, retrieve relevant chunks via RAG, compute
    frequency-based severity (distinct filing years mentioning it), and
    corroborate with the Phase 10 classifier. Returns one risk record per
    category with full evidence and explicit methodology notes.
    """
    classifier = _load_classifier()
    risks = []

    for category, query in RISK_CATEGORIES.items():
        chunks = rag_retrieve(query, top_k=top_k_per_category)
        if not chunks:
            log.warning("No chunks retrieved for risk category: %s", category)
            continue

        distinct_years = {c["filing_date"][:4] for c in chunks if c.get("filing_date")}
        severity = _severity_from_frequency(len(distinct_years))

        uncertainty_corroborated = sum(
            1 for c in chunks if _classify_for_uncertainty(c["text"], classifier)
        )

        top_evidence = sorted(chunks, key=lambda c: c["score"], reverse=True)[:3]

        risks.append({
            "category": category,
            "severity": severity,
            "severity_methodology": (
                f"Severity derived from frequency of mention: this risk topic "
                f"appears in filings spanning {len(distinct_years)} distinct "
                f"fiscal year(s) (years: {sorted(distinct_years)}). "
                f"Thresholds: HIGH>={SEVERITY_THRESHOLDS['HIGH']} years, "
                f"MEDIUM>={SEVERITY_THRESHOLDS['MEDIUM']} years, else LOW. "
                f"This is a frequency-based proxy, not a definitive measure "
                f"of real-world impact -- a newly-emerged severe risk could "
                f"score LOW under this methodology."
            ),
            "distinct_years_mentioned": len(distinct_years),
            "uncertainty_corroboration": (
                f"{uncertainty_corroborated}/{len(chunks)} retrieved chunks "
                f"were independently classified as UNCERTAINTY language by "
                f"the Phase 10 classifier (itself a limited model trained on "
                f"85 examples -- see Phase 10 notes)."
            ),
            "evidence": [
                {"chunk_id": c["chunk_id"], "text": c["text"][:400],
                 "score": c["score"], "filing_date": c["filing_date"]}
                for c in top_evidence
            ],
        })

    return risks


if __name__ == "__main__":
    risks = identify_risks("NVDA")
    for r in risks:
        print(f"\n=== {r['category'].upper()} — Severity: {r['severity']} ===")
        print(r["severity_methodology"])
        print(r["uncertainty_corroboration"])
        print(f"Top evidence: {r['evidence'][0]['text'][:200]}...")
