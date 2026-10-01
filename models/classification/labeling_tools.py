"""
Tools for building the manually-labeled training set for the News/Event
classifier (FACT / CLAIM / INTERPRETATION / UNCERTAINTY).

This is the honest, non-fabricated path to training data per project
constraints: real sentences from real 8-K filings, labeled by a human
(not invented, not scraped from an already-labeled dataset without
attribution).
"""

import csv
import re
from pathlib import Path
from typing import List

import pandas as pd

from core.config_loader import get_path, load_config
from core.logger import get_logger

log = get_logger(__name__)

MIN_SENTENCE_WORDS = 6
MAX_SENTENCE_WORDS = 60  # longer than this is likely a mis-split run-on

LABELS = ["FACT", "CLAIM", "INTERPRETATION", "UNCERTAINTY"]


def split_into_sentences(text: str) -> List[str]:
    """Split chunk text into individual sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def is_labelable_sentence(sentence: str) -> bool:
    """Filter out fragments too short/long to be a good labeling candidate."""
    word_count = len(sentence.split())
    if word_count < MIN_SENTENCE_WORDS or word_count > MAX_SENTENCE_WORDS:
        return False
    # Skip pure headers/boilerplate (all-caps, or no lowercase letters at all)
    if sentence.isupper():
        return False
    return True


def extract_candidate_sentences(sample_size: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Load BOTH 8-K chunks and primary 10-K/10-Q chunks, split into
    sentences, filter to labelable candidates, and return a random
    sample for manual labeling. seed is fixed for reproducibility.

    Rationale (discovered during Phase 10 testing): 8-K cover-page
    sections are mostly procedural FACT (bylaw amendments, indentures,
    board appointments). Genuine CLAIM/INTERPRETATION/UNCERTAINTY
    language concentrates in 10-K risk factors, MD&A, and forward-looking
    statement sections. Combining both sources gives real class balance
    without fabricating or cherry-picking examples -- everything is still
    real filing text.
    """
    config = load_config()
    ticker = config["company"]["ticker"]

    sources = [
        get_path(config, "data_processed") / f"chunks_{ticker}_8K.csv",
        get_path(config, "data_processed") / f"chunks_{ticker}.csv",
    ]

    all_sentences = []
    for chunks_path in sources:
        if not chunks_path.exists():
            log.warning("Chunk source not found, skipping: %s", chunks_path)
            continue
        chunks_df = pd.read_csv(chunks_path)
        for _, row in chunks_df.iterrows():
            sentences = split_into_sentences(row["text"])
            for s in sentences:
                if is_labelable_sentence(s):
                    all_sentences.append({
                        "sentence": s,
                        "source_chunk_id": row["chunk_id"],
                        "filing_date": row["filing_date"],
                    })

    if not all_sentences:
        raise FileNotFoundError(
            "No chunk sources found. Run rag.chunker.build_all_chunks() "
            "for both 10-K/10-Q and 8-K forms first."
        )

    df = pd.DataFrame(all_sentences).drop_duplicates(subset=["sentence"])
    log.info("Extracted %d unique labelable sentences from all chunk sources", len(df))

    if len(df) <= sample_size:
        return df.reset_index(drop=True)

    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def save_labeling_template(df: pd.DataFrame, output_path: Path = None) -> Path:
    """
    Save sentences to a CSV with an empty 'label' column for manual
    filling in (e.g. in Excel, Google Sheets, or a text editor).
    """
    if output_path is None:
        config = load_config()
        output_path = get_path(config, "data_processed") / "labeling_template.csv"

    df = df.copy()
    df["label"] = ""  # to be filled in manually
    df.to_csv(output_path, index=False)
    log.info("Saved labeling template: %s (%d sentences)", output_path, len(df))
    return output_path


if __name__ == "__main__":
    sample = extract_candidate_sentences(sample_size=100)
    path = save_labeling_template(sample)
    print(f"Saved {len(sample)} sentences to: {path}")
    print(f"\\nValid labels: {LABELS}")
    print("\\nOpen the CSV, fill in the 'label' column for each row, then save.")
