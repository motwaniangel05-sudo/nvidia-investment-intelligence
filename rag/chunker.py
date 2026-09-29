"""
Split clean filing text (from parser.py) into retrievable chunks with
metadata, ready for Phase 5's RAG retrieval index.

Chunking strategy: paragraph-aware with a target word count and overlap.
Small consecutive paragraphs are merged together; large paragraphs are
split on sentence boundaries. Overlap between chunks prevents relevant
context from being lost right at a chunk boundary.
"""

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pandas as pd

from core.config_loader import get_path, load_config
from core.logger import get_logger
from rag.parser import ParsingError, extract_text

log = get_logger(__name__)

TARGET_CHUNK_WORDS = 800
OVERLAP_WORDS = 150
MIN_CHUNK_WORDS = 50  # chunks smaller than this are merged into a neighbor


@dataclass
class Chunk:
    """One retrievable unit of filing text, with full provenance metadata."""
    chunk_id: str
    company_ticker: str
    form: str
    filing_date: str
    source_file: str
    chunk_index: int
    word_count: int
    text: str


def split_into_paragraphs(text: str) -> List[str]:
    """Split clean text back into paragraphs (parser.py joins with \\n\\n)."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def split_long_paragraph(paragraph: str, max_words: int) -> List[str]:
    """
    Split an oversized paragraph on sentence boundaries, keeping each
    piece under max_words. Falls back to a hard word-count split if a
    single "sentence" is itself absurdly long (e.g. a run-on legal clause).
    """
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if len(words) > max_words:
            # Single sentence too long on its own: hard-split by words
            if current:
                pieces.append(" ".join(current))
                current, current_len = [], 0
            for i in range(0, len(words), max_words):
                pieces.append(" ".join(words[i:i + max_words]))
            continue

        if current_len + len(words) > max_words and current:
            pieces.append(" ".join(current))
            current, current_len = [], 0

        current.append(sentence)
        current_len += len(words)

    if current:
        pieces.append(" ".join(current))

    return pieces


def merge_and_split(
    paragraphs: List[str],
    target_words: int = TARGET_CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> List[str]:
    """
    Combine paragraphs into chunks close to target_words, splitting any
    paragraph that alone exceeds target_words, and carrying a small word
    overlap forward into the next chunk for retrieval continuity.
    """
    # First, break down any individually oversized paragraphs
    normalized: List[str] = []
    for p in paragraphs:
        if len(p.split()) > target_words:
            normalized.extend(split_long_paragraph(p, target_words))
        else:
            normalized.append(p)

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for para in normalized:
        para_words = len(para.split())

        if current_words + para_words > target_words and current:
            chunk_text = " ".join(current)
            chunks.append(chunk_text)

            # Build overlap: carry the tail words of this chunk forward
            tail_words = chunk_text.split()[-overlap_words:]
            current = [" ".join(tail_words)] if tail_words else []
            current_words = len(tail_words)

        current.append(para)
        current_words += para_words

    if current:
        remaining_text = " ".join(current)
        if len(remaining_text.split()) < MIN_CHUNK_WORDS and chunks:
            # Too small on its own: merge into the previous chunk
            chunks[-1] = chunks[-1] + " " + remaining_text
        else:
            chunks.append(remaining_text)

    return chunks


def chunk_document(
    file_path: Path,
    company_ticker: str,
    form: str,
    filing_date: str,
) -> List[Chunk]:
    """
    Full pipeline for one filing document: extract text, chunk it, attach
    metadata. Returns an empty list (not an exception) if extraction fails
    or yields no usable text -- a bad document should not stop the batch.
    """
    try:
        text = extract_text(file_path)
    except ParsingError as e:
        log.error("Skipping %s due to parsing error: %s", file_path.name, e)
        return []

    if not text:
        log.warning("No usable text extracted from %s", file_path.name)
        return []

    paragraphs = split_into_paragraphs(text)
    chunk_texts = merge_and_split(paragraphs)

    chunks = []
    for i, chunk_text in enumerate(chunk_texts):
        chunk_id = f"{company_ticker}_{form}_{filing_date}_{i:04d}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            company_ticker=company_ticker,
            form=form,
            filing_date=filing_date,
            source_file=file_path.name,
            chunk_index=i,
            word_count=len(chunk_text.split()),
            text=chunk_text,
        ))
    return chunks


def build_all_chunks() -> pd.DataFrame:
    """
    Main entry: chunk every downloaded 10-K/10-Q document, combine into
    one DataFrame, save to data/processed/chunks_{TICKER}.csv.
    """
    config = load_config()
    ticker = config["company"]["ticker"]

    index_path = get_path(config, "data_financial") / f"filings_index_{ticker}.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Filings index not found at {index_path}. Run Phase 2 first."
        )

    filings = pd.read_csv(index_path)
    filings = filings[filings["form"].isin(["10-K", "10-Q"])]

    docs_dir = get_path(config, "data_documents")
    all_chunks: List[Chunk] = []
    skipped = []

    for _, row in filings.iterrows():
        local_name = f"{ticker}_{row['form']}_{row['accessionNumber']}_{row['primaryDocument']}"
        file_path = docs_dir / local_name

        if not file_path.exists():
            skipped.append(local_name)
            continue

        doc_chunks = chunk_document(
            file_path,
            company_ticker=ticker,
            form=row["form"],
            filing_date=row["filingDate"],
        )
        all_chunks.extend(doc_chunks)

    if skipped:
        log.warning("Skipped %d documents not found on disk: %s",
                     len(skipped), skipped[:3])

    if not all_chunks:
        log.warning("No chunks were produced from any document.")
        return pd.DataFrame()

    df = pd.DataFrame([asdict(c) for c in all_chunks])

    out_dir = get_path(config, "data_processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"chunks_{ticker}.csv"
    df.to_csv(out_file, index=False)
    log.info("Saved %d chunks from %d documents to %s",
              len(df), len(filings) - len(skipped), out_file)
    return df


if __name__ == "__main__":
    result = build_all_chunks()
    if result.empty:
        print("No chunks produced.")
    else:
        print(f"Total chunks: {len(result)}")
        print(f"Documents represented: {result['source_file'].nunique()}")
        print(f"Average words per chunk: {result['word_count'].mean():.0f}")
        print(f"Min/Max words per chunk: {result['word_count'].min()} / {result['word_count'].max()}")
        print()
        print("Sample chunk (first row):")
        print(f"  chunk_id: {result.iloc[0]['chunk_id']}")
        print(f"  words: {result.iloc[0]['word_count']}")
        print(f"  text preview: {result.iloc[0]['text'][:300]}...")
