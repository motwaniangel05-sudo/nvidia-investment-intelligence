"""
Extract clean, readable text from raw SEC filing HTML documents.

SEC filings are dense HTML with scripts, inline styles, and large financial
tables mixed into narrative prose. This module strips the non-prose content
and returns clean paragraph text, ready for chunking (chunker.py).

Financial tables are deliberately NOT parsed here -- structured numbers
already come from XBRL (Phase 2, xbrl_facts.py), which is more reliable
than scraping HTML tables. This module focuses on narrative text only
(MD&A, risk factors, business description, notes).
"""

import re
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup

from core.logger import get_logger

log = get_logger(__name__)

# Tags whose content is never useful prose and should be dropped entirely
TAGS_TO_REMOVE = ["script", "style", "table", "head", "noscript"]

# Block-level tags we treat as paragraph boundaries when extracting text
BLOCK_TAGS = ["p", "div", "td", "li", "br"]


class ParsingError(Exception):
    """Raised when a filing document cannot be parsed at all."""


def load_html(file_path: Path) -> str:
    """Read an HTML file from disk, trying common encodings."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise ParsingError(f"File not found: {file_path}")

    for encoding in ("utf-8", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ParsingError(f"Could not decode {file_path} with utf-8 or latin-1")


def strip_unwanted_tags(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove tags that never contain useful narrative prose."""
    for tag_name in TAGS_TO_REMOVE:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    return soup


def extract_paragraphs(soup: BeautifulSoup) -> List[str]:
    """
    Pull text out block-by-block so paragraph boundaries are preserved.
    Using get_text() on the whole document at once collapses everything
    into one run-on blob; extracting per block tag keeps structure.
    """
    paragraphs = []
    for tag in soup.find_all(BLOCK_TAGS):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)
    return paragraphs


def clean_text(text: str) -> str:
    """Normalize whitespace and drop common SEC filing boilerplate noise."""
    # Remove stylized checkbox glyphs (e.g. "Yes \u00fd No o") left over
    # from cover-page yes/no questions once their special font is stripped
    text = re.sub(r"\b(Yes|No)\s+[\u00fd\u00fem\u00e9o\u00a8]\b", r"\1", text)
    # Collapse multiple spaces/tabs into one
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_boilerplate(paragraph: str) -> bool:
    """
    Filter out common non-content lines: page numbers, table-of-contents
    dots, lone punctuation, and very short fragments with no real content.
    """
    stripped = paragraph.strip()
    if len(stripped) < 15:
        return True
    if re.fullmatch(r"[\d\.\s\-–—]+", stripped):  # just numbers/dashes/dots
        return True
    if re.fullmatch(r"(page\s*)?\d+", stripped, flags=re.IGNORECASE):
        return True
    return False


def extract_text(file_path: Path) -> str:
    """
    Main entry: load an HTML filing and return clean, paragraph-separated
    prose text with boilerplate and tables stripped out.
    """
    raw_html = load_html(file_path)

    try:
        soup = BeautifulSoup(raw_html, "lxml")
    except Exception as e:
        raise ParsingError(f"Failed to parse HTML for {file_path}: {e}") from e

    soup = strip_unwanted_tags(soup)
    paragraphs = extract_paragraphs(soup)
    paragraphs = [p for p in paragraphs if not is_boilerplate(p)]

    # De-duplicate consecutive identical paragraphs (common when block tags
    # are nested, e.g. a <div> wrapping a single <p> produces the same text
    # twice if we are not careful -- extract_paragraphs above avoids most of
    # this, but we guard here too since it's cheap and safe)
    deduped = []
    for p in paragraphs:
        if not deduped or deduped[-1] != p:
            deduped.append(p)

    full_text = "\n\n".join(deduped)
    full_text = clean_text(full_text)

    if not full_text:
        log.warning("Extracted zero usable text from %s", file_path)

    return full_text


if __name__ == "__main__":
    import sys
    from core.config_loader import get_path, load_config

    config = load_config()
    docs_dir = get_path(config, "data_documents")
    files = sorted(docs_dir.glob("*10-K*.htm"))

    if not files:
        print("No 10-K documents found. Run document_downloader first.")
        sys.exit(1)

    sample = files[0]
    print(f"Parsing sample file: {sample.name}")
    text = extract_text(sample)
    print(f"Extracted {len(text)} characters, {len(text.split())} words")
    print("\n--- First 1000 characters ---\n")
    print(text[:1000])
