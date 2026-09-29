"""
Manifest: a provenance log of every file we download.

Each record answers: what is this file, where did it come from, when did we
get it, and has its content changed? This is the foundation of the project's
"every claim must be traceable" rule, and it powers duplicate detection.
"""

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from core.config_loader import PROJECT_ROOT, load_config
from core.logger import get_logger

log = get_logger(__name__)

MANIFEST_COLUMNS = [
    "file_path",       # relative to project root
    "source_url",
    "source_name",     # e.g. "SEC_EDGAR", "YAHOO_FINANCE"
    "doc_type",        # e.g. "submissions", "10-K", "prices"
    "company_ticker",
    "downloaded_at",   # UTC ISO timestamp
    "sha256",
    "size_bytes",
]


def sha256_of_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Manifest:
    def __init__(self, manifest_path: Optional[Path] = None):
        if manifest_path is None:
            config = load_config()
            rel = config.get("acquisition", {}).get(
                "manifest_file", "data/raw/manifest.csv"
            )
            manifest_path = PROJECT_ROOT / rel
        self.path = Path(manifest_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS).writeheader()

    def read_all(self) -> List[Dict[str, str]]:
        with open(self.path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def find_by_hash(self, sha256: str) -> Optional[Dict[str, str]]:
        """Return the existing record with this content hash, if any."""
        for row in self.read_all():
            if row["sha256"] == sha256:
                return row
        return None

    def find_by_url(self, source_url: str) -> Optional[Dict[str, str]]:
        for row in self.read_all():
            if row["source_url"] == source_url:
                return row
        return None

    def add(
        self,
        file_path: Path,
        source_url: str,
        source_name: str,
        doc_type: str,
        company_ticker: str,
    ) -> Dict[str, str]:
        """Record a downloaded file. Skips exact duplicates (same hash)."""
        file_path = Path(file_path)
        digest = sha256_of_file(file_path)

        existing = self.find_by_hash(digest)
        if existing:
            log.info(
                "Duplicate content skipped (already recorded as %s)",
                existing["file_path"],
            )
            return existing

        try:
            relative = str(file_path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            relative = str(file_path)

        record = {
            "file_path": relative,
            "source_url": source_url,
            "source_name": source_name,
            "doc_type": doc_type,
            "company_ticker": company_ticker,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": digest,
            "size_bytes": str(file_path.stat().st_size),
        }
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS).writerow(record)
        log.info("Manifest: recorded %s", relative)
        return record
