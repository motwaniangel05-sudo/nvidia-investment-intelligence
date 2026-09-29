"""Tests for filing document downloader: filenames, skip logic, error handling."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_acquisition.document_downloader import (
    already_downloaded,
    build_local_filename,
    download_filing_documents,
)
from data_acquisition.manifest import Manifest


def test_build_local_filename_format():
    name = build_local_filename(
        "NVDA", "10-K", "0001045810-16-000205", "nvda-2016x10k.htm"
    )
    assert name == "NVDA_10-K_0001045810-16-000205_nvda-2016x10k.htm"


def test_build_local_filename_handles_slash_in_form():
    # Some SEC forms use "10-K/A" (amendments) - '/' is unsafe in filenames
    name = build_local_filename(
        "NVDA", "10-K/A", "0001045810-16-000999", "amend.htm"
    )
    assert "/" not in name
    assert name == "NVDA_10-K-A_0001045810-16-000999_amend.htm"


def test_already_downloaded_true_when_url_in_manifest(tmp_path):
    manifest = Manifest(tmp_path / "manifest.csv")
    f = tmp_path / "sample.htm"
    f.write_text("some filing content")
    manifest.add(f, "http://example.com/filing.htm", "SEC_EDGAR", "10-K", "NVDA")
    assert already_downloaded(manifest, "http://example.com/filing.htm") is True


def test_already_downloaded_false_when_url_not_in_manifest(tmp_path):
    manifest = Manifest(tmp_path / "manifest.csv")
    assert already_downloaded(manifest, "http://not-seen-before.com") is False


def test_download_raises_if_filings_index_missing(tmp_path, monkeypatch):
    from core import config_loader

    fake_config = {
        "company": {"ticker": "NVDA"},
        "paths": {
            "data_financial": str(tmp_path / "financial"),
            "data_documents": str(tmp_path / "documents"),
        },
    }
    monkeypatch.setattr(config_loader, "load_config", lambda: fake_config)
    monkeypatch.setattr(
        "data_acquisition.document_downloader.load_config", lambda: fake_config
    )
    monkeypatch.setattr(
        "data_acquisition.document_downloader.get_path",
        lambda cfg, key: tmp_path / cfg["paths"][key],
    )

    with pytest.raises(FileNotFoundError):
        download_filing_documents()


def test_download_skips_already_downloaded_and_downloads_new(tmp_path, monkeypatch):
    # Build a fake filings index with two rows: one "already downloaded",
    # one new.
    financial_dir = tmp_path / "financial"
    documents_dir = tmp_path / "documents"
    financial_dir.mkdir()
    documents_dir.mkdir()

    index_df = pd.DataFrame([
        {
            "form": "10-K", "filingDate": "2016-03-17", "reportDate": "2016-01-31",
            "accessionNumber": "0001-16-000205", "primaryDocument": "a.htm",
            "url": "http://example.com/already.htm",
        },
        {
            "form": "10-Q", "filingDate": "2016-05-25", "reportDate": "2016-04-30",
            "accessionNumber": "0001-16-000275", "primaryDocument": "b.htm",
            "url": "http://example.com/new.htm",
        },
    ])
    index_df.to_csv(financial_dir / "filings_index_NVDA.csv", index=False)

    fake_config = {
        "company": {"ticker": "NVDA"},
        "sec": {"user_agent": "Test User test@example.com"},
        "acquisition": {
            "sec_min_seconds_between_requests": 0,
            "http_timeout_seconds": 5,
            "http_max_retries": 3,
            "manifest_file": str(tmp_path / "manifest.csv"),
        },
        "paths": {
            "data_financial": str(financial_dir),
            "data_documents": str(documents_dir),
        },
    }

    monkeypatch.setattr(
        "data_acquisition.document_downloader.load_config", lambda: fake_config
    )
    monkeypatch.setattr(
        "data_acquisition.document_downloader.get_path",
        lambda cfg, key: tmp_path / cfg["paths"][key]
        if not str(cfg["paths"][key]).startswith(str(tmp_path))
        else Path_from(cfg["paths"][key]),
    )

    # Simpler: patch get_path to just return the already-absolute paths given
    def fake_get_path(cfg, key):
        from pathlib import Path
        return Path(cfg["paths"][key])

    monkeypatch.setattr(
        "data_acquisition.document_downloader.get_path", fake_get_path
    )
    monkeypatch.setattr(
        "data_acquisition.manifest.load_config", lambda: fake_config
    )

    # Pre-populate manifest so the first URL looks "already downloaded"
    manifest = Manifest(tmp_path / "manifest.csv")
    pre_existing_file = documents_dir / "pre.htm"
    pre_existing_file.write_text("old content")
    manifest.add(pre_existing_file, "http://example.com/already.htm",
                 "SEC_EDGAR", "10-K", "NVDA")

    # Mock the HTTP client so the "new" URL succeeds without real network
    with patch("data_acquisition.document_downloader.SecClient") as MockClient:
        instance = MockClient.return_value
        fake_response = MagicMock()
        fake_response.content = b"<html>fake filing text</html>"
        instance.get.return_value = fake_response

        result = download_filing_documents()

    statuses = dict(zip(result["accessionNumber"], result["status"]))
    assert statuses["0001-16-000205"] == "skipped_already_downloaded"
    assert statuses["0001-16-000275"] == "downloaded"
    # The new file should actually exist on disk now
    downloaded_files = list(documents_dir.glob("NVDA_10-Q_0001-16-000275*"))
    assert len(downloaded_files) == 1


def test_download_records_failure_without_crashing(tmp_path, monkeypatch):
    financial_dir = tmp_path / "financial"
    documents_dir = tmp_path / "documents"
    financial_dir.mkdir()
    documents_dir.mkdir()

    index_df = pd.DataFrame([
        {
            "form": "10-K", "filingDate": "2016-03-17", "reportDate": "2016-01-31",
            "accessionNumber": "0001-16-000205", "primaryDocument": "a.htm",
            "url": "http://example.com/will-fail.htm",
        },
    ])
    index_df.to_csv(financial_dir / "filings_index_NVDA.csv", index=False)

    fake_config = {
        "company": {"ticker": "NVDA"},
        "sec": {"user_agent": "Test User test@example.com"},
        "acquisition": {
            "sec_min_seconds_between_requests": 0,
            "http_timeout_seconds": 5,
            "http_max_retries": 3,
            "manifest_file": str(tmp_path / "manifest.csv"),
        },
        "paths": {
            "data_financial": str(financial_dir),
            "data_documents": str(documents_dir),
        },
    }

    def fake_get_path(cfg, key):
        from pathlib import Path
        return Path(cfg["paths"][key])

    monkeypatch.setattr(
        "data_acquisition.document_downloader.load_config", lambda: fake_config
    )
    monkeypatch.setattr(
        "data_acquisition.document_downloader.get_path", fake_get_path
    )
    monkeypatch.setattr(
        "data_acquisition.manifest.load_config", lambda: fake_config
    )

    with patch("data_acquisition.document_downloader.SecClient") as MockClient:
        instance = MockClient.return_value
        instance.get.side_effect = Exception("simulated network failure")

        result = download_filing_documents()

    assert result.iloc[0]["status"].startswith("failed")
    # No file should have been written for a failed download
    assert list(documents_dir.glob("*.htm")) == []
