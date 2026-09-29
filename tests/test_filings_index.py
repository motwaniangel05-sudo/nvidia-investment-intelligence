"""Tests for filings catalog building, filtering, URL generation, manifest."""

import pandas as pd

from data_acquisition.filings_index import (
    build_filing_url,
    filter_filings,
    submissions_to_dataframe,
)
from data_acquisition.manifest import Manifest


def fake_submissions():
    return {
        "name": "NVIDIA CORP",
        "filings": {
            "recent": {
                "accessionNumber": ["0001-24-000001", "0001-24-000002",
                                    "0001-24-000003", "0001-15-000009"],
                "form": ["10-K", "10-Q", "4", "10-K"],
                "filingDate": ["2024-02-21", "2024-05-22", "2024-06-01", "2015-02-20"],
                "reportDate": ["2024-01-28", "2024-04-28", "", "2015-01-25"],
                "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.htm"],
            },
            "files": [],
        },
    }


def test_submissions_to_dataframe_shape():
    df = submissions_to_dataframe(fake_submissions())
    assert len(df) == 4
    assert "accessionNumber" in df.columns


def test_empty_submissions_returns_empty_df():
    assert submissions_to_dataframe({}).empty


def test_filter_keeps_only_requested_forms_and_years():
    df = submissions_to_dataframe(fake_submissions())
    out = filter_filings(df, ["10-K", "10-Q"], 2016, 2026)
    assert set(out["form"]) == {"10-K", "10-Q"}       # form '4' removed
    assert (out["filingDate"].dt.year >= 2016).all()  # 2015 filing removed
    assert len(out) == 2


def test_filter_removes_duplicates():
    df = submissions_to_dataframe(fake_submissions())
    df = pd.concat([df, df], ignore_index=True)
    out = filter_filings(df, ["10-K", "10-Q"], 2016, 2026)
    assert out["accessionNumber"].is_unique


def test_build_filing_url():
    url = build_filing_url("0001045810", "0001045810-24-000029", "nvda-20240128.htm")
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581024000029/nvda-20240128.htm"
    )


def test_manifest_skips_duplicate_content(tmp_path):
    manifest = Manifest(tmp_path / "manifest.csv")
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("same content")
    f2.write_text("same content")
    manifest.add(f1, "http://u1", "TEST", "doc", "NVDA")
    manifest.add(f2, "http://u2", "TEST", "doc", "NVDA")
    assert len(manifest.read_all()) == 1   # duplicate skipped


def test_manifest_records_different_content(tmp_path):
    manifest = Manifest(tmp_path / "manifest.csv")
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("content one")
    f2.write_text("content two")
    manifest.add(f1, "http://u1", "TEST", "doc", "NVDA")
    manifest.add(f2, "http://u2", "TEST", "doc", "NVDA")
    assert len(manifest.read_all()) == 2
