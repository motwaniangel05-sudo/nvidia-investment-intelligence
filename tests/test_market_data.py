"""Tests for market price data: ticker resolution, download, column handling."""

from unittest.mock import patch

import pandas as pd
import pytest

from data_acquisition.market_data import (
    download_price_history,
    get_tickers_to_fetch,
)

FAKE_CONFIG = {
    "company": {"ticker": "NVDA"},
    "competitors": [
        {"name": "Advanced Micro Devices", "ticker": "AMD", "reason": "GPU competitor"},
        {"name": "Intel", "ticker": "INTC", "reason": "CPU competitor"},
    ],
}


def fake_yf_dataframe():
    """Simulates what yfinance.download() returns for a single ticker."""
    idx = pd.to_datetime(["2016-01-04", "2016-01-05", "2016-01-06"])
    df = pd.DataFrame({
        "Open": [1.0, 1.1, 1.2],
        "High": [1.2, 1.3, 1.4],
        "Low": [0.9, 1.0, 1.1],
        "Close": [1.1, 1.2, 1.3],
        "Adj Close": [1.05, 1.15, 1.25],
        "Volume": [1000, 2000, 3000],
    }, index=idx)
    df.index.name = "Date"
    return df


def test_get_tickers_includes_target_and_competitors():
    tickers = get_tickers_to_fetch(FAKE_CONFIG)
    assert tickers == [
        ("NVDA", "target"),
        ("AMD", "competitor"),
        ("INTC", "competitor"),
    ]


def test_get_tickers_with_no_competitors():
    config = {"company": {"ticker": "NVDA"}, "competitors": []}
    tickers = get_tickers_to_fetch(config)
    assert tickers == [("NVDA", "target")]


def test_get_tickers_missing_competitors_key_does_not_crash():
    config = {"company": {"ticker": "NVDA"}}
    tickers = get_tickers_to_fetch(config)
    assert tickers == [("NVDA", "target")]


@patch("data_acquisition.market_data.yf.download")
def test_download_price_history_renames_columns(mock_download):
    mock_download.return_value = fake_yf_dataframe()
    df = download_price_history("NVDA", 2016, 2016)

    expected_cols = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 3


@patch("data_acquisition.market_data.yf.download")
def test_download_price_history_values_correct(mock_download):
    mock_download.return_value = fake_yf_dataframe()
    df = download_price_history("NVDA", 2016, 2016)

    row = df.iloc[0]
    assert row["close"] == 1.1
    assert row["adj_close"] == 1.05


@patch("data_acquisition.market_data.yf.download")
def test_download_price_history_empty_returns_empty_df(mock_download):
    mock_download.return_value = pd.DataFrame()
    df = download_price_history("FAKETICKER", 2016, 2020)
    assert df.empty


@patch("data_acquisition.market_data.yf.download")
def test_download_price_history_handles_multiindex_columns(mock_download):
    # Some yfinance versions return MultiIndex columns even for one ticker
    df = fake_yf_dataframe()
    df.columns = pd.MultiIndex.from_product([df.columns, ["NVDA"]])
    mock_download.return_value = df

    result = download_price_history("NVDA", 2016, 2016)
    assert "close" in result.columns
    assert len(result) == 3
