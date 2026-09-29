"""
Download historical daily stock price data for the target company and its
configured competitors, using yfinance (free, no API key).

Both raw Close and Adjusted Close are stored. Adjusted Close should be used
for any historical trend/comparison, since it accounts for stock splits and
dividends; raw Close is kept for reference/transparency only.
"""

from datetime import date
from typing import List, Tuple

import pandas as pd
import yfinance as yf

from core.config_loader import get_path, load_config
from core.logger import get_logger
from data_acquisition.manifest import Manifest

log = get_logger(__name__)


def get_tickers_to_fetch(config: dict) -> List[Tuple[str, str]]:
    """
    Returns a list of (ticker, role) tuples: the target company plus all
    configured competitors. Company-agnostic: reads entirely from config.
    """
    tickers = [(config["company"]["ticker"], "target")]
    for comp in config.get("competitors", []):
        tickers.append((comp["ticker"], "competitor"))
    return tickers


def download_price_history(ticker: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Download daily OHLCV + adjusted close for one ticker via yfinance.
    Returns an empty DataFrame (not an exception) if the ticker has no data,
    so the caller can decide how to handle it -- we never fabricate prices.
    """
    start = f"{start_year}-01-01"
    end = f"{min(end_year, date.today().year) + 1}-01-01"  # yfinance end is exclusive

    log.info("Fetching price history for %s (%s to %s)", ticker, start, end)
    df = yf.download(
        ticker, start=start, end=end,
        auto_adjust=False,  # keep both raw Close and Adj Close separate
        progress=False,
    )

    if df.empty:
        log.warning("No price data returned for %s", ticker)
        return pd.DataFrame()

    # yfinance can return MultiIndex columns when given a single ticker
    # in some versions; flatten defensively.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })
    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    return df


def build_market_data() -> dict:
    """
    Main entry: fetch prices for target company + all competitors, save
    each to its own CSV, record in manifest. Returns a summary dict.
    """
    config = load_config()
    period = config["data_period"]
    manifest = Manifest()

    out_dir = get_path(config, "data_market")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for ticker, role in get_tickers_to_fetch(config):
        df = download_price_history(ticker, period["start_year"], period["end_year"])

        if df.empty:
            summary[ticker] = {"role": role, "rows": 0, "status": "no_data"}
            continue

        out_file = out_dir / f"prices_{ticker}.csv"
        df.to_csv(out_file, index=False)
        manifest.add(
            out_file,
            source_url=f"https://finance.yahoo.com/quote/{ticker}/history",
            source_name="YAHOO_FINANCE",
            doc_type="prices",
            company_ticker=ticker,
        )
        summary[ticker] = {
            "role": role,
            "rows": len(df),
            "status": "downloaded",
            "first_date": str(df["date"].min().date()),
            "last_date": str(df["date"].max().date()),
        }
        log.info("Saved prices: %s (%d rows, %s to %s)",
                  out_file, len(df), summary[ticker]["first_date"],
                  summary[ticker]["last_date"])

    return summary


if __name__ == "__main__":
    result = build_market_data()
    print(pd.DataFrame(result).T.to_string())
