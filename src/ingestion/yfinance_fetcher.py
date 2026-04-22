"""
src/ingestion/yfinance_fetcher.py
──────────────────────────────────
Downloads OHLCV price data from Yahoo Finance.
Compatible with yfinance 1.3.0 and pandas 3.0.
Handles US stocks, Canadian stocks (.TO), ETFs, and crypto (BTC-USD).
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import settings
from src.utils.logger import log


# ─── Cache Helpers ────────────────────────────────────────────

def _cache_key(ticker: str, interval: str, period: str) -> str:
    raw = f"{ticker}_{interval}_{period}"
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{ticker}_{interval}_{period}_{h}"


def _cache_path(key: str) -> Path:
    return settings.data_cache_path / f"{key}.parquet"


def _is_cache_fresh(path: Path, max_age_hours: float = 4.0) -> bool:
    if not path.exists():
        return False
    age_hours = (
        datetime.utcnow().timestamp() - path.stat().st_mtime
    ) / 3600
    return age_hours < max_age_hours


# ─── Column Flattener ─────────────────────────────────────────

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance 1.3.0 returns MultiIndex columns like ('Close', 'AAPL').
    This flattens them to simple lowercase names like 'close'.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [
            col[0].lower() if isinstance(col, tuple) else str(col).lower()
            for col in df.columns
        ]
    return df


# ─── Data Quality Checker ─────────────────────────────────────

def _clean_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean and validate downloaded price data.
    Removes bad rows and fixes common yfinance issues.
    """
    if df.empty:
        raise ValueError(f"[{ticker}] Empty data returned by yfinance")

    original_len = len(df)

    # Flatten MultiIndex columns first
    df = _flatten_columns(df)

    # Remove rows where all price columns are zero
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if price_cols:
        zero_mask = (df[price_cols] == 0).all(axis=1)
        if zero_mask.any():
            log.warning(f"[{ticker}] Removing {zero_mask.sum()} zero-price rows")
            df = df[~zero_mask]

    # Remove duplicate timestamps
    if df.index.duplicated().any():
        log.warning(f"[{ticker}] Removing duplicate timestamps")
        df = df[~df.index.duplicated(keep="last")]

    # Forward fill single missing values
    df = df.ffill(limit=1)

    # Fix High < Low data errors
    if "high" in df.columns and "low" in df.columns:
        bad_rows = df["high"] < df["low"]
        if bad_rows.any():
            log.warning(
                f"[{ticker}] Fixing {bad_rows.sum()} rows where high < low"
            )
            df.loc[bad_rows, ["high", "low"]] = (
                df.loc[bad_rows, ["low", "high"]].values
            )

    # Add ticker column
    df["ticker"] = ticker

    # Ensure index is timezone-aware UTC
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()

    log.info(f"[{ticker}] Cleaned: {len(df)}/{original_len} rows kept")
    return df


# ─── Main Fetcher Class ───────────────────────────────────────

class YFinanceFetcher:
    """
    Downloads stock price data from Yahoo Finance.

    Features:
      - Disk cache to avoid re-downloading same data
      - Automatic retry if download fails
      - Handles US, Canadian (.TO), ETF, and crypto tickers
      - Compatible with yfinance 1.3.0 and pandas 3.0
    """

    def __init__(
        self,
        cache_max_age_hours: float = 4.0,
        max_retries: int = 3,
    ):
        self.cache_max_age_hours = cache_max_age_hours
        self.max_retries = max_retries

    # ── Single Ticker Daily ───────────────────────────────────

    def fetch_daily(
        self,
        ticker: str,
        period: str = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download daily OHLCV data for one ticker.
        Returns DataFrame with columns: open, high, low, close, volume, ticker
        Index: DatetimeIndex (UTC)
        """
        period = period or settings.yf_daily_period
        cache_file = _cache_path(_cache_key(ticker, "1d", period))

        if use_cache and _is_cache_fresh(cache_file, self.cache_max_age_hours):
            log.debug(f"[{ticker}] Loading daily from cache")
            return pd.read_parquet(cache_file)

        raw = self._download(ticker, interval="1d", period=period)
        df = _clean_ohlcv(raw, ticker)
        df.to_parquet(cache_file)
        log.info(f"[{ticker}] Daily data cached ({len(df)} rows)")
        return df

    # ── Single Ticker Intraday ────────────────────────────────

    def fetch_intraday(
        self,
        ticker: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download 1-hour intraday data for one ticker.
        Yahoo Finance allows max 60 days for hourly data.
        """
        interval  = settings.yf_intraday_interval
        period    = settings.yf_intraday_period
        cache_file = _cache_path(_cache_key(ticker, interval, period))

        if use_cache and _is_cache_fresh(cache_file, max_age_hours=0.5):
            log.debug(f"[{ticker}] Loading intraday from cache")
            return pd.read_parquet(cache_file)

        raw = self._download(ticker, interval=interval, period=period)
        df = _clean_ohlcv(raw, ticker)
        df.to_parquet(cache_file)
        log.info(f"[{ticker}] Intraday cached ({len(df)} rows)")
        return df

    # ── Multiple Tickers Batch ────────────────────────────────

    def fetch_batch(
        self,
        tickers: list[str],
        period: str = None,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Download daily data for multiple tickers.
        Returns dict of {ticker: DataFrame}.
        Fetches each ticker individually to avoid MultiIndex complexity.
        """
        period = period or settings.yf_daily_period
        results: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            try:
                results[ticker] = self.fetch_daily(
                    ticker,
                    period=period,
                    use_cache=use_cache,
                )
            except Exception as e:
                log.error(f"[{ticker}] Failed: {e}")

        success = len(results)
        failed  = len(tickers) - success
        log.info(
            f"Batch complete: {success} succeeded, {failed} failed"
        )
        return results

    # ── Internal Download with Retry ──────────────────────────

    def _download(
        self,
        ticker: str,
        interval: str,
        period: str,
    ) -> pd.DataFrame:
        """Download from Yahoo Finance with automatic retry."""
        for attempt in range(1, self.max_retries + 1):
            try:
                log.info(
                    f"[{ticker}] Downloading {interval} data "
                    f"(attempt {attempt}/{self.max_retries})"
                )
                df = yf.download(
                    tickers=ticker,
                    interval=interval,
                    period=period,
                    auto_adjust=True,
                    progress=False,
                    multi_level_index=False,
                )
                if df.empty:
                    raise ValueError("Empty response from Yahoo Finance")
                return df

            except Exception as e:
                wait = 5 * attempt
                log.warning(
                    f"[{ticker}] Attempt {attempt} failed: {e}. "
                    f"Waiting {wait}s..."
                )
                if attempt < self.max_retries:
                    time.sleep(wait)

        raise RuntimeError(
            f"[{ticker}] All {self.max_retries} download attempts failed"
        )