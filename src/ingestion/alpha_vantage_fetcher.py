"""
src/ingestion/alpha_vantage_fetcher.py
───────────────────────────────────────
Fetches macro economic data and company fundamentals
from Alpha Vantage API.

Free tier limits: 25 requests per day, 5 per minute.
The 12.5 second delay between requests keeps us within limits.

What this fetches:
  - Macro indicators: GDP, CPI, Fed Rate, Unemployment, Treasury Yield
  - Company fundamentals: P/E ratio, Beta, Market Cap, etc.
  - Earnings history: EPS actuals vs estimates
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config.settings import settings
from src.utils.logger import log
from src.utils.models import FundamentalSnapshot


# ─── Alpha Vantage Base URL ───────────────────────────────────

AV_BASE = "https://www.alphavantage.co/query"

# ─── Macro Endpoint Definitions ───────────────────────────────

MACRO_ENDPOINTS = {
    "REAL_GDP": {
        "function": "REAL_GDP",
        "interval": "quarterly",
    },
    "CPI": {
        "function": "CPI",
        "interval": "monthly",
    },
    "FEDERAL_FUNDS_RATE": {
        "function": "FEDERAL_FUNDS_RATE",
        "interval": "daily",
    },
    "UNEMPLOYMENT": {
        "function": "UNEMPLOYMENT",
    },
    "TREASURY_YIELD": {
        "function": "TREASURY_YIELD",
        "interval": "daily",
        "maturity": "10year",
    },
}


# ─── Cache Helpers ────────────────────────────────────────────

def _av_cache_path(name: str, suffix: str = ".parquet") -> Path:
    return settings.data_cache_path / f"av_{name}{suffix}"


def _is_cache_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    age = (
        datetime.utcnow().timestamp() - path.stat().st_mtime
    ) / 3600
    return age < max_age_hours


# ─── Main Fetcher Class ───────────────────────────────────────

class AlphaVantageFetcher:
    """
    Fetches macro and fundamental data from Alpha Vantage.

    All responses are cached to disk so we don't waste our
    25 free daily requests re-downloading the same data.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.alpha_vantage_api_key
        self._last_request_time: float = 0.0

        if self.api_key == "demo":
            log.warning(
                "Using Alpha Vantage demo key — limited data available. "
                "Add your real key to the .env file."
            )

    # ── Rate Limited GET Request ──────────────────────────────

    def _get(self, params: dict[str, Any]) -> dict:
        """
        Make a rate-limited GET request to Alpha Vantage.
        Enforces minimum gap between requests to stay within
        the free tier limit of 5 requests per minute.
        """
        params["apikey"] = self.api_key

        for attempt in range(1, settings.av_max_retries + 1):

            # Enforce rate limit
            elapsed = time.time() - self._last_request_time
            if elapsed < settings.av_request_delay_s:
                sleep_for = settings.av_request_delay_s - elapsed
                log.debug(f"Rate limiter: waiting {sleep_for:.1f}s")
                time.sleep(sleep_for)

            try:
                self._last_request_time = time.time()
                resp = requests.get(AV_BASE, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                # Alpha Vantage puts errors in the JSON body
                if "Error Message" in data:
                    raise ValueError(
                        f"Alpha Vantage error: {data['Error Message']}"
                    )

                # Rate limit warning from Alpha Vantage
                if "Note" in data:
                    log.warning(
                        f"Alpha Vantage rate limit hit. "
                        f"Waiting 60s before retry..."
                    )
                    time.sleep(60)
                    continue

                return data

            except requests.RequestException as e:
                wait = settings.av_request_delay_s * (2 ** attempt)
                log.warning(
                    f"Request failed (attempt {attempt}): {e}. "
                    f"Waiting {wait:.0f}s"
                )
                if attempt < settings.av_max_retries:
                    time.sleep(wait)

        raise RuntimeError(
            f"All {settings.av_max_retries} Alpha Vantage "
            f"attempts failed for {params.get('function')}"
        )

    # ── Macro Indicators ──────────────────────────────────────

    def fetch_macro_indicator(
        self,
        indicator: str,
        use_cache: bool = True,
        cache_hours: float = 24.0,
    ) -> pd.DataFrame:
        """
        Fetch one macro indicator time series.
        Returns DataFrame with columns: date, value, indicator

        Available indicators:
          REAL_GDP, CPI, FEDERAL_FUNDS_RATE,
          UNEMPLOYMENT, TREASURY_YIELD
        """
        cache_file = _av_cache_path(f"macro_{indicator}")

        if use_cache and _is_cache_fresh(cache_file, cache_hours):
            log.debug(f"Macro {indicator}: loaded from cache")
            return pd.read_parquet(cache_file)

        if indicator not in MACRO_ENDPOINTS:
            raise ValueError(
                f"Unknown indicator '{indicator}'. "
                f"Choose from: {list(MACRO_ENDPOINTS.keys())}"
            )

        log.info(f"Fetching macro indicator: {indicator}")
        params = MACRO_ENDPOINTS[indicator].copy()
        data = self._get(params)

        records = data.get("data", [])
        if not records:
            log.warning(f"No data returned for {indicator}")
            return pd.DataFrame(columns=["date", "value", "indicator"])

        df = pd.DataFrame(records)
        df.columns = df.columns.str.lower()
        df["date"]      = pd.to_datetime(df["date"])
        df["value"]     = pd.to_numeric(df["value"], errors="coerce")
        df["indicator"] = indicator
        df = df.dropna(subset=["value"])
        df = df.sort_values("date").reset_index(drop=True)

        df.to_parquet(cache_file, index=False)
        log.info(f"Macro {indicator}: {len(df)} records cached")
        return df

    def fetch_all_macro(
        self,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch all macro indicators defined in settings.
        Returns dict of {indicator_name: DataFrame}.
        """
        results: dict[str, pd.DataFrame] = {}

        for indicator in settings.macro_indicators:
            try:
                results[indicator] = self.fetch_macro_indicator(
                    indicator, use_cache=use_cache
                )
            except Exception as e:
                log.error(f"Failed to fetch {indicator}: {e}")

        log.info(
            f"Macro fetch complete: "
            f"{len(results)}/{len(settings.macro_indicators)} successful"
        )
        return results

    def get_macro_wide(
        self,
        macro_dict: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Merge all macro series into one wide DataFrame.
        Resampled to month-end frequency.
        Forward-filled so every date has a value.

        This is the format XGBoost and the regime detector consume.
        """
        frames = []
        for name, df in macro_dict.items():
            if df.empty:
                continue
            s = df.set_index("date")["value"].rename(name)
            frames.append(s)

        if not frames:
            return pd.DataFrame()

        wide = pd.concat(frames, axis=1)
        wide.index = pd.DatetimeIndex(wide.index)
        wide = wide.resample("ME").last()
        wide = wide.ffill()
        return wide

    # ── Company Fundamentals ──────────────────────────────────

    def fetch_fundamentals(
        self,
        ticker: str,
        use_cache: bool = True,
        cache_hours: float = 48.0,
    ) -> FundamentalSnapshot:
        """
        Fetch company overview for one ticker.
        Returns a validated FundamentalSnapshot object.

        Note: Canadian tickers (.TO) are not supported by
        Alpha Vantage free tier — US tickers only.
        """
        # Skip Canadian tickers — not supported by AV free tier
        if ticker.endswith(".TO"):
            log.info(
                f"[{ticker}] Skipping fundamentals "
                f"(Canadian tickers not supported by AV free tier)"
            )
            return FundamentalSnapshot(
                ticker=ticker,
                fetched_at=datetime.utcnow(),
            )

        cache_file = _av_cache_path(
            f"fundamentals_{ticker}", ".json"
        )

        if use_cache and _is_cache_fresh(cache_file, cache_hours):
            log.debug(f"[{ticker}] Fundamentals: loaded from cache")
            with open(cache_file) as f:
                raw = json.load(f)
            return FundamentalSnapshot(**raw)

        log.info(f"[{ticker}] Fetching fundamentals from Alpha Vantage")
        data = self._get({
            "function": "OVERVIEW",
            "symbol": ticker,
        })

        def _float(key: str) -> float | None:
            try:
                return float(data.get(key, "None"))
            except (TypeError, ValueError):
                return None

        snap = FundamentalSnapshot(
            ticker=ticker,
            fetched_at=datetime.utcnow(),
            market_cap=_float("MarketCapitalization"),
            pe_ratio=_float("PERatio"),
            pb_ratio=_float("PriceToBookRatio"),
            eps=_float("EPS"),
            profit_margin=_float("ProfitMargin"),
            dividend_yield=_float("DividendYield"),
            beta=_float("Beta"),
            week_52_high=_float("52WeekHigh"),
            week_52_low=_float("52WeekLow"),
            analyst_target_price=_float("AnalystTargetPrice"),
            sector=data.get("Sector", ""),
            industry=data.get("Industry", ""),
        )

        with open(cache_file, "w") as f:
            json.dump(
                snap.model_dump(mode="json"),
                f, indent=2, default=str
            )

        log.info(
            f"[{ticker}] Fundamentals cached: "
            f"P/E={snap.pe_ratio}, Beta={snap.beta}, "
            f"Sector={snap.sector}"
        )
        return snap

    # ── Earnings History ──────────────────────────────────────

    def fetch_earnings(
        self,
        ticker: str,
        use_cache: bool = True,
        cache_hours: float = 24.0,
    ) -> pd.DataFrame:
        """
        Fetch quarterly earnings history.
        Returns DataFrame with: fiscal_date, reported_eps,
        estimated_eps, surprise, surprise_pct
        """
        if ticker.endswith(".TO"):
            log.info(f"[{ticker}] Skipping earnings (Canadian ticker)")
            return pd.DataFrame()

        cache_file = _av_cache_path(f"earnings_{ticker}")

        if use_cache and _is_cache_fresh(cache_file, cache_hours):
            log.debug(f"[{ticker}] Earnings: loaded from cache")
            return pd.read_parquet(cache_file)

        log.info(f"[{ticker}] Fetching earnings from Alpha Vantage")
        data = self._get({
            "function": "EARNINGS",
            "symbol": ticker,
        })

        quarterly = data.get("quarterlyEarnings", [])
        if not quarterly:
            return pd.DataFrame()

        df = pd.DataFrame(quarterly)
        df.columns = df.columns.str.lower()

        rename = {
            "fiscaldateending":    "fiscal_date",
            "reporteddate":        "report_date",
            "reportedeps":         "reported_eps",
            "estimatedeps":        "estimated_eps",
            "surprise":            "surprise",
            "surprisepercentage":  "surprise_pct",
        }
        df = df.rename(columns=rename)

        for col in ["reported_eps", "estimated_eps",
                    "surprise", "surprise_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ["fiscal_date", "report_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        df["ticker"] = ticker
        df = df.sort_values("fiscal_date").reset_index(drop=True)
        df.to_parquet(cache_file, index=False)

        log.info(f"[{ticker}] Earnings: {len(df)} quarters cached")
        return df