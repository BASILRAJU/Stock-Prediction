"""
src/sentiment/news_fetcher.py
──────────────────────────────
Fetches financial news headlines from NewsAPI for each ticker.

Free tier: 100 requests/day, articles from last 30 days.

For each ticker we search:
  - The company name (e.g. "Apple")
  - The ticker symbol (e.g. "AAPL")
  - Financial keywords combined (e.g. "Apple stock earnings")

Headlines are cached to disk so we don't waste daily quota
re-fetching the same articles.

Research basis:
  FinBERT study (2025): sentiment + time-series ensemble hits ~80%
  accuracy vs ~70% for standalone models.
  Critical: neutral headlines must be filtered out.
  Look-ahead bias prevention: timestamps strictly aligned.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from config.settings import settings
from src.utils.logger import log


# ─── Company name mapping ─────────────────────────────────────
# NewsAPI searches by keywords — ticker symbols alone miss many
# relevant articles. We search by company name for better coverage.

TICKER_TO_COMPANY = {
    "AAPL":    "Apple",
    "MSFT":    "Microsoft",
    "NVDA":    "Nvidia",
    "TSLA":    "Tesla",
    "SPY":     "S&P 500",
    "CVX":     "Chevron",
    "AMD":     "AMD semiconductor",
    "GLD":     "gold price",
    "BTC-USD": "Bitcoin",
    "DOL.TO":  "Dollarama",
    "VFV.TO":  "Vanguard S&P 500",
    "CSU.TO":  "Constellation Software",
}

NEWSAPI_BASE = "https://newsapi.org/v2/everything"


# ─── Cache helpers ────────────────────────────────────────────

def _news_cache_path(ticker: str, date_str: str) -> Path:
    """One cache file per ticker per day."""
    safe = ticker.replace("-", "_").replace(".", "_")
    return settings.data_cache_path / f"news_{safe}_{date_str}.json"


def _is_cache_fresh(path: Path, max_age_hours: float = 6.0) -> bool:
    if not path.exists():
        return False
    age = (datetime.utcnow().timestamp() - path.stat().st_mtime) / 3600
    return age < max_age_hours


# ─── Main Fetcher ─────────────────────────────────────────────

class NewsFetcher:
    """
    Fetches financial news headlines from NewsAPI.

    Usage:
        fetcher = NewsFetcher()
        headlines = fetcher.fetch_ticker("AAPL", days=7)
        all_headlines = fetcher.fetch_all_tickers(days=7)
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.news_api_key
        self._last_request: float = 0.0

        if not self.api_key:
            raise ValueError(
                "NewsAPI key not set. "
                "Add NEWS_API_KEY=your_key to your .env file."
            )

    def fetch_ticker(
        self,
        ticker: str,
        days: int = 7,
        max_articles: int = 50,
        use_cache: bool = True,
    ) -> list[dict]:
        """
        Fetch news headlines for one ticker.

        Args:
            ticker:       Ticker symbol e.g. "AAPL"
            days:         How many days back to search (max 30 free tier)
            max_articles: Maximum headlines to return
            use_cache:    Use cached results if available

        Returns:
            List of article dicts with keys:
            title, description, publishedAt, source, url
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        cache_file = _news_cache_path(ticker, today)

        if use_cache and _is_cache_fresh(cache_file):
            log.debug(f"[{ticker}] News: loaded from cache")
            with open(cache_file) as f:
                return json.load(f)

        # Build search query
        company = TICKER_TO_COMPANY.get(ticker, ticker)
        query = f'"{company}" stock OR earnings OR forecast OR analyst'

        # Date range
        from_date = (
            datetime.utcnow() - timedelta(days=days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Rate limiting — NewsAPI free tier is generous but be safe
        elapsed = time.time() - self._last_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        log.info(f"[{ticker}] Fetching news: '{company}' last {days} days")

        try:
            self._last_request = time.time()
            resp = requests.get(
                NEWSAPI_BASE,
                params={
                    "q":          query,
                    "from":       from_date,
                    "language":   "en",
                    "sortBy":     "relevancy",
                    "pageSize":   min(max_articles, 100),
                    "apiKey":     self.api_key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "ok":
                log.warning(
                    f"[{ticker}] NewsAPI error: {data.get('message', 'unknown')}"
                )
                return []

            articles = data.get("articles", [])

            # Clean and filter articles
            cleaned = []
            for a in articles:
                title = a.get("title", "") or ""
                desc  = a.get("description", "") or ""

                # Skip articles with no useful text
                if not title or title == "[Removed]":
                    continue

                cleaned.append({
                    "ticker":      ticker,
                    "title":       title.strip(),
                    "description": desc.strip(),
                    "published_at": a.get("publishedAt", ""),
                    "source":      a.get("source", {}).get("name", ""),
                    "url":         a.get("url", ""),
                })

            # Cache results
            with open(cache_file, "w") as f:
                json.dump(cleaned, f, indent=2)

            log.info(
                f"[{ticker}] News: {len(cleaned)} headlines fetched and cached"
            )
            return cleaned

        except requests.RequestException as e:
            log.error(f"[{ticker}] News fetch failed: {e}")
            return []

    def fetch_all_tickers(
        self,
        tickers: list[str] = None,
        days: int = 7,
        use_cache: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Fetch news for all tickers.
        Returns dict of {ticker: [articles]}
        """
        tickers = tickers or settings.all_tickers
        results: dict[str, list[dict]] = {}

        for ticker in tickers:
            try:
                articles = self.fetch_ticker(
                    ticker, days=days, use_cache=use_cache
                )
                results[ticker] = articles
            except Exception as e:
                log.error(f"[{ticker}] News fetch error: {e}")
                results[ticker] = []

        total = sum(len(v) for v in results.values())
        log.info(
            f"News fetch complete: {total} total headlines "
            f"across {len(results)} tickers"
        )
        return results