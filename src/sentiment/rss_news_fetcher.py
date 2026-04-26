"""
src/sentiment/rss_news_fetcher.py
──────────────────────────────────
Fetches stock news from FREE RSS feeds — no API limits.

Sources:
  - Yahoo Finance ticker-specific RSS
  - Reuters business news
  - MarketWatch ticker pages
  - Seeking Alpha ticker symbols

Why RSS over NewsAPI:
  - Unlimited requests (no 100/day cap)
  - Real-time updates
  - Direct from source (less aggregator noise)
  - Yahoo Finance has per-ticker feeds

Usage:
    fetcher = RSSNewsFetcher()
    articles = fetcher.fetch_ticker('AAPL', hours_back=24)
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from xml.etree import ElementTree as ET

import requests

from src.utils.logger import log


# Yahoo Finance ticker RSS template
YAHOO_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

# General market news RSS (no ticker required)
MARKET_FEEDS = [
    "https://www.reuters.com/business/finance/rss",
    "https://www.cnbc.com/id/15839135/device/rss/rss.html",  # markets
    "https://feeds.marketwatch.com/marketwatch/topstories",
]


class RSSNewsFetcher:
    """Free RSS news fetcher — no API key, no rate limits."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Stock Signal Bot)",
        })

    def fetch_ticker(
        self,
        ticker: str,
        hours_back: int = 24,
    ) -> list[dict]:
        """
        Fetch news for a specific ticker via Yahoo Finance RSS.

        Yahoo's feed handles US stocks well (AAPL, MSFT etc).
        For Canadian stocks (.TO), strip the suffix.

        Returns:
            List of articles with: title, description, link,
            published_at, source
        """
        # Clean ticker for Yahoo (e.g., DOL.TO → DOL)
        clean_ticker = ticker.replace(".TO", "").replace("-USD", "-USD")

        url = YAHOO_FEED.format(ticker=clean_ticker)

        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                log.warning(
                    f"[{ticker}] Yahoo RSS returned {resp.status_code}"
                )
                return []

            articles = self._parse_rss(resp.text)
            articles = self._filter_recent(articles, hours_back)

            for a in articles:
                a["source"]  = "Yahoo Finance"
                a["ticker"]  = ticker

            log.info(
                f"[{ticker}] RSS: {len(articles)} headlines "
                f"in last {hours_back}h"
            )
            return articles

        except Exception as e:
            log.warning(f"[{ticker}] RSS fetch failed: {e}")
            return []

    def fetch_market_news(self, hours_back: int = 24) -> list[dict]:
        """
        Fetch general market news from multiple sources.
        Use this to detect macro events (Fed, economy, war, etc.)
        """
        all_articles = []

        for url in MARKET_FEEDS:
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code != 200:
                    continue

                articles = self._parse_rss(resp.text)
                articles = self._filter_recent(articles, hours_back)
                source   = self._extract_source(url)

                for a in articles:
                    a["source"] = source
                    a["ticker"] = "MARKET"

                all_articles.extend(articles)

            except Exception as e:
                log.debug(f"Market feed {url} failed: {e}")
                continue

        log.info(
            f"Market RSS: {len(all_articles)} headlines "
            f"in last {hours_back}h"
        )
        return all_articles

    def _parse_rss(self, xml_text: str) -> list[dict]:
        """Parse RSS XML into article dicts."""
        articles = []

        try:
            root = ET.fromstring(xml_text)

            # RSS structure: rss > channel > item[]
            for item in root.iter("item"):
                title       = self._get_text(item, "title")
                description = self._get_text(item, "description")
                link        = self._get_text(item, "link")
                pub_date    = self._get_text(item, "pubDate")

                if not title:
                    continue

                # Parse pub date
                published_at = self._parse_date(pub_date)

                # Strip HTML from description
                if description:
                    description = re.sub(r"<[^>]+>", "", description)
                    description = description.strip()[:500]

                articles.append({
                    "title":        title.strip(),
                    "description":  description or "",
                    "link":         link or "",
                    "published_at": published_at,
                })

        except ET.ParseError as e:
            log.warning(f"RSS parse error: {e}")

        return articles

    def _get_text(self, element, tag: str) -> Optional[str]:
        """Safely extract text from XML element."""
        child = element.find(tag)
        return child.text if child is not None else None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse RSS pub date (RFC 822 format)."""
        if not date_str:
            return None
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            return dt
        except Exception:
            return None

    def _filter_recent(
        self,
        articles: list[dict],
        hours_back: int,
    ) -> list[dict]:
        """Keep only articles within hours_back."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        recent = []
        for a in articles:
            pub = a.get("published_at")
            if pub is None:
                # No date — assume recent
                recent.append(a)
                continue
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub >= cutoff:
                recent.append(a)

        return recent

    def _extract_source(self, url: str) -> str:
        """Get source name from URL."""
        if "reuters" in url:
            return "Reuters"
        if "cnbc" in url:
            return "CNBC"
        if "marketwatch" in url:
            return "MarketWatch"
        if "yahoo" in url:
            return "Yahoo Finance"
        return "RSS"