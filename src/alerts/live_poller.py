"""
src/alerts/live_poller.py
──────────────────────────
Polls news and signals every 15 minutes.
Sends Telegram alerts when:
  1. New high-impact news headline appears (RSS + NewsAPI)
  2. Signal direction changes (neutral → bullish, etc)
  3. Strong signal appears (confidence > threshold)

Smart features:
  - Only alerts on signal direction CHANGE (not repeated)
  - 25% confidence minimum (real signals, not noise)
  - RSS for unlimited news + NewsAPI 4x/day max
  - Per-headline cooldown to avoid spam

State tracking:
  Persists last alerts to disk to survive restarts.

Design:
  - Runs in infinite loop with 15-minute sleeps
  - Safe to kill and restart (state saved to disk)
  - Logs all alerts for audit
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import settings
from src.alerts.telegram_bot import TelegramAlerter
from src.sentiment.news_fetcher import NewsFetcher
from src.sentiment.rss_news_fetcher import RSSNewsFetcher
from src.sentiment.finbert_scorer import FinBERTScorer
from src.ensemble.signal_engine import SignalEngine
from src.position_calculator import PositionCalculator
from src.utils.logger import log

import pandas as pd


# ─── State persistence ────────────────────────────────────────

STATE_FILE = settings.data_cache_path / "alert_state.json"


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ─── Poller ───────────────────────────────────────────────────

class LivePoller:
    """
    Background poller that checks for new signals every N minutes.

    Usage:
        poller = LivePoller(capital=500, poll_minutes=15)
        poller.run_forever()
    """

    def __init__(
        self,
        capital:               float = 500.0,
        poll_minutes:          int   = 15,
        confidence_threshold:  float = 0.25,    # balanced mode
        news_alert_threshold:  float = 0.80,    # only strong news
        news_cooldown_hours:   float = 6.0,     # same headline silenced 6h
    ):
        self.capital              = capital
        self.poll_minutes         = poll_minutes
        self.confidence_threshold = confidence_threshold
        self.news_alert_threshold = news_alert_threshold
        self.news_cooldown        = timedelta(hours=news_cooldown_hours)

        self.alerter    = TelegramAlerter()
        self.news       = NewsFetcher()
        self.rss        = RSSNewsFetcher()
        self.scorer     = FinBERTScorer()
        self.engine     = SignalEngine()
        self.calculator = PositionCalculator(
            capital=capital, allow_fractional=True
        )

        # Load models once
        log.info("Loading all trained models...")
        self.engine.load_models(settings.all_tickers)
        log.info("Poller ready.")

        # State
        self.state = _load_state()

    # ── Helpers ──────────────────────────────────────────────

    def _news_key(self, ticker: str, headline: str) -> str:
        # Use first 60 chars of headline as unique key
        return f"news_{ticker}_{headline[:60]}"

    def _is_recent(self, key: str, cooldown: timedelta) -> bool:
        last_sent = self.state.get(key)
        if not last_sent:
            return False
        try:
            last_dt = datetime.fromisoformat(last_sent)
            return (datetime.now() - last_dt) < cooldown
        except Exception:
            return False

    def _mark_sent(self, key: str) -> None:
        self.state[key] = datetime.now().isoformat()
        _save_state(self.state)

    def _should_poll_newsapi(self) -> bool:
        """
        Limit NewsAPI to 4 polls per day (every 6 hours).
        Free tier = 100 calls/day. With 25 tickers,
        4 polls × 25 tickers = 100 — exactly the limit.
        """
        last = self.state.get("last_newsapi_poll")
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
            hours_since = (
                datetime.now() - last_dt
            ).total_seconds() / 3600
            return hours_since >= 6.0
        except Exception:
            return True

    # ── One polling cycle ────────────────────────────────────

    def poll_once(self) -> dict:
        """
        Run one polling cycle.
        Returns counts of alerts sent.
        """
        log.info("=" * 50)
        log.info(f"Poll cycle starting at {datetime.now()}")

        alerts_sent = {
            "signal":  0,
            "news":    0,
            "skipped": 0,
        }

        # ── Step 1: Check for signal direction changes ────
        try:
            signals = self.engine.generate_signals(
                settings.all_tickers
            )

            # Load features for position sizing
            features_dict = {}
            for ticker in settings.all_tickers:
                path = (
                    settings.data_processed_path /
                    f"{ticker}_features_with_sentiment.parquet"
                )
                if path.exists():
                    features_dict[ticker] = pd.read_parquet(path)

            # Calculate positions
            recs = self.calculator.calculate_portfolio(
                signals, features_dict
            )

            for ticker, sig in signals.items():
                # Skip neutral signals
                if sig.signal == "NEUTRAL":
                    continue

                # Skip low-confidence signals
                if sig.confidence < self.confidence_threshold:
                    continue

                # Direction-change check
                last_direction = self.state.get(
                    f"last_signal_{ticker}"
                )
                if last_direction == sig.signal:
                    # Same direction as before — skip
                    alerts_sent["skipped"] += 1
                    continue

                rec  = recs.get(ticker)
                sent = self.alerter.send_signal_alert(sig, rec)
                if sent:
                    # Track last direction
                    self.state[f"last_signal_{ticker}"] = sig.signal
                    _save_state(self.state)
                    alerts_sent["signal"] += 1
                    log.info(
                        f"[{ticker}] ALERT SENT — "
                        f"{sig.signal} ({sig.confidence:.1%})"
                    )

        except Exception as e:
            log.error(f"Signal polling failed: {e}")

        # ── Step 2: Breaking news (RSS unlimited + NewsAPI 4x/day) ──
        try:
            use_newsapi = self._should_poll_newsapi()
            if use_newsapi:
                log.info(
                    "Using NewsAPI this cycle "
                    "(6h since last call)"
                )
            else:
                log.info(
                    "Skipping NewsAPI this cycle "
                    "(rate limit conservation)"
                )

            for ticker in settings.all_tickers:
                try:
                    # ALWAYS use RSS (free, unlimited)
                    rss_articles = self.rss.fetch_ticker(
                        ticker, hours_back=24
                    )

                    # Optionally use NewsAPI (limited)
                    api_articles = []
                    if use_newsapi:
                        try:
                            api_articles = self.news.fetch_ticker(
                                ticker, days=1, use_cache=False
                            )
                        except Exception:
                            pass   # rate limit hit silently

                    # Combine + dedupe by title
                    all_articles = rss_articles + api_articles
                    seen_titles  = set()
                    unique = []
                    for a in all_articles:
                        title = a.get("title", "")[:80]
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            unique.append(a)

                    if not unique:
                        continue

                    # Score with FinBERT
                    scored = self.scorer.score_ticker(
                        ticker, unique,
                        use_cache=False,
                        filter_neutral=True,
                    )

                    # Alert on high-confidence headlines
                    for item in scored:
                        confidence = max(
                            item.get("positive", 0),
                            item.get("negative", 0),
                        )

                        if confidence < self.news_alert_threshold:
                            continue

                        headline = item.get("title", "")
                        key = self._news_key(ticker, headline)
                        if self._is_recent(key, self.news_cooldown):
                            continue

                        sent = self.alerter.send_news_alert(
                            ticker=ticker,
                            headline=headline,
                            sentiment_label=item["label"],
                            confidence=confidence,
                            source=item.get("source", ""),
                        )
                        if sent:
                            self._mark_sent(key)
                            alerts_sent["news"] += 1
                            log.info(
                                f"[{ticker}] NEWS ALERT — "
                                f"{item['label']} "
                                f"({confidence:.1%})"
                            )

                except Exception as e:
                    log.warning(
                        f"[{ticker}] news check failed: {e}"
                    )

            if use_newsapi:
                self.state["last_newsapi_poll"] = (
                    datetime.now().isoformat()
                )
                _save_state(self.state)

        except Exception as e:
            log.error(f"News polling failed: {e}")

        # Cleanup old state entries (> 48 hours)
        cutoff = datetime.now() - timedelta(hours=48)
        cleaned_state = {}
        for k, v in self.state.items():
            # Keep direction tracking and meta keys forever
            if k.startswith("last_signal_") or k == "last_newsapi_poll":
                cleaned_state[k] = v
                continue
            # Time-based cleanup for news/signal keys
            if isinstance(v, str):
                try:
                    if datetime.fromisoformat(v) > cutoff:
                        cleaned_state[k] = v
                except Exception:
                    cleaned_state[k] = v
            else:
                cleaned_state[k] = v
        self.state = cleaned_state
        _save_state(self.state)

        log.info(
            f"Poll complete: "
            f"{alerts_sent['signal']} signal alerts, "
            f"{alerts_sent['news']} news alerts, "
            f"{alerts_sent['skipped']} skipped"
        )
        return alerts_sent

    # ── Continuous polling loop ──────────────────────────────

    def run_forever(self) -> None:
        """
        Run polling loop indefinitely.
        Sleeps poll_minutes between cycles.
        """
        log.info(
            f"Starting live poller — "
            f"polling every {self.poll_minutes} minutes"
        )

        # Send startup message
        try:
            self.alerter.send_text(
                f"🚀 <b>Live Poller Started</b>\n\n"
                f"Polling every {self.poll_minutes} minutes\n"
                f"Capital: ${self.capital:,.2f}\n"
                f"Tickers: {len(settings.all_tickers)}\n"
                f"Min confidence: "
                f"{self.confidence_threshold:.0%}\n\n"
                f"<i>You'll receive alerts when signal "
                f"direction changes or breaking news appears.</i>"
            )
        except Exception as e:
            log.warning(f"Startup message failed: {e}")

        cycle = 0
        while True:
            cycle += 1
            try:
                self.poll_once()
            except KeyboardInterrupt:
                log.info("Poller stopped by user")
                break
            except Exception as e:
                log.error(f"Cycle {cycle} crashed: {e}")

            log.info(
                f"Sleeping {self.poll_minutes} minutes "
                f"until next cycle..."
            )
            time.sleep(self.poll_minutes * 60)


if __name__ == "__main__":
    poller = LivePoller(capital=500.0, poll_minutes=15)
    poller.run_forever()