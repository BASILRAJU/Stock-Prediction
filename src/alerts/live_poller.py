"""
src/alerts/live_poller.py
──────────────────────────
Polls news and signals every 15 minutes.
Sends Telegram alerts when:
  1. New high-impact news headline appears
  2. Signal changes direction (neutral → bullish, etc)
  3. Strong signal appears (confidence > threshold)

State tracking:
  We persist the last alerts sent to avoid spam.
  Same signal won't alert twice within a cooldown window.

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
        capital:              float = 500.0,
        poll_minutes:         int   = 15,
        confidence_threshold: float = 0.08,
        news_alert_threshold: float = 0.75,
        signal_cooldown_hours: float = 4.0,
        news_cooldown_hours:   float = 1.0,
    ):
        self.capital              = capital
        self.poll_minutes         = poll_minutes
        self.confidence_threshold = confidence_threshold
        self.news_alert_threshold = news_alert_threshold
        self.signal_cooldown      = timedelta(hours=signal_cooldown_hours)
        self.news_cooldown        = timedelta(hours=news_cooldown_hours)

        self.alerter     = TelegramAlerter()
        self.news        = NewsFetcher()
        self.scorer      = FinBERTScorer()
        self.engine      = SignalEngine()
        self.calculator  = PositionCalculator(
            capital=capital, allow_fractional=True
        )

        # Load models once
        log.info("Loading all trained models...")
        self.engine.load_models(settings.all_tickers)
        log.info("Poller ready.")

        # State
        self.state = _load_state()

    # ── Helpers ──────────────────────────────────────────────

    def _signal_key(self, ticker: str, signal: str) -> str:
        return f"signal_{ticker}_{signal}"

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

        # ── Step 1: Check for signal changes ──────────────
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

                # Cooldown check
                key = self._signal_key(ticker, sig.signal)
                if self._is_recent(key, self.signal_cooldown):
                    alerts_sent["skipped"] += 1
                    continue

                rec = recs.get(ticker)
                sent = self.alerter.send_signal_alert(sig, rec)
                if sent:
                    self._mark_sent(key)
                    alerts_sent["signal"] += 1
                    log.info(
                        f"[{ticker}] ALERT SENT — "
                        f"{sig.signal} ({sig.confidence:.1%})"
                    )

        except Exception as e:
            log.error(f"Signal polling failed: {e}")

        # ── Step 2: Check for breaking news ───────────────
        try:
            for ticker in settings.all_tickers:
                try:
                    # Fetch latest news (uses cache so this is fast)
                    articles = self.news.fetch_ticker(
                        ticker, days=1, use_cache=False
                    )
                    if not articles:
                        continue

                    # Score all headlines
                    scored = self.scorer.score_ticker(
                        ticker, articles,
                        use_cache=False,
                        filter_neutral=True,
                    )

                    # Alert on high-confidence non-neutral headlines
                    for item in scored:
                        confidence = max(
                            item.get("positive", 0),
                            item.get("negative", 0),
                        )

                        # Must be high confidence
                        if confidence < self.news_alert_threshold:
                            continue

                        # Cooldown check
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
                                f"{item['label']} ({confidence:.1%})"
                            )

                except Exception as e:
                    log.warning(f"[{ticker}] news check failed: {e}")

        except Exception as e:
            log.error(f"News polling failed: {e}")

        # Cleanup old state entries (> 48 hours)
        cutoff = datetime.now() - timedelta(hours=48)
        self.state = {
            k: v for k, v in self.state.items()
            if isinstance(v, str)
            and datetime.fromisoformat(v) > cutoff
        }
        _save_state(self.state)

        log.info(
            f"Poll complete: "
            f"{alerts_sent['signal']} signal alerts, "
            f"{alerts_sent['news']} news alerts, "
            f"{alerts_sent['skipped']} skipped (cooldown)"
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
                f"Tickers: {len(settings.all_tickers)}\n\n"
                f"<i>You'll receive alerts when signals or "
                f"high-confidence news appears.</i>"
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