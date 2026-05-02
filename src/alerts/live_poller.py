"""
src/alerts/live_poller.py
──────────────────────────
Polls news and signals every 15 minutes.
Sends Telegram alerts with annotated charts when:
  1. New high-impact news headline appears (RSS + NewsAPI)
  2. Signal direction changes (neutral → bullish, etc)
  3. Strong signal appears (confidence > threshold)

Smart features:
  - Annotated charts with S/R, POC, MAs, entry/stop/target
  - Only alerts on signal direction CHANGE (not repeated)
  - 25% confidence minimum (real signals, not noise)
  - RSS for unlimited news + NewsAPI 4x/day max
  - Per-headline cooldown to avoid spam

State tracking:
  Persists last alerts to disk to survive restarts.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config.settings import settings
from src.alerts.telegram_bot import TelegramAlerter
from src.alerts.chart_annotator import generate_alert_chart
from src.sentiment.news_fetcher import NewsFetcher
from src.sentiment.rss_news_fetcher import RSSNewsFetcher
from src.sentiment.finbert_scorer import FinBERTScorer
from src.ensemble.signal_engine import SignalEngine
from src.position_calculator import PositionCalculator
from src.ingestion.yfinance_fetcher import YFinanceFetcher
from src.utils.logger import log


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
    Background poller with annotated chart alerts.
    """

    def __init__(
        self,
        capital:               float = 500.0,
        poll_minutes:          int   = 15,
        confidence_threshold:  float = 0.25,
        news_alert_threshold:  float = 0.80,
        news_cooldown_hours:   float = 6.0,
        send_charts:           bool  = True,
    ):
        self.capital              = capital
        self.poll_minutes         = poll_minutes
        self.confidence_threshold = confidence_threshold
        self.news_alert_threshold = news_alert_threshold
        self.news_cooldown        = timedelta(hours=news_cooldown_hours)
        self.send_charts          = send_charts

        self.alerter    = TelegramAlerter()
        self.news       = NewsFetcher()
        self.rss        = RSSNewsFetcher()
        self.scorer     = FinBERTScorer()
        self.engine     = SignalEngine()
        self.calculator = PositionCalculator(
            capital=capital, allow_fractional=True
        )
        self.fetcher    = YFinanceFetcher()

        log.info("Loading all trained models...")
        self.engine.load_models(settings.all_tickers)
        log.info("Poller ready.")

        self.state = _load_state()

    # ── Helpers ──────────────────────────────────────────────

    def _news_key(self, ticker: str, headline: str) -> str:
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

    def _detect_patterns_for_ticker(self, ticker: str) -> list:
        """
        Get currently-detected patterns for the latest day.
        Reads from feature parquet to find active candlestick flags.
        """
        try:
            path = (
                settings.data_processed_path /
                f"{ticker}_features_with_sentiment.parquet"
            )
            if not path.exists():
                return []

            df = pd.read_parquet(path)
            if df.empty:
                return []

            latest = df.iloc[-1]

            # Pattern columns to check
            pattern_map = {
                "cdl_engulfing_bull":  "bullish_engulfing",
                "cdl_engulfing_bear":  "bearish_engulfing",
                "cdl_hammer":          "hammer",
                "cdl_shooting_star":   "shooting_star",
                "cdl_three_soldiers":  "three_white_soldiers",
                "cdl_three_crows":     "three_black_crows",
                "cdl_morning_star":    "morning_star",
                "cdl_evening_star":    "evening_star",
                "cdl_doji":            "doji",
                "lq_swept_high":       "liquidity_sweep_high",
                "lq_swept_low":        "liquidity_sweep_low",
                "lq_sweep_reversal":   "sweep_reversal",
                "lq_bullish_ob":       "bullish_order_block",
                "lq_bearish_ob":       "bearish_order_block",
            }

            active = []
            for col, name in pattern_map.items():
                val = latest.get(col, 0)
                if val and val != 0 and not pd.isna(val):
                    active.append(name)

            return active[:4]  # max 4 to keep title clean

        except Exception as e:
            log.warning(f"[{ticker}] Pattern detection failed: {e}")
            return []

    def _generate_chart_for_alert(
        self,
        ticker:         str,
        signal,
        recommendation,
    ) -> bytes:
        """Generate annotated chart image for the alert."""
        try:
            # Get raw OHLCV
            raw_df = self.fetcher.fetch_daily(
                ticker, use_cache=True
            )
            if raw_df.empty:
                return None

            # Get current trade levels
            entry  = (
                recommendation.entry_price
                if recommendation else float(raw_df["close"].iloc[-1])
            )
            stop   = (
                recommendation.stop_loss
                if recommendation else None
            )
            tgt1   = (
                recommendation.target_1
                if recommendation else None
            )
            tgt2   = (
                recommendation.target_2
                if recommendation else None
            )

            patterns = self._detect_patterns_for_ticker(ticker)

            chart_bytes = generate_alert_chart(
                ticker=ticker,
                df=raw_df,
                signal=signal.signal,
                confidence=signal.confidence,
                entry_price=entry,
                stop_loss=stop,
                target_1=tgt1,
                target_2=tgt2,
                detected_patterns=patterns,
                window_days=30,
            )
            return chart_bytes

        except Exception as e:
            log.error(f"[{ticker}] Chart for alert failed: {e}")
            return None

    # ── One polling cycle ────────────────────────────────────

    def poll_once(self) -> dict:
        log.info("=" * 50)
        log.info(f"Poll cycle starting at {datetime.now()}")

        alerts_sent = {
            "signal":  0,
            "news":    0,
            "skipped": 0,
        }

        # ── Step 1: Signal direction changes ────────────────
        try:
            signals = self.engine.generate_signals(
                settings.all_tickers
            )

            features_dict = {}
            for ticker in settings.all_tickers:
                path = (
                    settings.data_processed_path /
                    f"{ticker}_features_with_sentiment.parquet"
                )
                if path.exists():
                    features_dict[ticker] = pd.read_parquet(path)

            recs = self.calculator.calculate_portfolio(
                signals, features_dict
            )

            for ticker, sig in signals.items():
                if sig.signal == "NEUTRAL":
                    continue

                if sig.confidence < self.confidence_threshold:
                    continue

                last_direction = self.state.get(
                    f"last_signal_{ticker}"
                )
                if last_direction == sig.signal:
                    alerts_sent["skipped"] += 1
                    continue

                rec = recs.get(ticker)

                # Generate annotated chart
                chart_bytes = None
                if self.send_charts:
                    chart_bytes = self._generate_chart_for_alert(
                        ticker, sig, rec
                    )

                sent = self.alerter.send_signal_alert(
                    sig, rec, chart_image=chart_bytes,
                )
                if sent:
                    self.state[f"last_signal_{ticker}"] = sig.signal
                    _save_state(self.state)
                    alerts_sent["signal"] += 1
                    log.info(
                        f"[{ticker}] ALERT SENT — "
                        f"{sig.signal} ({sig.confidence:.1%}) "
                        f"{'[with chart]' if chart_bytes else '[text only]'}"
                    )

        except Exception as e:
            log.error(f"Signal polling failed: {e}")

        # ── Step 2: Breaking news ───────────────────────────
        try:
            use_newsapi = self._should_poll_newsapi()
            if use_newsapi:
                log.info("Using NewsAPI this cycle (6h since last)")
            else:
                log.info("Skipping NewsAPI (rate limit conservation)")

            for ticker in settings.all_tickers:
                try:
                    rss_articles = self.rss.fetch_ticker(
                        ticker, hours_back=24
                    )

                    api_articles = []
                    if use_newsapi:
                        try:
                            api_articles = self.news.fetch_ticker(
                                ticker, days=1, use_cache=False
                            )
                        except Exception:
                            pass

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

                    scored = self.scorer.score_ticker(
                        ticker, unique,
                        use_cache=False,
                        filter_neutral=True,
                    )

                    bullish_count = sum(
                        1 for item in scored
                        if item.get("label") == "positive"
                        and item.get("positive", 0) >= self.news_alert_threshold
                    )
                    bearish_count = sum(
                        1 for item in scored
                        if item.get("label") == "negative"
                        and item.get("negative", 0) >= self.news_alert_threshold
                    )

                    total_strong = bullish_count + bearish_count
                    if total_strong < 3:
                        continue

                    bull_ratio = (
                        bullish_count / total_strong
                        if total_strong else 0
                    )
                    bear_ratio = (
                        bearish_count / total_strong
                        if total_strong else 0
                    )

                    if bull_ratio >= 0.65:
                        direction = "BULLISH"
                        sample_label = "positive"
                    elif bear_ratio >= 0.65:
                        direction = "BEARISH"
                        sample_label = "negative"
                    else:
                        continue

                    key = f"news_summary_{ticker}_{direction}"
                    if self._is_recent(key, self.news_cooldown):
                        continue

                    representative = max(
                        (item for item in scored
                         if item.get("label") == sample_label),
                        key=lambda x: x.get(sample_label, 0),
                        default=None,
                    )
                    if not representative:
                        continue

                    confidence = representative.get(sample_label, 0)
                    headline = representative.get("title", "")

                    summary = (
                        f"[{bullish_count}🟢/{bearish_count}🔴 from "
                        f"{len(scored)} headlines] {headline}"
                    )

                    sent = self.alerter.send_news_alert(
                        ticker=ticker,
                        headline=summary,
                        sentiment_label=sample_label,
                        confidence=confidence,
                        source=representative.get("source", ""),
                    )
                    if sent:
                        self._mark_sent(key)
                        alerts_sent["news"] += 1
                        log.info(
                            f"[{ticker}] NEWS ALERT — "
                            f"{direction} "
                            f"({bullish_count}/{bearish_count})"
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

        # Cleanup old state
        cutoff = datetime.now() - timedelta(hours=48)
        cleaned_state = {}
        for k, v in self.state.items():
            if (
                k.startswith("last_signal_") or
                k == "last_newsapi_poll"
            ):
                cleaned_state[k] = v
                continue
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

    # ── Continuous loop ─────────────────────────────────────

    def run_forever(self) -> None:
        log.info(
            f"Starting live poller — "
            f"polling every {self.poll_minutes} minutes"
        )

        try:
            self.alerter.send_text(
                f"🚀 <b>Live Poller Started</b>\n\n"
                f"Polling every {self.poll_minutes} minutes\n"
                f"Capital: ${self.capital:,.2f}\n"
                f"Tickers: {len(settings.all_tickers)}\n"
                f"Min confidence: "
                f"{self.confidence_threshold:.0%}\n"
                f"Charts: "
                f"{'ENABLED 📊' if self.send_charts else 'DISABLED'}\n\n"
                f"<i>You'll receive annotated chart alerts when "
                f"signal direction changes or breaking news appears.</i>"
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
    poller = LivePoller(
        capital=500.0,
        poll_minutes=15,
        send_charts=True,
    )
    poller.run_forever()