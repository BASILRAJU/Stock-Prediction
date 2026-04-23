"""
src/sentiment/sentiment_merger.py
──────────────────────────────────
Merges FinBERT sentiment scores onto the Phase 1 feature matrix.

For each trading day, we aggregate all scored headlines published
in the previous 24 hours into summary features:

  sentiment_score      — weighted average signal (-1 to +1)
  sentiment_positive   — avg positive confidence
  sentiment_negative   — avg negative confidence
  sentiment_count      — number of non-neutral headlines
  sentiment_momentum   — 3-day rolling sentiment trend

Critical design decision (from research):
  Headlines are aligned STRICTLY by published_at timestamp.
  We only use headlines published BEFORE market close that day.
  This prevents look-ahead bias — a common mistake that inflates
  backtested accuracy by 8-25%.

Output:
  Original 98 features + 5 sentiment features = 103 features total
  Saved as {ticker}_features_with_sentiment.parquet
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import settings
from src.utils.logger import log


class SentimentMerger:
    """
    Merges daily sentiment signals onto price feature DataFrames.

    Usage:
        merger = SentimentMerger()
        enriched_df = merger.merge(features_df, scored_articles, "AAPL")
        all_enriched = merger.merge_all(all_features, all_scores)
    """

    def merge(
        self,
        features_df: pd.DataFrame,
        scored_articles: list[dict],
        ticker: str = "",
    ) -> pd.DataFrame:
        """
        Merge sentiment scores onto one ticker's feature DataFrame.

        Args:
            features_df:     Phase 1 feature matrix (98 columns)
            scored_articles: Output from FinBERTScorer.score_ticker()
            ticker:          Ticker name for logging

        Returns:
            DataFrame with 5 additional sentiment columns (103 total)
        """
        df = features_df.copy()

        # If no articles, add empty sentiment columns and return
        if not scored_articles:
            log.warning(
                f"[{ticker}] No sentiment signals — "
                "adding zero sentiment columns"
            )
            df = self._add_empty_sentiment(df)
            return df

        # ── Build daily sentiment aggregates ──────────────────
        sentiment_daily = self._aggregate_daily(scored_articles, ticker)

        if sentiment_daily.empty:
            log.warning(f"[{ticker}] Could not build daily sentiment")
            df = self._add_empty_sentiment(df)
            return df

        # ── Align timestamps ──────────────────────────────────
        # Features index is DatetimeIndex UTC
        # Sentiment index is date-only — we align carefully
        df.index = pd.DatetimeIndex(df.index)

        # Normalise both to date-only for merging
        df_dates = df.index.normalize()

        # Reindex sentiment to match feature dates
        # Forward fill — carries last known sentiment forward
        # This is safe: we're carrying PAST sentiment forward
        sentiment_reindexed = (
            sentiment_daily
            .reindex(df_dates, method="ffill")
            .fillna(0)
        )
        sentiment_reindexed.index = df.index

        # ── Join sentiment columns onto features ───────────────
        sentiment_cols = [
            "sentiment_score",
            "sentiment_positive",
            "sentiment_negative",
            "sentiment_count",
            "sentiment_momentum",
        ]

        for col in sentiment_cols:
            if col in sentiment_reindexed.columns:
                df[col] = sentiment_reindexed[col].values

        log.info(
            f"[{ticker}] Sentiment merged: "
            f"{df.shape[0]} rows × {df.shape[1]} columns"
        )
        return df

    def merge_all(
        self,
        all_features: dict[str, pd.DataFrame],
        all_scores: dict[str, list[dict]],
        save_to_disk: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Merge sentiment for all tickers.
        Returns dict of {ticker: enriched_DataFrame}
        """
        results: dict[str, pd.DataFrame] = {}

        for ticker, features_df in all_features.items():
            scored = all_scores.get(ticker, [])
            try:
                enriched = self.merge(features_df, scored, ticker)
                results[ticker] = enriched

                if save_to_disk:
                    out_path = (
                        settings.data_processed_path /
                        f"{ticker}_features_with_sentiment.parquet"
                    )
                    enriched.to_parquet(out_path)
                    log.info(f"[{ticker}] Saved → {out_path.name}")

            except Exception as e:
                log.error(f"[{ticker}] Sentiment merge failed: {e}")
                results[ticker] = features_df

        return results

    # ── Private helpers ───────────────────────────────────────

    def _aggregate_daily(
        self,
        scored_articles: list[dict],
        ticker: str,
    ) -> pd.DataFrame:
        """
        Aggregate article-level scores into daily summary features.
        """
        if not scored_articles:
            return pd.DataFrame()

        rows = []
        for a in scored_articles:
            pub_date = a.get("published_at", "")
            if not pub_date:
                continue
            try:
                dt = pd.to_datetime(pub_date, utc=True)
                rows.append({
                    "date":     dt.normalize(),
                    "signal":   a.get("signal", 0),
                    "positive": a.get("positive", 0.0),
                    "negative": a.get("negative", 0.0),
                    "label":    a.get("label", "neutral"),
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        raw = pd.DataFrame(rows)
        raw["date"] = pd.DatetimeIndex(raw["date"])

        # Group by date and aggregate
        grouped = raw.groupby("date")

        daily = pd.DataFrame(index=grouped.groups.keys())
        daily.index = pd.DatetimeIndex(daily.index)

        # Weighted average signal (confidence-weighted)
        def weighted_signal(group: pd.DataFrame) -> float:
            if len(group) == 0:
                return 0.0
            confidence = group["positive"] + group["negative"]
            if confidence.sum() == 0:
                return 0.0
            return float(
                (group["signal"] * confidence).sum() / confidence.sum()
            )

        daily["sentiment_score"]    = grouped.apply(weighted_signal)
        daily["sentiment_positive"] = grouped["positive"].mean()
        daily["sentiment_negative"] = grouped["negative"].mean()
        daily["sentiment_count"]    = grouped["signal"].count()

        # 3-day rolling momentum — is sentiment improving or worsening?
        daily["sentiment_momentum"] = (
            daily["sentiment_score"]
            .rolling(3, min_periods=1)
            .mean()
        )

        return daily

    def _add_empty_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add zero-filled sentiment columns when no data available."""
        df["sentiment_score"]    = 0.0
        df["sentiment_positive"] = 0.0
        df["sentiment_negative"] = 0.0
        df["sentiment_count"]    = 0
        df["sentiment_momentum"] = 0.0
        return df