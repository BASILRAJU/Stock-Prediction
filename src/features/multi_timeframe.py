"""
src/features/multi_timeframe.py
─────────────────────────────────
Multi-timeframe analysis features.

Professional traders never look at just one timeframe. The best
trades happen when MULTIPLE timeframes align in the same direction:

  Monthly bullish + Weekly bullish + Daily bullish = strong buy
  Monthly bullish + Weekly bullish + Daily neutral = wait
  Monthly bearish + anything bullish = avoid (counter-trend rally)

Why this matters:
  - Daily noise gets smoothed out by weekly/monthly views
  - Major trend direction is set on higher timeframes
  - Lower timeframes provide entry timing within higher trend
  - Counter-trend trades on lower timeframes typically fail

What we add as features (8 total):
  - weekly_trend_up:    1 if weekly close > weekly 10-MA
  - weekly_rsi:         RSI on weekly timeframe
  - weekly_pct_change:  weekly % change
  - monthly_trend_up:   1 if monthly close > monthly 6-MA
  - monthly_rsi:        RSI on monthly timeframe
  - monthly_pct_change: monthly % change
  - mtf_alignment:      0-3 score, how many timeframes agree
  - mtf_strength:       continuous 0-1 score of alignment

Research basis:
  Multiple timeframe analysis (Brian Shannon, Linda Raschke)
  Wyckoff method emphasizes monthly/weekly context for daily setups
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import log


def calculate_mtf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multi-timeframe features to daily OHLCV DataFrame.

    Args:
        df: Daily OHLCV DataFrame with DatetimeIndex

    Returns:
        Same DataFrame with new columns:
          weekly_trend_up, weekly_rsi, weekly_pct_change,
          monthly_trend_up, monthly_rsi, monthly_pct_change,
          mtf_alignment, mtf_strength
    """
    df = df.copy()

    # Initialize columns
    new_cols = [
        "weekly_trend_up",
        "weekly_rsi",
        "weekly_pct_change",
        "monthly_trend_up",
        "monthly_rsi",
        "monthly_pct_change",
        "mtf_alignment",
        "mtf_strength",
    ]
    for c in new_cols:
        df[c] = np.nan

    if len(df) < 200:
        log.warning(
            f"Not enough data for MTF features ({len(df)} < 200)"
        )
        return df

    try:
        # ── Resample to weekly ─────────────────────────────────
        weekly = _resample_ohlcv(df, "W-FRI")
        weekly["sma10"]    = weekly["close"].rolling(10).mean()
        weekly["rsi"]      = _calculate_rsi(weekly["close"], period=14)
        weekly["pct_chg"]  = weekly["close"].pct_change()
        weekly["trend_up"] = (
            weekly["close"] > weekly["sma10"]
        ).astype(int)

        # ── Resample to monthly ─────────────────────────────────
        monthly = _resample_ohlcv(df, "ME")
        monthly["sma6"]     = monthly["close"].rolling(6).mean()
        monthly["rsi"]      = _calculate_rsi(monthly["close"], period=14)
        monthly["pct_chg"]  = monthly["close"].pct_change()
        monthly["trend_up"] = (
            monthly["close"] > monthly["sma6"]
        ).astype(int)

        # ── Merge weekly/monthly back to daily index ───────────
        # Use forward-fill so each day gets the latest weekly/monthly
        weekly_aligned = (
            weekly[["sma10", "rsi", "pct_chg", "trend_up"]]
            .reindex(df.index, method="ffill")
        )
        monthly_aligned = (
            monthly[["sma6", "rsi", "pct_chg", "trend_up"]]
            .reindex(df.index, method="ffill")
        )

        df["weekly_trend_up"]    = weekly_aligned["trend_up"]
        df["weekly_rsi"]         = weekly_aligned["rsi"]
        df["weekly_pct_change"]  = weekly_aligned["pct_chg"]
        df["monthly_trend_up"]   = monthly_aligned["trend_up"]
        df["monthly_rsi"]        = monthly_aligned["rsi"]
        df["monthly_pct_change"] = monthly_aligned["pct_chg"]

        # ── Daily trend (using existing 50-MA) ─────────────────
        sma50 = df["close"].rolling(50).mean()
        daily_trend_up = (df["close"] > sma50).astype(int)

        # ── Alignment score (0-3) ──────────────────────────────
        df["mtf_alignment"] = (
            daily_trend_up.fillna(0).astype(int) +
            df["weekly_trend_up"].fillna(0).astype(int) +
            df["monthly_trend_up"].fillna(0).astype(int)
        )

        # ── Strength score (continuous, 0-1) ───────────────────
        # Higher when all timeframes have RSI in same direction
        df["mtf_strength"] = _calculate_strength(
            daily_rsi=_calculate_rsi(df["close"], period=14),
            weekly_rsi=df["weekly_rsi"],
            monthly_rsi=df["monthly_rsi"],
        )

        # Cast binary flags to int
        for c in ["weekly_trend_up", "monthly_trend_up"]:
            df[c] = df[c].fillna(0).astype(int)

        df["mtf_alignment"] = df["mtf_alignment"].fillna(0).astype(int)

        log.info(f"Multi-Timeframe: {len(new_cols)} features calculated")

    except Exception as e:
        log.warning(f"MTF features failed: {e}")

    return df


def _resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample daily OHLCV to higher timeframe.

    freq: 'W' for weekly, 'M' for monthly
    """
    resampled = pd.DataFrame()
    resampled["open"]   = df["open"].resample(freq).first()
    resampled["high"]   = df["high"].resample(freq).max()
    resampled["low"]    = df["low"].resample(freq).min()
    resampled["close"]  = df["close"].resample(freq).last()
    resampled["volume"] = df["volume"].resample(freq).sum()
    resampled = resampled.dropna()
    return resampled


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI on a price series."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _calculate_strength(
    daily_rsi:   pd.Series,
    weekly_rsi:  pd.Series,
    monthly_rsi: pd.Series,
) -> pd.Series:
    """
    Calculate continuous alignment strength (0-1).

    Higher when all RSIs agree on direction (both > 50 or both < 50).
    Maximum strength when all 3 are strongly directional and aligned.
    """
    daily_dir   = (daily_rsi   - 50) / 50
    weekly_dir  = (weekly_rsi  - 50) / 50
    monthly_dir = (monthly_rsi - 50) / 50

    # All same sign = aligned
    same_sign = (
        (np.sign(daily_dir) == np.sign(weekly_dir)) &
        (np.sign(weekly_dir) == np.sign(monthly_dir))
    )

    # Average magnitude of directionality
    avg_strength = (
        daily_dir.abs() + weekly_dir.abs() + monthly_dir.abs()
    ) / 3

    strength = avg_strength.where(same_sign, 0)
    return strength.fillna(0).clip(0, 1)