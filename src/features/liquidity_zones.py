"""
src/features/liquidity_zones.py
─────────────────────────────────
Detects institutional liquidity zones and stop hunt patterns.

What this captures:
  1. Stop hunt zones — where retail stops cluster (just beyond
     recent swing highs/lows). Institutions sweep these for
     liquidity before reversing direction.

  2. Order blocks — the last bullish/bearish candle before a
     strong reversal. Institutions return to these to fill more
     orders (institutional supply/demand zones).

  3. Round number levels — psychological levels ($100, $200) where
     retail traders cluster orders. Smart money fades these.

  4. Liquidity sweep detection — when price quickly takes out a
     recent high/low then reverses. This is a stop hunt → reversal
     pattern with 70%+ accuracy when properly identified.

  5. Equal highs/lows — multiple touches of same price level.
     These are obvious targets institutions sweep before reversing.

Research basis:
  Smart Money Concepts (SMC) trading methodology
  ICT (Inner Circle Trader) liquidity theory
  Order flow analysis (Bookmap, NinjaTrader research)

Reference:
  "Institutional Order Flow" — DTrader Academy
  "Wyckoff Spring/Upthrust" — Richard Wyckoff (1930s, still valid)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import log


def calculate_liquidity_features(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Add liquidity zone features to OHLCV DataFrame.

    Args:
        df:       OHLCV DataFrame with open/high/low/close/volume
        lookback: How many days back to scan for swing points

    Returns:
        Same DataFrame with new columns:
          - lq_swept_high:        1 if recent high was swept (stop hunt up)
          - lq_swept_low:         1 if recent low was swept (stop hunt down)
          - lq_near_resistance:   distance to nearest resistance cluster
          - lq_near_support:      distance to nearest support cluster
          - lq_at_round_number:   1 if price near $X.00 / $X.50
          - lq_equal_highs_count: number of equal highs in window
          - lq_equal_lows_count:  number of equal lows in window
          - lq_bullish_ob:        1 if just printed a bullish order block
          - lq_bearish_ob:        1 if just printed a bearish order block
          - lq_sweep_reversal:    1 if liquidity sweep + reversal detected
    """
    df = df.copy()

    # Initialize all columns
    new_cols = [
        "lq_swept_high", "lq_swept_low",
        "lq_near_resistance", "lq_near_support",
        "lq_at_round_number",
        "lq_equal_highs_count", "lq_equal_lows_count",
        "lq_bullish_ob", "lq_bearish_ob",
        "lq_sweep_reversal",
    ]
    for c in new_cols:
        df[c] = 0.0

    if len(df) < lookback + 5:
        log.warning(
            f"Not enough data for liquidity features "
            f"({len(df)} < {lookback+5})"
        )
        return df

    # Iterate through each row using lookback window
    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i + 1]
        current = df.iloc[i]
        prev    = df.iloc[i - 1] if i > 0 else current

        try:
            # ── Stop hunt detection ────────────────────────────
            # Did today's high pierce the recent high then close below?
            recent_high = window.iloc[:-1]["high"].max()
            recent_low  = window.iloc[:-1]["low"].min()

            swept_high = (
                current["high"] > recent_high and
                current["close"] < recent_high * 0.998
            )
            swept_low = (
                current["low"] < recent_low and
                current["close"] > recent_low * 1.002
            )

            df.iloc[i, df.columns.get_loc("lq_swept_high")] = (
                int(swept_high)
            )
            df.iloc[i, df.columns.get_loc("lq_swept_low")] = (
                int(swept_low)
            )

            # ── Distance to nearest support/resistance ──────────
            close = current["close"]
            df.iloc[i, df.columns.get_loc("lq_near_resistance")] = (
                (recent_high - close) / close
            )
            df.iloc[i, df.columns.get_loc("lq_near_support")] = (
                (close - recent_low) / close
            )

            # ── Round number proximity ──────────────────────────
            df.iloc[i, df.columns.get_loc("lq_at_round_number")] = (
                int(_is_at_round_number(close))
            )

            # ── Equal highs/lows count ──────────────────────────
            equal_highs, equal_lows = _count_equal_levels(window)
            df.iloc[i, df.columns.get_loc("lq_equal_highs_count")] = (
                equal_highs
            )
            df.iloc[i, df.columns.get_loc("lq_equal_lows_count")] = (
                equal_lows
            )

            # ── Order block detection ───────────────────────────
            bullish_ob, bearish_ob = _detect_order_block(window)
            df.iloc[i, df.columns.get_loc("lq_bullish_ob")] = (
                int(bullish_ob)
            )
            df.iloc[i, df.columns.get_loc("lq_bearish_ob")] = (
                int(bearish_ob)
            )

            # ── Sweep + reversal pattern ────────────────────────
            sweep_reversal = _detect_sweep_reversal(window)
            df.iloc[i, df.columns.get_loc("lq_sweep_reversal")] = (
                int(sweep_reversal)
            )

        except Exception:
            continue

    # Cast binary flags to int
    binary_cols = [
        "lq_swept_high", "lq_swept_low", "lq_at_round_number",
        "lq_bullish_ob", "lq_bearish_ob", "lq_sweep_reversal",
    ]
    for c in binary_cols:
        df[c] = df[c].fillna(0).astype(int)

    log.info(
        f"Liquidity Zones: {len(new_cols)} features calculated"
    )
    return df


def _is_at_round_number(price: float, threshold_pct: float = 0.005) -> bool:
    """
    Check if price is near a psychological round number.

    Round numbers depend on price magnitude:
      Under $10:   $1, $2.5, $5
      $10-$100:    $5, $10, $25, $50
      $100-$1000:  $50, $100, $250, $500
      $1000+:      $500, $1000, $5000

    Returns True if within threshold_pct of a round number.
    """
    if price <= 0:
        return False

    # Pick rounding step based on price magnitude
    if price < 10:
        steps = [1, 2.5, 5]
    elif price < 100:
        steps = [5, 10, 25, 50]
    elif price < 1000:
        steps = [50, 100, 250, 500]
    else:
        steps = [500, 1000, 5000]

    threshold = price * threshold_pct
    for step in steps:
        nearest = round(price / step) * step
        if abs(price - nearest) <= threshold:
            return True
    return False


def _count_equal_levels(
    window: pd.DataFrame,
    tolerance_pct: float = 0.003,
) -> tuple[int, int]:
    """
    Count instances of equal highs and equal lows in window.

    "Equal" means within tolerance_pct of each other.
    Multiple equal levels = obvious target for institutions.
    """
    highs = window["high"].values
    lows  = window["low"].values

    equal_highs = 0
    equal_lows  = 0

    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            avg_h = (highs[i] + highs[j]) / 2
            if avg_h > 0 and abs(highs[i] - highs[j]) / avg_h < tolerance_pct:
                equal_highs += 1

            avg_l = (lows[i] + lows[j]) / 2
            if avg_l > 0 and abs(lows[i] - lows[j]) / avg_l < tolerance_pct:
                equal_lows += 1

    return equal_highs, equal_lows


def _detect_order_block(window: pd.DataFrame) -> tuple[bool, bool]:
    """
    Detect institutional order block (OB).

    Bullish OB: last bearish candle BEFORE a strong rally.
    Bearish OB: last bullish candle BEFORE a strong selloff.

    Strong move = 2%+ in 3 days.
    Returns (bullish_ob_recently, bearish_ob_recently).
    """
    if len(window) < 5:
        return False, False

    # Look at last 5 days
    recent = window.iloc[-5:]
    closes = recent["close"].values

    # Calculate 3-day return ending at current
    if len(closes) >= 4:
        three_day_return = (closes[-1] / closes[-4]) - 1
    else:
        three_day_return = 0

    bullish_ob = False
    bearish_ob = False

    # Bullish OB: large up move (3-day return > 2%) and
    # we just printed a red candle 3-4 days ago
    if three_day_return > 0.02:
        candidate = recent.iloc[-4] if len(recent) >= 4 else recent.iloc[0]
        if candidate["close"] < candidate["open"]:
            bullish_ob = True

    # Bearish OB: large down move (3-day return < -2%) and
    # we printed a green candle 3-4 days ago
    if three_day_return < -0.02:
        candidate = recent.iloc[-4] if len(recent) >= 4 else recent.iloc[0]
        if candidate["close"] > candidate["open"]:
            bearish_ob = True

    return bullish_ob, bearish_ob


def _detect_sweep_reversal(window: pd.DataFrame) -> bool:
    """
    Detect liquidity sweep + reversal pattern.

    Pattern (Wyckoff Spring / Upthrust):
      Day 1-3: range-bound trading
      Day 4: spike beyond recent high/low (stop hunt)
      Day 5: strong reversal closing back inside range

    This is a 70%+ accuracy pattern when properly formed.
    """
    if len(window) < 5:
        return False

    last = window.iloc[-1]
    prev = window.iloc[-2]
    earlier = window.iloc[:-2]

    range_high = earlier["high"].max()
    range_low  = earlier["low"].min()

    # Bullish sweep: yesterday's low broke range, today closed above
    bullish_sweep = (
        prev["low"] < range_low and
        last["close"] > range_low and
        last["close"] > last["open"]
    )

    # Bearish sweep: yesterday's high broke range, today closed below
    bearish_sweep = (
        prev["high"] > range_high and
        last["close"] < range_high and
        last["close"] < last["open"]
    )

    return bullish_sweep or bearish_sweep