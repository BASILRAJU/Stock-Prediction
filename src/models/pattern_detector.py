"""
src/models/pattern_detector.py
────────────────────────────────
Pre-detects high-probability candlestick patterns for CNN training.

Patterns labeled here become AUXILIARY targets during CNN training.
The CNN learns BOTH:
  1. Will price go up in 5 days? (main task)
  2. What pattern is in this chart? (auxiliary task)

Multi-task learning forces the CNN to learn semantic features
(real patterns) instead of overfitting to noise. Research shows
3-7% accuracy improvements over single-task training.

Patterns detected (10 high-probability setups):
  - bullish_engulfing
  - bearish_engulfing
  - hammer (bullish reversal)
  - shooting_star (bearish reversal)
  - three_white_soldiers
  - three_black_crows
  - morning_star
  - evening_star
  - bullish_pin_bar
  - bearish_pin_bar

Returns one-hot vector of shape (10,) per chart.
Multiple patterns can be present (multi-label).

Research:
  Bulkowski "Encyclopedia of Candlestick Charts" (2008)
  Has statistical win rates for each pattern at S/R levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Patterns we detect, in fixed order
PATTERN_NAMES = [
    "bullish_engulfing",
    "bearish_engulfing",
    "hammer",
    "shooting_star",
    "three_white_soldiers",
    "three_black_crows",
    "morning_star",
    "evening_star",
    "bullish_pin_bar",
    "bearish_pin_bar",
]

N_PATTERNS = len(PATTERN_NAMES)


def detect_patterns(window_df: pd.DataFrame) -> np.ndarray:
    """
    Detect all patterns in the last few candles of a window.

    Args:
        window_df: OHLCV DataFrame for chart window
                   Last few rows are most important
                   Columns: open, high, low, close, volume

    Returns:
        Array shape (10,) — binary flags for each pattern
    """
    flags = np.zeros(N_PATTERNS, dtype=np.float32)

    if len(window_df) < 4:
        return flags

    # Normalize column names
    df = window_df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Get last 4 candles for pattern detection
    c4 = df.iloc[-4] if len(df) >= 4 else None
    c3 = df.iloc[-3] if len(df) >= 3 else None
    c2 = df.iloc[-2] if len(df) >= 2 else None
    c1 = df.iloc[-1]

    # Compute candle metrics for c1 (most recent)
    body1   = abs(c1["close"] - c1["open"])
    range1  = c1["high"] - c1["low"]
    upper1  = c1["high"] - max(c1["open"], c1["close"])
    lower1  = min(c1["open"], c1["close"]) - c1["low"]
    bullish1 = c1["close"] > c1["open"]
    bearish1 = c1["close"] < c1["open"]

    # ── Pattern 1: Bullish Engulfing ──────────────────────
    if c2 is not None:
        prev_bearish = c2["close"] < c2["open"]
        curr_bullish = bullish1
        engulfs = (
            c1["close"] > c2["open"] and
            c1["open"]  < c2["close"]
        )
        if prev_bearish and curr_bullish and engulfs:
            flags[0] = 1.0

    # ── Pattern 2: Bearish Engulfing ──────────────────────
    if c2 is not None:
        prev_bullish = c2["close"] > c2["open"]
        curr_bearish = bearish1
        engulfs = (
            c1["close"] < c2["open"] and
            c1["open"]  > c2["close"]
        )
        if prev_bullish and curr_bearish and engulfs:
            flags[1] = 1.0

    # ── Pattern 3: Hammer (bullish reversal) ─────────────
    if range1 > 0:
        body_pct = body1 / range1
        if (
            lower1 > 2 * body1 and
            upper1 < body1 and
            body_pct < 0.4
        ):
            flags[2] = 1.0

    # ── Pattern 4: Shooting Star ──────────────────────────
    if range1 > 0:
        body_pct = body1 / range1
        if (
            upper1 > 2 * body1 and
            lower1 < body1 and
            body_pct < 0.4 and
            bearish1
        ):
            flags[3] = 1.0

    # ── Pattern 5: Three White Soldiers ───────────────────
    if c3 is not None and c2 is not None:
        all_bullish = (
            c3["close"] > c3["open"] and
            c2["close"] > c2["open"] and
            c1["close"] > c1["open"]
        )
        ascending = (
            c1["close"] > c2["close"] and
            c2["close"] > c3["close"]
        )
        big_bodies = True
        for c in [c3, c2, c1]:
            cr = c["high"] - c["low"]
            cb = abs(c["close"] - c["open"])
            if cr <= 0 or cb / cr < 0.5:
                big_bodies = False
                break
        if all_bullish and ascending and big_bodies:
            flags[4] = 1.0

    # ── Pattern 6: Three Black Crows ──────────────────────
    if c3 is not None and c2 is not None:
        all_bearish = (
            c3["close"] < c3["open"] and
            c2["close"] < c2["open"] and
            c1["close"] < c1["open"]
        )
        descending = (
            c1["close"] < c2["close"] and
            c2["close"] < c3["close"]
        )
        big_bodies = True
        for c in [c3, c2, c1]:
            cr = c["high"] - c["low"]
            cb = abs(c["close"] - c["open"])
            if cr <= 0 or cb / cr < 0.5:
                big_bodies = False
                break
        if all_bearish and descending and big_bodies:
            flags[5] = 1.0

    # ── Pattern 7: Morning Star (3-candle bullish) ───────
    if c3 is not None and c2 is not None:
        body3 = abs(c3["close"] - c3["open"])
        body2 = abs(c2["close"] - c2["open"])
        c3_red   = c3["close"] < c3["open"]
        small_c2 = body2 < body3 * 0.4
        c1_green = bullish1
        midpoint = (c3["open"] + c3["close"]) / 2
        c1_strong = c1["close"] > midpoint
        if c3_red and small_c2 and c1_green and c1_strong:
            flags[6] = 1.0

    # ── Pattern 8: Evening Star (3-candle bearish) ───────
    if c3 is not None and c2 is not None:
        body3 = abs(c3["close"] - c3["open"])
        body2 = abs(c2["close"] - c2["open"])
        c3_green = c3["close"] > c3["open"]
        small_c2 = body2 < body3 * 0.4
        c1_red   = bearish1
        midpoint = (c3["open"] + c3["close"]) / 2
        c1_weak = c1["close"] < midpoint
        if c3_green and small_c2 and c1_red and c1_weak:
            flags[7] = 1.0

    # ── Pattern 9: Bullish Pin Bar ────────────────────────
    # Long lower wick, small body, closing in upper third
    if range1 > 0 and lower1 > 2.5 * body1:
        close_position = (c1["close"] - c1["low"]) / range1
        if close_position > 0.66:
            flags[8] = 1.0

    # ── Pattern 10: Bearish Pin Bar ───────────────────────
    if range1 > 0 and upper1 > 2.5 * body1:
        close_position = (c1["close"] - c1["low"]) / range1
        if close_position < 0.33:
            flags[9] = 1.0

    return flags