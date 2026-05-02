"""
src/features/volume_profile.py
───────────────────────────────
Calculates Volume Profile features for institutional analysis.

Volume Profile shows WHERE in the price range the most trading
happened — not just total volume per day. This reveals:
  - POC (Point of Control): price with highest volume — magnet level
  - Value Area: range holding 70% of volume — fair-value zone
  - High Volume Nodes (HVN): support/resistance from heavy trading
  - Low Volume Nodes (LVN): price gaps — fast-move zones

How institutions use it:
  - Place limit orders at HVN (where they got filled before)
  - Stops clustered just beyond HVN → stop hunts happen here
  - Price tends to gravitate to POC ("magnet" effect)
  - Breakouts through LVN are usually fast and sustained

Implementation:
  We approximate per-day volume profile by distributing each
  day's volume across its OHLC range using a triangular weight.
  Real Level 2 data would be more precise but unavailable in
  free OHLCV feeds.

Research basis:
  CME volume profile theory (Steidlmayer, 1980s)
  Auction Market Theory — price moves to facilitate trade
  Modern adaptations: Sierra Chart, TPO charts, Bookmap

Reference:
  Dalton, Mind Over Markets (Steidlmayer's auction theory)
  Bookmap whitepaper on volume profile institutional usage
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import log


def calculate_volume_profile(
    df: pd.DataFrame,
    lookback_days: int = 20,
    n_bins: int = 20,
    value_area_pct: float = 0.70,
) -> pd.DataFrame:
    """
    Calculate rolling volume profile features.

    For each day, looks back N days and builds a price-volume
    histogram. Extracts POC, Value Area, and node features.

    Args:
        df:             OHLCV DataFrame (must have open/high/low/close/volume)
        lookback_days:  How many days of data to build profile (default 20)
        n_bins:         Number of price buckets for histogram
        value_area_pct: % of volume to include in Value Area (70% standard)

    Returns:
        DataFrame with new columns:
          - vp_poc:           Point of Control price
          - vp_val_high:      Value Area High
          - vp_val_low:       Value Area Low
          - vp_in_value:      1 if current price in Value Area
          - vp_above_poc:     1 if current price above POC
          - vp_dist_to_poc:   Distance to POC as % of price
          - vp_dist_to_vah:   Distance to Value Area High as % of price
          - vp_dist_to_val:   Distance to Value Area Low as % of price
          - vp_hvn_above:     1 if HVN exists above current price
          - vp_hvn_below:     1 if HVN exists below current price
    """
    df = df.copy()

    # Initialize all columns to NaN
    cols = [
        "vp_poc", "vp_val_high", "vp_val_low",
        "vp_in_value", "vp_above_poc",
        "vp_dist_to_poc", "vp_dist_to_vah", "vp_dist_to_val",
        "vp_hvn_above", "vp_hvn_below",
    ]
    for c in cols:
        df[c] = np.nan

    if len(df) < lookback_days + 5:
        log.warning(
            f"Not enough data for volume profile "
            f"({len(df)} bars < {lookback_days+5} needed)"
        )
        return df

    # Calculate for each row using rolling window
    for i in range(lookback_days, len(df)):
        window = df.iloc[i - lookback_days:i]
        close  = df.iloc[i]["close"]

        try:
            profile = _build_profile(window, n_bins)
            poc, val_high, val_low = _extract_levels(
                profile, value_area_pct
            )
            hvn_above, hvn_below = _find_hvn(profile, close)

            df.iloc[i, df.columns.get_loc("vp_poc")]       = poc
            df.iloc[i, df.columns.get_loc("vp_val_high")]  = val_high
            df.iloc[i, df.columns.get_loc("vp_val_low")]   = val_low

            in_value = 1 if val_low <= close <= val_high else 0
            above_poc = 1 if close > poc else 0

            df.iloc[i, df.columns.get_loc("vp_in_value")]  = in_value
            df.iloc[i, df.columns.get_loc("vp_above_poc")] = above_poc

            df.iloc[i, df.columns.get_loc("vp_dist_to_poc")] = (
                (close - poc) / close
            )
            df.iloc[i, df.columns.get_loc("vp_dist_to_vah")] = (
                (close - val_high) / close
            )
            df.iloc[i, df.columns.get_loc("vp_dist_to_val")] = (
                (close - val_low) / close
            )

            df.iloc[i, df.columns.get_loc("vp_hvn_above")] = hvn_above
            df.iloc[i, df.columns.get_loc("vp_hvn_below")] = hvn_below

        except Exception:
            continue

    # Forward fill the binary flags (they were ints)
    for c in ["vp_in_value", "vp_above_poc", "vp_hvn_above", "vp_hvn_below"]:
        df[c] = df[c].fillna(0).astype(int)

    log.info(
        f"Volume Profile: {len(cols)} features calculated"
    )
    return df


def _build_profile(
    window: pd.DataFrame,
    n_bins: int,
) -> pd.DataFrame:
    """
    Build a price-volume histogram from OHLCV window.

    Distributes each day's volume across its high-low range
    using triangular weighting (more weight near close).

    Returns DataFrame with [price_low, price_high, volume]
    """
    # Find price range across the window
    overall_low  = window["low"].min()
    overall_high = window["high"].max()

    if overall_high <= overall_low:
        return pd.DataFrame()

    # Create price bins
    bin_edges = np.linspace(overall_low, overall_high, n_bins + 1)
    bin_volume = np.zeros(n_bins)

    for _, row in window.iterrows():
        day_low    = row["low"]
        day_high   = row["high"]
        day_close  = row["close"]
        day_volume = row["volume"]

        if day_high <= day_low or day_volume <= 0:
            continue

        # Distribute this day's volume across bins it overlaps
        # Use triangular weight peaked at close
        for j in range(n_bins):
            bin_low  = bin_edges[j]
            bin_high = bin_edges[j + 1]

            # Does this bin overlap with day's range?
            overlap_low  = max(bin_low,  day_low)
            overlap_high = min(bin_high, day_high)
            if overlap_high <= overlap_low:
                continue

            # Triangular weight: peak at close, taper to extremes
            bin_mid = (bin_low + bin_high) / 2
            distance_from_close = abs(bin_mid - day_close)
            range_size = day_high - day_low
            weight = max(0, 1 - distance_from_close / range_size)

            # Allocate volume proportional to overlap × weight
            overlap_pct = (overlap_high - overlap_low) / range_size
            bin_volume[j] += day_volume * overlap_pct * weight

    profile = pd.DataFrame({
        "price_low":  bin_edges[:-1],
        "price_high": bin_edges[1:],
        "price_mid":  (bin_edges[:-1] + bin_edges[1:]) / 2,
        "volume":     bin_volume,
    })

    return profile


def _extract_levels(
    profile: pd.DataFrame,
    value_area_pct: float,
) -> tuple[float, float, float]:
    """
    Extract POC, Value Area High, Value Area Low from profile.

    Algorithm:
      1. POC = bin with highest volume
      2. Start from POC, expand outward picking highest-volume
         neighboring bins until 70% of total volume is captured
      3. The boundaries of that range = VAH and VAL
    """
    if profile.empty:
        return np.nan, np.nan, np.nan

    poc_idx = profile["volume"].idxmax()
    poc     = profile.loc[poc_idx, "price_mid"]

    total_volume  = profile["volume"].sum()
    target_volume = total_volume * value_area_pct

    # Expand from POC outward
    included    = {poc_idx}
    cur_volume  = profile.loc[poc_idx, "volume"]
    upper_idx   = poc_idx
    lower_idx   = poc_idx

    max_iterations = len(profile) * 2
    iterations = 0

    while cur_volume < target_volume and iterations < max_iterations:
        iterations += 1

        # Look at next bin above and below
        next_up   = upper_idx + 1
        next_down = lower_idx - 1

        up_vol   = (
            profile.loc[next_up, "volume"]
            if next_up < len(profile) else 0
        )
        down_vol = (
            profile.loc[next_down, "volume"]
            if next_down >= 0 else 0
        )

        # Pick whichever has more volume
        if up_vol >= down_vol and up_vol > 0:
            included.add(next_up)
            cur_volume += up_vol
            upper_idx = next_up
        elif down_vol > 0:
            included.add(next_down)
            cur_volume += down_vol
            lower_idx = next_down
        else:
            break   # no more bins to add

    val_high = profile.loc[upper_idx, "price_high"]
    val_low  = profile.loc[lower_idx, "price_low"]

    return float(poc), float(val_high), float(val_low)


def _find_hvn(
    profile: pd.DataFrame,
    current_price: float,
    threshold_pct: float = 1.5,
) -> tuple[int, int]:
    """
    Find High Volume Nodes above and below current price.

    HVN = bin with volume > threshold_pct × average bin volume

    Returns (1, 1) flags for HVN above and below current price.
    """
    if profile.empty:
        return 0, 0

    avg_volume = profile["volume"].mean()
    threshold  = avg_volume * threshold_pct

    above_mask = (
        (profile["price_mid"] > current_price) &
        (profile["volume"] > threshold)
    )
    below_mask = (
        (profile["price_mid"] < current_price) &
        (profile["volume"] > threshold)
    )

    return (
        int(above_mask.any()),
        int(below_mask.any()),
    )