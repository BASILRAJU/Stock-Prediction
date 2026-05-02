"""
src/alerts/chart_annotator.py
──────────────────────────────
Generates trade-ready annotated chart images for Telegram alerts.

For each signal, builds a chart showing:
  - 30-day candlestick price action
  - 20-MA (orange) and 50-MA (purple) lines
  - Volume Profile POC line (yellow)
  - Value Area band (cyan)
  - Support zone (green band)
  - Resistance zone (red band)
  - Smart stop level (orange dashed)
  - Target level (cyan dashed)
  - Entry marker (green/red arrow)
  - Pattern label if detected
  - Title with ticker, signal, confidence

Returns PNG bytes ready to send via Telegram.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    from PIL import Image
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

from src.utils.logger import log


def generate_alert_chart(
    ticker:           str,
    df:               pd.DataFrame,
    signal:           str,
    confidence:       float,
    entry_price:      float,
    stop_loss:        Optional[float] = None,
    target_1:         Optional[float] = None,
    target_2:         Optional[float] = None,
    detected_patterns: list = None,
    window_days:      int = 30,
) -> Optional[bytes]:
    """
    Generate annotated chart and return PNG bytes.

    Args:
        ticker:            Ticker symbol
        df:                OHLCV DataFrame with full history
        signal:            "BULLISH" or "BEARISH"
        confidence:        0.0-1.0
        entry_price:       Entry price
        stop_loss:         Smart stop level
        target_1:          1:2 R:R target
        target_2:          1:3 R:R target
        detected_patterns: List of pattern names found
        window_days:       Chart window length

    Returns:
        PNG image bytes, or None on error
    """
    if not MPL_AVAILABLE:
        log.error("mplfinance not installed")
        return None

    try:
        # Get last N days
        window = df[
            ["open", "high", "low", "close", "volume"]
        ].tail(window_days).copy()
        window.columns = ["Open", "High", "Low", "Close", "Volume"]

        if len(window) < 10:
            log.warning(f"[{ticker}] Not enough data for chart")
            return None

        # Compute levels for annotation
        levels = _compute_levels(window)

        # Setup chart style
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mpf.make_marketcolors(
                up="#26a69a",
                down="#ef5350",
                edge="inherit",
                wick="inherit",
                volume="in",
            ),
            facecolor="#1a1a2e",
            figcolor="#0f0f1e",
            gridcolor="#2a2a3e",
            rc={
                "axes.labelcolor":  "white",
                "axes.edgecolor":   "#3a3a4e",
                "xtick.color":      "white",
                "ytick.color":      "white",
                "axes.titlecolor":  "white",
            },
        )

        # Moving averages
        ma20 = window["Close"].rolling(20, min_periods=1).mean()
        ma50 = window["Close"].rolling(50, min_periods=1).mean()

        addplots = [
            mpf.make_addplot(
                ma20, color="#ffa726", width=1.2,
                label="20 MA",
            ),
            mpf.make_addplot(
                ma50, color="#ab47bc", width=1.2,
                label="50 MA",
            ),
        ]

        # Build figure
        fig, axes = mpf.plot(
            window,
            type="candle",
            style=style,
            volume=True,
            addplot=addplots,
            returnfig=True,
            figsize=(11, 7),
            tight_layout=True,
        )

        price_ax  = axes[0]

        # Add overlays
        _add_volume_profile(price_ax, levels)
        _add_support_resistance(price_ax, levels)
        _add_trade_levels(
            price_ax, entry_price, stop_loss,
            target_1, target_2, signal,
        )

        # Title with signal info
        title = _build_title(
            ticker, signal, confidence, detected_patterns
        )
        price_ax.set_title(
            title, fontsize=13, fontweight="bold",
            color="white", pad=10,
        )

        # Watermark / footer
        fig.text(
            0.99, 0.01,
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
            ha="right", va="bottom",
            color="#666", fontsize=8,
        )

        # Save to bytes
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", dpi=110,
            bbox_inches="tight",
            facecolor="#0f0f1e",
        )
        plt.close(fig)
        buf.seek(0)

        log.info(f"[{ticker}] Alert chart generated")
        return buf.getvalue()

    except Exception as e:
        log.error(f"[{ticker}] Chart generation failed: {e}")
        plt.close("all")
        return None


def _compute_levels(window: pd.DataFrame) -> dict:
    """Calculate annotation levels from OHLCV window."""
    levels = {}

    # Volume profile
    try:
        poc, vah, val = _quick_volume_profile(window)
        levels["poc"]      = poc
        levels["val_high"] = vah
        levels["val_low"]  = val
    except Exception:
        levels["poc"]      = None
        levels["val_high"] = None
        levels["val_low"]  = None

    # Recent swing levels
    levels["resistance"] = float(window["High"].max())
    levels["support"]    = float(window["Low"].min())

    # Recent 5-day swings (for tighter S/R)
    if len(window) >= 5:
        recent = window.tail(10)
        levels["recent_high"] = float(recent["High"].max())
        levels["recent_low"]  = float(recent["Low"].min())

    return levels


def _quick_volume_profile(
    df: pd.DataFrame,
    n_bins: int = 15,
) -> tuple:
    """Quick VP — returns (POC, Value Area High, Value Area Low)."""
    overall_low  = df["Low"].min()
    overall_high = df["High"].max()
    if overall_high <= overall_low:
        return None, None, None

    bin_edges = np.linspace(overall_low, overall_high, n_bins + 1)
    bin_volume = np.zeros(n_bins)

    for _, row in df.iterrows():
        if row["High"] <= row["Low"] or row["Volume"] <= 0:
            continue
        day_range = row["High"] - row["Low"]
        for j in range(n_bins):
            bin_low = bin_edges[j]
            bin_high = bin_edges[j + 1]
            overlap_low  = max(bin_low,  row["Low"])
            overlap_high = min(bin_high, row["High"])
            if overlap_high <= overlap_low:
                continue
            overlap_pct = (overlap_high - overlap_low) / day_range
            bin_volume[j] += row["Volume"] * overlap_pct

    if bin_volume.sum() == 0:
        return None, None, None

    poc_idx  = int(np.argmax(bin_volume))
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    poc      = float(bin_mids[poc_idx])

    target = bin_volume.sum() * 0.70
    cur_vol = bin_volume[poc_idx]
    upper, lower = poc_idx, poc_idx

    while cur_vol < target:
        up_idx   = upper + 1
        down_idx = lower - 1
        up_v = bin_volume[up_idx]   if up_idx   < n_bins else 0
        dn_v = bin_volume[down_idx] if down_idx >= 0     else 0
        if up_v >= dn_v and up_v > 0:
            cur_vol += up_v
            upper = up_idx
        elif dn_v > 0:
            cur_vol += dn_v
            lower = down_idx
        else:
            break

    val_high = float(bin_edges[upper + 1])
    val_low  = float(bin_edges[lower])

    return poc, val_high, val_low


def _add_volume_profile(ax, levels: dict) -> None:
    """Draw POC line and Value Area band."""
    if levels.get("poc"):
        ax.axhline(
            y=levels["poc"],
            color="#ffeb3b",
            linewidth=1.5,
            alpha=0.9,
            linestyle="-",
            label=f"POC ${levels['poc']:.2f}",
        )
        # POC label
        ax.text(
            0.02, levels["poc"],
            f" POC ${levels['poc']:.2f}",
            transform=ax.get_yaxis_transform(),
            color="#ffeb3b", fontsize=9, fontweight="bold",
            verticalalignment="center",
        )

    if (levels.get("val_high") is not None and
            levels.get("val_low") is not None):
        ax.axhspan(
            levels["val_low"],
            levels["val_high"],
            color="#00bcd4",
            alpha=0.08,
        )


def _add_support_resistance(ax, levels: dict) -> None:
    """Draw support and resistance zones."""
    if levels.get("support") is not None:
        sup = levels["support"]
        band = sup * 0.005
        ax.axhspan(
            sup - band, sup + band,
            color="#26a69a",
            alpha=0.20,
        )
        ax.text(
            0.98, sup,
            f"SUP ${sup:.2f} ",
            transform=ax.get_yaxis_transform(),
            color="#26a69a", fontsize=9, fontweight="bold",
            verticalalignment="center",
            horizontalalignment="right",
        )

    if levels.get("resistance") is not None:
        res = levels["resistance"]
        band = res * 0.005
        ax.axhspan(
            res - band, res + band,
            color="#ef5350",
            alpha=0.20,
        )
        ax.text(
            0.98, res,
            f"RES ${res:.2f} ",
            transform=ax.get_yaxis_transform(),
            color="#ef5350", fontsize=9, fontweight="bold",
            verticalalignment="center",
            horizontalalignment="right",
        )


def _add_trade_levels(
    ax,
    entry_price: float,
    stop_loss:   Optional[float],
    target_1:    Optional[float],
    target_2:    Optional[float],
    signal:      str,
) -> None:
    """Draw entry, stop, and target lines."""
    # Entry line
    color = "#4caf50" if signal == "BULLISH" else "#f44336"
    ax.axhline(
        y=entry_price,
        color=color,
        linewidth=2,
        linestyle="-",
        alpha=0.9,
    )
    ax.text(
        0.5, entry_price,
        f" ENTRY ${entry_price:.2f} ",
        transform=ax.get_yaxis_transform(),
        color=color, fontsize=10, fontweight="bold",
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="black", edgecolor=color,
        ),
    )

    # Stop loss
    if stop_loss is not None and stop_loss > 0:
        ax.axhline(
            y=stop_loss,
            color="#ff9800",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
        )
        ax.text(
            0.5, stop_loss,
            f" STOP ${stop_loss:.2f} ",
            transform=ax.get_yaxis_transform(),
            color="#ff9800", fontsize=9, fontweight="bold",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="black", edgecolor="#ff9800",
            ),
        )

    # Target 1
    if target_1 is not None and target_1 > 0:
        ax.axhline(
            y=target_1,
            color="#00e5ff",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
        )
        ax.text(
            0.5, target_1,
            f" TGT1 ${target_1:.2f} ",
            transform=ax.get_yaxis_transform(),
            color="#00e5ff", fontsize=9, fontweight="bold",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="black", edgecolor="#00e5ff",
            ),
        )

    # Target 2
    if target_2 is not None and target_2 > 0:
        ax.axhline(
            y=target_2,
            color="#69f0ae",
            linewidth=1.2,
            linestyle=":",
            alpha=0.7,
        )
        ax.text(
            0.5, target_2,
            f" TGT2 ${target_2:.2f} ",
            transform=ax.get_yaxis_transform(),
            color="#69f0ae", fontsize=9, fontweight="bold",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="black", edgecolor="#69f0ae",
            ),
        )


def _build_title(
    ticker:            str,
    signal:            str,
    confidence:        float,
    detected_patterns: list = None,
) -> str:
    """Build chart title string."""
    emoji  = "🟢" if signal == "BULLISH" else "🔴"
    title  = f"{emoji} {ticker}  {signal}  ({confidence:.0%} confidence)"

    if detected_patterns:
        patterns_str = " • ".join(
            p.replace("_", " ").upper()
            for p in detected_patterns[:3]
        )
        title += f"\n{patterns_str}"

    return title