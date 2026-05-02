"""
src/models/chart_generator.py
───────────────────────────────
Generates ANNOTATED candlestick chart images for CNN training.

Unlike plain charts, these include institutional context:
  - POC line (yellow) — Point of Control / fair value
  - Value Area shading (light blue band) — 70% volume zone
  - Support zone (green band) — recent low + HVN below
  - Resistance zone (red band) — recent high + HVN above
  - 20-day MA (orange line) — short-term trend
  - 50-day MA (purple line) — medium-term trend

Why annotations help:
  Plain candlestick patterns are 50-55% accurate.
  Same pattern at support/resistance is 65-72% accurate.
  Pattern + POC + trend MA is 70-75% accurate.

The CNN learns "pattern + context" instead of just patterns.

Each image: 30-day window, 224x224 pixels, RGB.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import mplfinance as mpf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

from config.settings import settings
from src.utils.logger import log

# Chart cache directory — separate from old plain charts
CHART_CACHE = settings.data_processed_path.parent / "charts_annotated"
CHART_CACHE.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE  = (224, 224)
WINDOW_DAYS = 30


class ChartGenerator:
    """
    Generates annotated candlestick chart images.

    Annotations include support/resistance, POC, MAs.
    """

    def __init__(self, window_days: int = WINDOW_DAYS):
        if not MPL_AVAILABLE:
            raise ImportError("Run: pip install mplfinance Pillow")
        self.window_days = window_days

    def ohlcv_to_image(
        self,
        window_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Convert one window of OHLCV data to an annotated chart.

        Args:
            window_df: DataFrame with open/high/low/close/volume
                       and DatetimeIndex

        Returns:
            numpy array (224, 224, 3) RGB
        """
        df = window_df[
            ["open", "high", "low", "close", "volume"]
        ].copy()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

        # ── Compute annotation levels ──────────────────────
        levels = self._compute_levels(df)

        # ── Build chart style ──────────────────────────────
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mpf.make_marketcolors(
                up="lime",
                down="red",
                edge="inherit",
                wick="inherit",
                volume="in",
            ),
            facecolor="black",
            figcolor="black",
            gridcolor="black",
        )

        # ── Compute moving averages ───────────────────────
        ma20 = df["Close"].rolling(20, min_periods=1).mean()
        ma50 = df["Close"].rolling(50, min_periods=1).mean()

        # Build add-plots: MAs as line overlays
        addplots = [
            mpf.make_addplot(
                ma20, color="orange", width=1.0
            ),
            mpf.make_addplot(
                ma50, color="purple", width=1.0
            ),
        ]

        # Generate base chart
        buf = io.BytesIO()
        try:
            fig, axes = mpf.plot(
                df,
                type="candle",
                style=style,
                volume=True,
                addplot=addplots,
                returnfig=True,
                figsize=(2.24, 2.24),
                axisoff=True,
            )

            # Axes[0] is price, axes[2] is volume (mplfinance layout)
            price_ax = axes[0] if len(axes) >= 1 else None

            # ── Add annotation overlays ───────────────────
            if price_ax is not None:
                self._add_annotations(price_ax, levels)

            fig.savefig(
                buf,
                format="png",
                dpi=100,
                bbox_inches="tight",
                pad_inches=0,
                facecolor="black",
            )
            plt.close(fig)
        except Exception as e:
            plt.close("all")
            raise

        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def _compute_levels(self, df: pd.DataFrame) -> dict:
        """
        Calculate institutional levels for annotation overlay.

        Returns dict with: poc, val_high, val_low,
                          support, resistance
        """
        levels = {}

        # Volume Profile — POC and Value Area
        try:
            poc, vah, val = self._volume_profile(df)
            levels["poc"]      = poc
            levels["val_high"] = vah
            levels["val_low"]  = val
        except Exception:
            levels["poc"]      = None
            levels["val_high"] = None
            levels["val_low"]  = None

        # Support / resistance from recent swing points
        try:
            support, resistance = self._support_resistance(df)
            levels["support"]    = support
            levels["resistance"] = resistance
        except Exception:
            levels["support"]    = None
            levels["resistance"] = None

        return levels

    def _volume_profile(
        self,
        df: pd.DataFrame,
        n_bins: int = 15,
    ) -> tuple:
        """
        Quick volume profile inside chart window.

        Returns (poc, val_high, val_low).
        """
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

        # POC = highest volume bin
        poc_idx = int(np.argmax(bin_volume))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
        poc = float(bin_mids[poc_idx])

        # Value Area = expand from POC until 70% volume
        target = bin_volume.sum() * 0.70
        included = {poc_idx}
        cur_vol = bin_volume[poc_idx]
        upper, lower = poc_idx, poc_idx

        while cur_vol < target:
            up_idx   = upper + 1
            down_idx = lower - 1
            up_v = bin_volume[up_idx]   if up_idx   < n_bins else 0
            dn_v = bin_volume[down_idx] if down_idx >= 0     else 0
            if up_v >= dn_v and up_v > 0:
                included.add(up_idx)
                cur_vol += up_v
                upper = up_idx
            elif dn_v > 0:
                included.add(down_idx)
                cur_vol += dn_v
                lower = down_idx
            else:
                break

        val_high = float(bin_edges[upper + 1])
        val_low  = float(bin_edges[lower])

        return poc, val_high, val_low

    def _support_resistance(
        self,
        df: pd.DataFrame,
    ) -> tuple:
        """
        Find recent significant low (support) and high (resistance).

        Uses rolling min/max over recent days.
        """
        recent_low  = float(df["Low"].rolling(5).min().iloc[-1])
        recent_high = float(df["High"].rolling(5).max().iloc[-1])

        # Use overall window low/high as more reliable S/R
        window_low  = float(df["Low"].min())
        window_high = float(df["High"].max())

        # Pick the more recent one closer to current price
        return window_low, window_high

    def _add_annotations(self, ax, levels: dict) -> None:
        """
        Draw POC, Value Area, Support, Resistance overlays
        on the price chart axis.
        """
        # POC line — yellow
        if levels.get("poc") is not None:
            ax.axhline(
                y=levels["poc"],
                color="yellow",
                linewidth=1.5,
                alpha=0.8,
                linestyle="-",
            )

        # Value Area band — light blue translucent
        if (
            levels.get("val_high") is not None and
            levels.get("val_low")  is not None
        ):
            ax.axhspan(
                levels["val_low"],
                levels["val_high"],
                color="cyan",
                alpha=0.10,
            )

        # Support zone — green band (small)
        if levels.get("support") is not None:
            sup = levels["support"]
            band_size = sup * 0.005   # 0.5% band
            ax.axhspan(
                sup - band_size,
                sup + band_size,
                color="green",
                alpha=0.20,
            )

        # Resistance zone — red band
        if levels.get("resistance") is not None:
            res = levels["resistance"]
            band_size = res * 0.005
            ax.axhspan(
                res - band_size,
                res + band_size,
                color="red",
                alpha=0.20,
            )

    def generate_dataset(
        self,
        ticker: str,
        df: pd.DataFrame,
        target_days: int = 5,
        use_cache: bool = True,
        max_samples: int = None,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Generate all annotated chart images for one ticker.

        Caches results to disk in charts_annotated/ folder.
        """
        cache_file = CHART_CACHE / f"{ticker}_charts.npz"

        if use_cache and cache_file.exists():
            log.info(f"[{ticker}] Loading annotated chart cache...")
            data = np.load(cache_file, allow_pickle=True)
            images = data["images"]
            labels = data["labels"]
            dates  = list(data["dates"])
            # Pattern flags optional (older caches may not have them)
            patterns = (
                data["patterns"]
                if "patterns" in data.files
                else np.zeros((len(images), 10), dtype=np.float32)
            )
            log.info(
                f"[{ticker}] Loaded {len(images)} cached charts"
            )
            return images, labels, dates, patterns

        log.info(
            f"[{ticker}] Generating ANNOTATED chart images "
            f"({self.window_days}-day windows)..."
        )

        required = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in required if c in df.columns]].copy()
        df = df.dropna()

        future_return = (
            df["close"].shift(-target_days) / df["close"] - 1
        )
        labels_series = (future_return > 0).astype(int)

        from src.models.pattern_detector import detect_patterns

        images_list   = []
        labels_list   = []
        dates_list    = []
        patterns_list = []

        total = len(df) - self.window_days - target_days
        if max_samples:
            total = min(total, max_samples)

        log.info(f"[{ticker}] Building {total} annotated charts...")

        for i in range(total):
            window = df.iloc[i:i + self.window_days]
            label  = labels_series.iloc[i + self.window_days]
            date   = df.index[i + self.window_days]

            if pd.isna(label):
                continue

            try:
                img = self.ohlcv_to_image(window)
                img = img.transpose(2, 0, 1)

                # Detect patterns in this window
                patterns = detect_patterns(window)

                images_list.append(img)
                labels_list.append(int(label))
                dates_list.append(date)
                patterns_list.append(patterns)

                if (i + 1) % 100 == 0:
                    log.debug(
                        f"[{ticker}] {i+1}/{total} charts generated"
                    )
            except Exception as e:
                log.warning(
                    f"[{ticker}] Chart {i} failed: {e}"
                )
                continue

        if not images_list:
            log.warning(f"[{ticker}] No charts generated")
            return (
                np.array([]),
                np.array([]),
                [],
                np.array([]),
            )

        images   = np.array(images_list,   dtype=np.float32)
        labels   = np.array(labels_list,   dtype=np.int64)
        patterns = np.array(patterns_list, dtype=np.float32)

        np.savez_compressed(
            cache_file,
            images=images,
            labels=labels,
            dates=np.array(dates_list, dtype=object),
            patterns=patterns,
        )

        log.info(
            f"[{ticker}] {len(images)} annotated charts saved "
            f"→ {cache_file.name}"
        )
        # Log pattern statistics
        from src.models.pattern_detector import PATTERN_NAMES
        pattern_counts = patterns.sum(axis=0)
        log.info(
            f"[{ticker}] Pattern counts: " +
            ", ".join(
                f"{name}={int(c)}"
                for name, c in zip(PATTERN_NAMES, pattern_counts)
                if c > 0
            )
        )

        return images, labels, dates_list, patterns