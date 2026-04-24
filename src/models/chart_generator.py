"""
src/models/chart_generator.py
───────────────────────────────
Converts OHLCV price data into candlestick chart images
for CNN training and inference.

Each image represents a 30-day window of price action.
The CNN learns to recognise visual patterns the same way
a human trader reads charts — head and shoulders, flags,
wedges, double tops, etc.

Research basis:
  "Algorithmic trading using CNN on candlestick images"
  (Applied Soft Computing, 2024): CNN on chart images
  outperforms indicator-based models by 8-12% on
  direction prediction across 5 major indices.

  MPDTransformer (2025): converts features to 2D images,
  achieving 67.3% accuracy on CSI 300 index.

Image format:
  - Size: 224x224 pixels (ResNet standard input)
  - Candlestick chart: green=bullish, red=bearish
  - Includes volume bar chart at bottom
  - No axes/labels — CNN reads pure visual patterns
  - Saved as PNG, loaded as tensor for training
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import mplfinance as mpf
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    from PIL import Image
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

from config.settings import settings
from src.utils.logger import log

# Chart cache directory
CHART_CACHE = settings.data_processed_path.parent / "charts"
CHART_CACHE.mkdir(parents=True, exist_ok=True)

# CNN input size (ResNet standard)
IMAGE_SIZE = (224, 224)

# Days of price history per chart image
WINDOW_DAYS = 30


class ChartGenerator:
    """
    Generates candlestick chart images from OHLCV data.

    Usage:
        gen = ChartGenerator()
        images, labels, dates = gen.generate_dataset("AAPL", df)
        # images shape: (N, 3, 224, 224) — ready for CNN
    """

    def __init__(self, window_days: int = WINDOW_DAYS):
        if not MPL_AVAILABLE:
            raise ImportError(
                "Run: pip install mplfinance Pillow"
            )
        self.window_days = window_days

    def ohlcv_to_image(
        self,
        window_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Convert one window of OHLCV data to a chart image.

        Args:
            window_df: DataFrame with open/high/low/close/volume
                       index must be DatetimeIndex

        Returns:
            numpy array of shape (224, 224, 3) — RGB image
        """
        # mplfinance needs specific column names
        df = window_df[["open", "high", "low", "close", "volume"]].copy()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

        # Style: clean chart, no axes text
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

        # Generate chart to buffer
        buf = io.BytesIO()
        fig, _ = mpf.plot(
            df,
            type="candle",
            style=style,
            volume=True,
            returnfig=True,
            figsize=(2.24, 2.24),   # 224px at 100dpi
            axisoff=True,           # no axes — pure visual
        )

        fig.savefig(
            buf,
            format="png",
            dpi=100,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="black",
        )
        plt.close(fig)
        buf.seek(0)

        # Convert to numpy array
        img = Image.open(buf).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0

        return arr   # shape: (224, 224, 3)

    def generate_dataset(
        self,
        ticker: str,
        df: pd.DataFrame,
        target_days: int = 5,
        use_cache: bool = True,
        max_samples: int = None,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Generate all chart images for one ticker.

        Slides a window_days window across the full history.
        Each window becomes one image labeled by future return.

        Args:
            ticker:      Ticker symbol
            df:          OHLCV DataFrame (10 years)
            target_days: Prediction horizon
            use_cache:   Load from disk if cached
            max_samples: Limit number of images (for testing)

        Returns:
            images: np.ndarray shape (N, 3, 224, 224)
            labels: np.ndarray shape (N,) — 1=up, 0=down
            dates:  list of dates for each sample
        """
        cache_file = CHART_CACHE / f"{ticker}_charts.npz"

        if use_cache and cache_file.exists():
            log.info(f"[{ticker}] Loading chart cache...")
            data = np.load(cache_file, allow_pickle=True)
            images = data["images"]
            labels = data["labels"]
            dates  = list(data["dates"])
            log.info(
                f"[{ticker}] Loaded {len(images)} cached charts"
            )
            return images, labels, dates

        log.info(
            f"[{ticker}] Generating chart images "
            f"({self.window_days}-day windows)..."
        )

        # Need open/high/low/close/volume
        required = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in required if c in df.columns]].copy()
        df = df.dropna()

        # Build target labels
        future_return = (
            df["close"].shift(-target_days) / df["close"] - 1
        )
        labels_series = (future_return > 0).astype(int)

        images_list = []
        labels_list = []
        dates_list  = []

        total = len(df) - self.window_days - target_days
        if max_samples:
            total = min(total, max_samples)

        log.info(f"[{ticker}] Building {total} chart images...")

        for i in range(total):
            window = df.iloc[i:i + self.window_days]
            label  = labels_series.iloc[i + self.window_days]
            date   = df.index[i + self.window_days]

            if pd.isna(label):
                continue

            try:
                img = self.ohlcv_to_image(window)
                # Convert to channel-first for PyTorch: (3, 224, 224)
                img = img.transpose(2, 0, 1)
                images_list.append(img)
                labels_list.append(int(label))
                dates_list.append(date)

                if (i + 1) % 100 == 0:
                    log.debug(
                        f"[{ticker}] {i+1}/{total} charts generated"
                    )

            except Exception as e:
                log.warning(
                    f"[{ticker}] Chart generation failed at {i}: {e}"
                )
                continue

        if not images_list:
            log.warning(f"[{ticker}] No charts generated")
            return np.array([]), np.array([]), []

        images = np.array(images_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64)

        # Cache to disk
        np.savez_compressed(
            cache_file,
            images=images,
            labels=labels,
            dates=np.array(dates_list, dtype=object),
        )
        log.info(
            f"[{ticker}] {len(images)} charts saved "
            f"→ {cache_file.name}"
        )

        return images, labels, dates_list