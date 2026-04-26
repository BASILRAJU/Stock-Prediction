"""
startup.py
───────────
Railway deployment startup with memory-efficient training.

On first boot:
  1. Run pipeline (data + features + sentiment)
  2. Train each ticker's models in a SEPARATE subprocess
     (this releases memory between tickers)
  3. Start live poller

On subsequent boots:
  - If models already exist in volume → skip training
  - Jump straight to polling

Uses Railway volume mounted at /app/data to persist:
  - data/cache/    (feature parquets)
  - data/models/   (trained model files)
  - data/raw/      (price data)
"""

import subprocess
import sys
from pathlib import Path

from config.settings import settings
from src.models.base_model import MODELS_DIR
from src.utils.logger import log


def all_models_exist() -> bool:
    """Check if trained models are already on disk (from volume)."""
    missing = []
    for ticker in settings.all_tickers:
        xgb_path  = MODELS_DIR / f"xgboost_{ticker}.joblib"
        lstm_path = MODELS_DIR / f"lstm_{ticker}.pt"
        if not xgb_path.exists():
            missing.append(f"xgboost_{ticker}")
        if not lstm_path.exists():
            missing.append(f"lstm_{ticker}")

    if missing:
        log.info(
            f"Missing {len(missing)} models: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
        return False

    log.info(f"✓ All {len(settings.all_tickers)} models found")
    return True


def run_pipeline_if_needed() -> None:
    """Run data pipeline unless features already cached."""
    processed_dir = settings.data_processed_path
    existing = list(
        processed_dir.glob("*_features_with_sentiment.parquet")
    )

    if len(existing) >= len(settings.all_tickers) * 0.9:
        log.info(
            f"✓ Found {len(existing)} feature files "
            f"— skipping pipeline"
        )
        return

    log.info("=" * 50)
    log.info("STEP 1: Running data pipeline")
    log.info("=" * 50)

    from run_pipeline import run_pipeline
    run_pipeline(include_sentiment=True, use_cache=True)


def train_ticker_subprocess(ticker: str) -> bool:
    """
    Train one ticker in an isolated subprocess.
    Memory is fully freed when subprocess exits.
    Output is streamed live to logs so we see progress.
    """
    log.info(f"[{ticker}] Starting training subprocess...")

    cmd = [
        sys.executable, "-u", "-c",   # -u = unbuffered output
        f"""
from src.ensemble.ensemble_trainer import train_all_tickers
train_all_tickers(tickers=['{ticker}'], target_days=5)
"""
    ]

    try:
        # Stream output live so we can see training progress
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output line by line in real time
        try:
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    log.info(f"[{ticker}] {line}")
        except Exception as e:
            log.warning(f"[{ticker}] Stream error: {e}")

        # Wait for completion with timeout
        try:
            return_code = process.wait(timeout=900)   # 15 min max
        except subprocess.TimeoutExpired:
            process.kill()
            log.error(f"[{ticker}] ✗ Training timeout (15 min)")
            return False

        if return_code == 0:
            log.info(f"[{ticker}] ✓ Training complete")
            return True
        else:
            log.error(
                f"[{ticker}] ✗ Training failed "
                f"(exit code {return_code})"
            )
            return False

    except Exception as e:
        log.error(f"[{ticker}] ✗ Training error: {e}")
        return False


def train_all_models_safely() -> None:
    """
    Train each ticker in a separate process.
    This prevents memory accumulation across tickers.
    """
    if all_models_exist():
        log.info("Skipping training — all models present")
        return

    log.info("=" * 50)
    log.info(
        f"STEP 2: Training models for "
        f"{len(settings.all_tickers)} tickers"
    )
    log.info(
        "Each ticker trained in isolated subprocess "
        "(15 min timeout per ticker)"
    )
    log.info("=" * 50)

    success = 0
    failed  = []

    for i, ticker in enumerate(settings.all_tickers, 1):
        log.info(
            f"({i}/{len(settings.all_tickers)}) "
            f"Training {ticker}..."
        )

        # Skip if already trained
        xgb_path  = MODELS_DIR / f"xgboost_{ticker}.joblib"
        lstm_path = MODELS_DIR / f"lstm_{ticker}.pt"
        if xgb_path.exists() and lstm_path.exists():
            log.info(f"[{ticker}] Already trained — skipping")
            success += 1
            continue

        if train_ticker_subprocess(ticker):
            success += 1
        else:
            failed.append(ticker)

    log.info("=" * 50)
    log.info(
        f"Training complete: "
        f"{success}/{len(settings.all_tickers)} succeeded"
    )
    if failed:
        log.warning(f"Failed tickers: {failed}")
    log.info("=" * 50)


def start_live_poller() -> None:
    """Start the live poller (main process, stays alive forever)."""
    log.info("=" * 50)
    log.info("STEP 3: Starting live poller")
    log.info("=" * 50)

    from src.alerts.live_poller import LivePoller
    poller = LivePoller(capital=500.0, poll_minutes=15)
    poller.run_forever()


if __name__ == "__main__":
    try:
        run_pipeline_if_needed()
        train_all_models_safely()
        start_live_poller()
    except Exception as e:
        log.error(f"Fatal startup error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)