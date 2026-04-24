"""
startup.py
───────────
Railway deployment startup script.

On first boot:
  1. Fetch 10 years of data for all tickers
  2. Train XGBoost + LSTM for all tickers (no CNN on cloud - too slow)
  3. Start the live poller

On subsequent boots:
  - If models exist, skip training
  - Otherwise retrain

Run: python startup.py
"""

import sys
from pathlib import Path

from config.settings import settings
from src.models.base_model import MODELS_DIR
from src.utils.logger import log


def models_exist() -> bool:
    """Check if trained models are already on disk."""
    expected = []
    for ticker in settings.all_tickers:
        expected.append(MODELS_DIR / f"xgboost_{ticker}.joblib")
        expected.append(MODELS_DIR / f"lstm_{ticker}.pt")

    existing = sum(1 for p in expected if p.exists())
    total    = len(expected)

    log.info(f"Models on disk: {existing}/{total}")
    return existing >= total * 0.9   # allow a few missing


def ensure_data() -> None:
    """Make sure pipeline has been run."""
    log.info("=" * 50)
    log.info("STEP 1: Running data pipeline")
    log.info("=" * 50)

    from run_pipeline import run_pipeline
    run_pipeline(
        include_sentiment=True,
        use_cache=True,
    )


def ensure_models() -> None:
    """Train models if they don't exist."""
    if models_exist():
        log.info("All models found — skipping training")
        return

    log.info("=" * 50)
    log.info("STEP 2: Training models (this takes ~20 minutes)")
    log.info("=" * 50)

    from src.ensemble.ensemble_trainer import train_all_tickers
    train_all_tickers(target_days=5)


def start_poller() -> None:
    """Start the live poller."""
    log.info("=" * 50)
    log.info("STEP 3: Starting live poller")
    log.info("=" * 50)

    from src.alerts.live_poller import LivePoller
    poller = LivePoller(capital=500.0, poll_minutes=15)
    poller.run_forever()


if __name__ == "__main__":
    try:
        ensure_data()
        ensure_models()
        start_poller()
    except Exception as e:
        log.error(f"Startup failed: {e}")
        sys.exit(1)