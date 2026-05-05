"""
startup.py
───────────
Railway startup — POLLER ONLY (no training).

Workflow:
  1. Download models from Hugging Face Hub (if missing locally)
  2. Verify models are present
  3. Start the live poller

Models are trained LOCALLY on your computer, uploaded to
Hugging Face Hub, and downloaded by Railway on first start.
"""

import sys

from config.settings import settings
from src.models.base_model import MODELS_DIR
from src.utils.model_downloader import ensure_models_downloaded
from src.utils.logger import log


def check_models() -> bool:
    """Make sure trained models are present."""
    found   = 0
    missing = []
    for ticker in settings.all_tickers:
        xgb  = MODELS_DIR / f"xgboost_{ticker}.joblib"
        lstm = MODELS_DIR / f"lstm_{ticker}.pt"
        if xgb.exists() and lstm.exists():
            found += 1
        else:
            missing.append(ticker)

    log.info(
        f"Models found: {found}/{len(settings.all_tickers)}"
    )
    if missing:
        log.warning(f"Missing models: {missing}")
        return found > 0
    return True


def start_live_poller() -> None:
    log.info("=" * 50)
    log.info("Starting live poller")
    log.info("=" * 50)

    from src.alerts.live_poller import LivePoller
    poller = LivePoller(
        capital=6000.0,
        poll_minutes=15,
        send_charts=True,
    )
    poller.run_forever()


if __name__ == "__main__":
    try:
        log.info("Checking for trained models...")
        if not ensure_models_downloaded():
            log.error(
                "Failed to download models from HF Hub. "
                "Check HF_TOKEN environment variable."
            )
            sys.exit(1)

        if not check_models():
            log.error("No trained models found! Aborting.")
            sys.exit(1)

        start_live_poller()

    except Exception as e:
        log.error(f"Fatal startup error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)