"""
startup.py
───────────
Railway startup — POLLER ONLY (no training).

Models must be pre-trained locally and committed to git or
synced to the volume. This script:
  1. Verifies models exist
  2. Starts the live poller

Training is done locally on your computer (much faster) then
models are pushed to GitHub. Railway picks them up on deploy.
"""

import sys
from config.settings import settings
from src.models.base_model import MODELS_DIR
from src.utils.logger import log


def check_models() -> bool:
    """Make sure trained models are present."""
    found    = 0
    missing  = []
    for ticker in settings.all_tickers:
        xgb = MODELS_DIR / f"xgboost_{ticker}.joblib"
        lstm = MODELS_DIR / f"lstm_{ticker}.pt"
        if xgb.exists() and lstm.exists():
            found += 1
        else:
            missing.append(ticker)

    log.info(f"Models found: {found}/{len(settings.all_tickers)}")
    if missing:
        log.warning(f"Missing models: {missing}")
        log.warning(
            "Train locally first: "
            "python -m src.ensemble.ensemble_trainer"
        )
        return found > 0   # at least some models, can poll those

    return True


def start_live_poller() -> None:
    log.info("=" * 50)
    log.info("Starting live poller")
    log.info("=" * 50)

    from src.alerts.live_poller import LivePoller
    poller = LivePoller(capital=500.0, poll_minutes=15)
    poller.run_forever()


if __name__ == "__main__":
    try:
        if not check_models():
            log.error("No trained models found! Aborting.")
            sys.exit(1)
        start_live_poller()
    except Exception as e:
        log.error(f"Fatal startup error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)