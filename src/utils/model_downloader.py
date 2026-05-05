"""
src/utils/model_downloader.py
──────────────────────────────
Downloads trained models and processed features from
Hugging Face Hub to the local filesystem.

Called on Railway startup before models are loaded.
Caches files so subsequent restarts are instant.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from config.settings import settings
from src.utils.logger import log


HF_USERNAME  = "Baski97"
HF_REPO_NAME = "stock-prediction-models"
HF_REPO_ID   = f"{HF_USERNAME}/{HF_REPO_NAME}"


def ensure_models_downloaded() -> bool:
    """
    Check if models exist locally. If not, download from HF Hub.

    Returns True if models are ready to use, False on error.
    """
    models_dir   = settings.data_processed_path.parent / "models"
    processed_dir = settings.data_processed_path

    # Check if models already present
    model_count   = len(list(models_dir.glob("*.pt"))) if models_dir.exists() else 0
    feature_count = len(list(processed_dir.glob("*.parquet"))) if processed_dir.exists() else 0

    if model_count >= 27 and feature_count >= 27:
        log.info(
            f"Models already present locally "
            f"({model_count} models, {feature_count} features) — "
            f"skipping download"
        )
        return True

    log.info(
        f"Downloading models from Hugging Face Hub "
        f"({HF_REPO_ID})..."
    )

    token = os.getenv("HF_TOKEN")
    if not token:
        log.error(
            "HF_TOKEN environment variable not set. "
            "Cannot download models."
        )
        return False

    try:
        # Download models folder
        models_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="model",
            local_dir=str(models_dir.parent),
            allow_patterns=["models/*", "processed/*"],
            token=token,
        )

        # Verify
        model_count = len(list(models_dir.glob("*.pt"))) + \
                      len(list(models_dir.glob("*.joblib")))
        feature_count = len(list(processed_dir.glob("*.parquet")))

        log.info(
            f"Download complete: {model_count} models, "
            f"{feature_count} features"
        )
        return True

    except Exception as e:
        log.error(f"Failed to download models from HF: {e}")
        return False


if __name__ == "__main__":
    success = ensure_models_downloaded()
    if success:
        log.info("Models ready")
    else:
        log.error("Model download failed")