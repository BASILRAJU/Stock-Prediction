"""
scripts/upload_models_to_hf.py
───────────────────────────────
One-time script to upload all trained models and processed
features to Hugging Face Hub.

Run this whenever you train new models or update features.
The Railway deployment will download from HF on startup.

Usage:
    python scripts/upload_models_to_hf.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from huggingface_hub import HfApi, login
from rich.console import Console

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

console = Console()

# ─── Config ───────────────────────────────────────────────────

HF_USERNAME  = "Baski97"
HF_REPO_NAME = "stock-prediction-models"
HF_REPO_ID   = f"{HF_USERNAME}/{HF_REPO_NAME}"


def main():
    console.print(
        f"\n[bold cyan]Uploading models to Hugging Face Hub[/bold cyan]"
    )
    console.print(f"Target repo: [yellow]{HF_REPO_ID}[/yellow]\n")

    # Authenticate
    token = os.getenv("HF_TOKEN")
    if not token:
        console.print(
            "[red]ERROR: HF_TOKEN not found in environment[/red]"
        )
        console.print(
            "Set it in your .env file or run: "
            "[yellow]export HF_TOKEN=hf_xxx[/yellow]"
        )
        sys.exit(1)

    login(token=token, add_to_git_credential=False)
    api = HfApi()

    # ── Upload models directory ────────────────────────────
    models_dir = settings.data_processed_path.parent / "models"
    if not models_dir.exists():
        console.print(f"[red]Models dir not found: {models_dir}[/red]")
        sys.exit(1)

    model_files = list(models_dir.glob("*"))
    console.print(
        f"Uploading [bold]{len(model_files)}[/bold] model files..."
    )

    api.upload_folder(
        folder_path=str(models_dir),
        path_in_repo="models",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Update trained models",
    )
    console.print("  [green]✓[/green] Models uploaded\n")

    # ── Upload processed features ──────────────────────────
    processed_dir = settings.data_processed_path
    if not processed_dir.exists():
        console.print(
            f"[red]Processed dir not found: {processed_dir}[/red]"
        )
        sys.exit(1)

    feature_files = list(processed_dir.glob("*.parquet"))
    console.print(
        f"Uploading [bold]{len(feature_files)}[/bold] feature files..."
    )

    api.upload_folder(
        folder_path=str(processed_dir),
        path_in_repo="processed",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Update processed features",
    )
    console.print("  [green]✓[/green] Features uploaded\n")

    console.print(
        f"[bold green]✅ All files uploaded to "
        f"https://huggingface.co/{HF_REPO_ID}[/bold green]"
    )


if __name__ == "__main__":
    main()