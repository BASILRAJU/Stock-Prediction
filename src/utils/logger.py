"""
src/utils/logger.py
───────────────────
Centralised logging for the entire project.
Writes to both console (coloured) and a log file.

Usage anywhere in the project:
    from src.utils.logger import log
    log.info("Fetching AAPL data...")
    log.warning("Rate limit hit")
    log.error("Something went wrong")
"""

import sys
from pathlib import Path
from loguru import logger as log

from config.settings import settings


def setup_logging() -> None:
    log.remove()  # Remove default handler

    # ── Console output (coloured, human readable) ──────────────
    log.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # ── File output (full detail, rotates at 10MB) ─────────────
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    log.add(
        str(settings.log_file),
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    )


setup_logging()

__all__ = ["log"]