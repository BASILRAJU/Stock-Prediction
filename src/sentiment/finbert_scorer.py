"""
src/sentiment/finbert_scorer.py
────────────────────────────────
Scores financial news headlines using FinBERT.

FinBERT is a BERT model fine-tuned specifically on financial text.
It classifies each headline as:
  - positive  (bullish signal)
  - negative  (bearish signal)
  - neutral   (no signal — filtered out per research findings)

Research basis:
  FinBERT study (2025): achieves ~80% accuracy in ensemble.
  Critical finding: neutral headlines must be removed —
  they add noise and degrade downstream model performance by 5-15%.
  Look-ahead bias prevention: we score headlines BEFORE merging
  onto price data, and align by published_at timestamp only.

First run: downloads FinBERT model (~400MB) and caches it locally.
Subsequent runs: loads from local cache instantly.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from config.settings import settings
from src.utils.logger import log


# ─── FinBERT model name ───────────────────────────────────────
# ProsusAI/finbert is the most cited financial sentiment model
# in academic research (2024-2025 studies)

FINBERT_MODEL = "ProsusAI/finbert"

# ─── Score cache ──────────────────────────────────────────────

def _score_cache_path(ticker: str, date_str: str) -> Path:
    safe = ticker.replace("-", "_").replace(".", "_")
    return settings.data_cache_path / f"scores_{safe}_{date_str}.json"


def _is_cache_fresh(path: Path, max_age_hours: float = 6.0) -> bool:
    if not path.exists():
        return False
    age = (datetime.utcnow().timestamp() - path.stat().st_mtime) / 3600
    return age < max_age_hours


# ─── Main Scorer ──────────────────────────────────────────────

class FinBERTScorer:
    """
    Scores financial headlines using FinBERT.

    Loads the model once and reuses it for all tickers.
    Model is cached locally after first download (~400MB).

    Usage:
        scorer = FinBERTScorer()
        scores = scorer.score_ticker("AAPL", articles)
        all_scores = scorer.score_all_tickers(all_articles)
    """

    def __init__(self):
        self._pipeline = None
        log.info("FinBERTScorer initialised (model loads on first use)")

    def _load_model(self) -> None:
        """
        Load FinBERT model. Downloads on first run (~400MB).
        Cached locally by HuggingFace in ~/.cache/huggingface/
        """
        if self._pipeline is not None:
            return

        log.info(
            f"Loading FinBERT model: {FINBERT_MODEL}\n"
            "  First run downloads ~400MB — subsequent runs are instant."
        )

        # Use GPU if available, otherwise CPU
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        log.info(f"Running on: {device_name}")

        self._pipeline = pipeline(
            task="text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            device=device,
            top_k=None,       # return all 3 class scores
            truncation=True,
            max_length=512,
        )
        log.info("FinBERT model loaded and ready")

    def score_headline(self, text: str) -> dict:
        """
        Score a single headline.

        Returns dict with:
            label:    'positive', 'negative', or 'neutral'
            positive: confidence score 0.0-1.0
            negative: confidence score 0.0-1.0
            neutral:  confidence score 0.0-1.0
            signal:   +1 (bullish), -1 (bearish), 0 (neutral)
        """
        self._load_model()

        if not text or len(text.strip()) < 5:
            return {
                "label": "neutral",
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "signal": 0,
            }

        try:
            # Truncate very long headlines to first 512 chars
            text = text[:512]
            results = self._pipeline(text)[0]

            # Convert list of {label, score} to dict
            scores = {r["label"]: r["score"] for r in results}
            positive = scores.get("positive", 0.0)
            negative = scores.get("negative", 0.0)
            neutral  = scores.get("neutral", 0.0)

            # Determine dominant label
            label = max(scores, key=scores.get)

            # Signal: +1 bullish, -1 bearish, 0 neutral
            signal = (
                1  if label == "positive" else
                -1 if label == "negative" else
                0
            )

            return {
                "label":    label,
                "positive": round(positive, 4),
                "negative": round(negative, 4),
                "neutral":  round(neutral, 4),
                "signal":   signal,
            }

        except Exception as e:
            log.warning(f"Scoring failed for text: '{text[:50]}...' — {e}")
            return {
                "label": "neutral",
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "signal": 0,
            }

    def score_ticker(
        self,
        ticker: str,
        articles: list[dict],
        use_cache: bool = True,
        filter_neutral: bool = True,
    ) -> list[dict]:
        """
        Score all headlines for one ticker.

        Args:
            ticker:         Ticker symbol
            articles:       List of article dicts from NewsFetcher
            use_cache:      Use cached scores if available
            filter_neutral: Remove neutral headlines (recommended)
                           Research: neutral headlines degrade accuracy

        Returns:
            List of scored article dicts with added sentiment fields
        """
        if not articles:
            log.warning(f"[{ticker}] No articles to score")
            return []

        today = datetime.utcnow().strftime("%Y-%m-%d")
        cache_file = _score_cache_path(ticker, today)

        if use_cache and _is_cache_fresh(cache_file):
            log.debug(f"[{ticker}] Scores: loaded from cache")
            with open(cache_file) as f:
                scored = json.load(f)
            if filter_neutral:
                scored = [s for s in scored if s["label"] != "neutral"]
            return scored

        log.info(f"[{ticker}] Scoring {len(articles)} headlines with FinBERT...")
        self._load_model()

        scored = []
        for article in articles:
            # Score title + description combined for better context
            text = article["title"]
            if article.get("description"):
                text = text + ". " + article["description"]

            sentiment = self.score_headline(text)

            scored.append({
                **article,
                **sentiment,
            })

        # Cache all scores (including neutral — filter happens after load)
        with open(cache_file, "w") as f:
            json.dump(scored, f, indent=2)

        # Count results
        positive = sum(1 for s in scored if s["label"] == "positive")
        negative = sum(1 for s in scored if s["label"] == "negative")
        neutral  = sum(1 for s in scored if s["label"] == "neutral")

        log.info(
            f"[{ticker}] Scored: "
            f"{positive} bullish, {negative} bearish, {neutral} neutral"
        )

        if filter_neutral:
            scored = [s for s in scored if s["label"] != "neutral"]
            log.debug(
                f"[{ticker}] After neutral filter: {len(scored)} signals kept"
            )

        return scored

    def score_all_tickers(
        self,
        all_articles: dict[str, list[dict]],
        use_cache: bool = True,
        filter_neutral: bool = True,
    ) -> dict[str, list[dict]]:
        """
        Score headlines for all tickers.
        Returns dict of {ticker: [scored_articles]}
        """
        results: dict[str, list[dict]] = {}

        for ticker, articles in all_articles.items():
            try:
                results[ticker] = self.score_ticker(
                    ticker,
                    articles,
                    use_cache=use_cache,
                    filter_neutral=filter_neutral,
                )
            except Exception as e:
                log.error(f"[{ticker}] Scoring failed: {e}")
                results[ticker] = []

        # Summary
        total_signals = sum(len(v) for v in results.values())
        bullish = sum(
            sum(1 for s in v if s["signal"] == 1)
            for v in results.values()
        )
        bearish = sum(
            sum(1 for s in s_list if s["signal"] == -1)
            for s_list in results.values()
        )

        log.info(
            f"Scoring complete: {total_signals} signals | "
            f"{bullish} bullish, {bearish} bearish"
        )
        return results