"""
src/ensemble/signal_engine.py
───────────────────────────────
Combines XGBoost + LSTM + CNN predictions into a single
Bullish / Neutral / Bearish trading signal with confidence.

Three specialist models:
  XGBoost — reads 117 numeric features (tabular patterns)
  LSTM    — reads 60-day sequences (temporal patterns)
  CNN     — reads 30-day chart images (visual patterns)

Dynamic weighting:
  Models weighted by AUC score from training.
  Better model gets more influence on final signal.

Signal thresholds:
  probability > 0.55  → BULLISH
  probability < 0.45  → BEARISH
  otherwise           → NEUTRAL

Confidence filtering:
  Only act when models AGREE on direction.
  Disagreement → NEUTRAL (no trade).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.cnn_model import CNNModel
from src.models.chart_generator import ChartGenerator
from src.models.base_model import MODELS_DIR
from src.utils.logger import log

RESULTS_FILE = MODELS_DIR / "ensemble_results.json"

BULLISH_THRESHOLD = 0.55
BEARISH_THRESHOLD = 0.45


@dataclass
class TradingSignal:
    ticker:        str
    signal:        str
    confidence:    float
    xgb_prob:      float
    lstm_prob:     float
    cnn_prob:      float
    ensemble_prob: float
    agreement:     bool
    xgb_weight:    float
    lstm_weight:   float
    cnn_weight:    float

    def __str__(self) -> str:
        emoji = (
            "🟢" if self.signal == "BULLISH" else
            "🔴" if self.signal == "BEARISH" else
            "⚪"
        )
        agree = "✓ agree" if self.agreement else "✗ disagree"
        return (
            f"{emoji} {self.ticker:8s} {self.signal:8s} "
            f"confidence={self.confidence:.1%} | "
            f"xgb={self.xgb_prob:.3f} "
            f"lstm={self.lstm_prob:.3f} "
            f"cnn={self.cnn_prob:.3f} "
            f"ensemble={self.ensemble_prob:.3f} | {agree}"
        )


class SignalEngine:
    """
    Loads XGBoost + LSTM + CNN models and generates
    ensemble trading signals.
    """

    def __init__(self):
        self.xgb_models:  dict[str, XGBoostModel] = {}
        self.lstm_models: dict[str, LSTMModel]    = {}
        self.cnn_models:  dict[str, CNNModel]      = {}
        self.weights:     dict[str, dict]          = {}
        self.chart_gen    = None
        self._load_weights()

    def _load_weights(self) -> None:
        """Load AUC-based dynamic weights for all models."""
        if not RESULTS_FILE.exists():
            log.warning("No ensemble_results.json found.")
            return

        with open(RESULTS_FILE) as f:
            results = json.load(f)

        for ticker, model_results in results.items():
            xgb_auc  = model_results.get(
                "xgboost", {}
            ).get("auc_roc", 0.5)
            lstm_auc = model_results.get(
                "lstm", {}
            ).get("auc_roc", 0.5)
            cnn_auc  = model_results.get(
                "cnn", {}
            ).get("auc_roc", 0.5)

            total = xgb_auc + lstm_auc + cnn_auc
            if total > 0:
                xgb_w  = xgb_auc  / total
                lstm_w = lstm_auc / total
                cnn_w  = cnn_auc  / total
            else:
                xgb_w = lstm_w = cnn_w = 1/3

            self.weights[ticker] = {
                "xgboost": round(xgb_w,  4),
                "lstm":    round(lstm_w, 4),
                "cnn":     round(cnn_w,  4),
            }

        log.info(
            f"Dynamic weights loaded for "
            f"{len(self.weights)} tickers"
        )

    def load_models(self, tickers: list[str] = None) -> None:
        """Load XGBoost + LSTM + CNN models from disk."""
        tickers = tickers or settings.all_tickers

        for ticker in tickers:
            xgb_path  = MODELS_DIR / f"xgboost_{ticker}.joblib"
            lstm_path = MODELS_DIR / f"lstm_{ticker}.pt"
            cnn_path  = MODELS_DIR / f"cnn_{ticker}.pt"

            if xgb_path.exists():
                try:
                    xgb = XGBoostModel(ticker)
                    xgb.load(xgb_path)
                    self.xgb_models[ticker] = xgb
                except Exception as e:
                    log.error(f"[{ticker}] XGBoost load: {e}")

            if lstm_path.exists():
                try:
                    lstm = LSTMModel(ticker)
                    lstm.load(lstm_path)
                    self.lstm_models[ticker] = lstm
                except Exception as e:
                    log.error(f"[{ticker}] LSTM load: {e}")

            if cnn_path.exists():
                try:
                    cnn = CNNModel(ticker)
                    cnn.load(cnn_path)
                    self.cnn_models[ticker] = cnn
                except Exception as e:
                    log.error(f"[{ticker}] CNN load: {e}")

            w = self.weights.get(
                ticker,
                {"xgboost": 0.34, "lstm": 0.33, "cnn": 0.33}
            )
            models_loaded = []
            if ticker in self.xgb_models:
                models_loaded.append("XGBoost")
            if ticker in self.lstm_models:
                models_loaded.append("LSTM")
            if ticker in self.cnn_models:
                models_loaded.append("CNN")

            if models_loaded:
                log.info(
                    f"[{ticker}] Loaded: {', '.join(models_loaded)} | "
                    f"weights xgb={w['xgboost']:.2f} "
                    f"lstm={w['lstm']:.2f} "
                    f"cnn={w['cnn']:.2f}"
                )

    def _get_latest_features(
        self,
        ticker: str,
        sequence_len: int = 60,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load latest features for XGBoost, LSTM, and CNN.

        Returns:
            xgb_X:   (1, n_features)
            lstm_X:  (1, 60, n_features)
            cnn_X:   (3, 224, 224) or None
        """
        data_path = (
            settings.data_processed_path /
            f"{ticker}_features_with_sentiment.parquet"
        )
        if not data_path.exists():
            raise FileNotFoundError(
                f"No feature file for {ticker}"
            )

        df = pd.read_parquet(data_path)

        exclude = {
            "open", "high", "low", "close",
            "volume", "ticker",
        }

        # ── XGBoost features ──────────────────────────────────
        xgb_model = self.xgb_models.get(ticker)
        feat_cols  = (
            xgb_model.feature_cols
            if xgb_model and xgb_model.feature_cols
            else [
                c for c in df.columns
                if c not in exclude
                and df[c].dtype in [
                    np.float64, np.float32,
                    np.int64, np.int32
                ]
            ]
        )

        df_clean   = df.dropna(subset=feat_cols)
        latest_row = df_clean[feat_cols].iloc[-1:].values
        xgb_X      = (
            xgb_model.scaler.transform(latest_row)
            if xgb_model and xgb_model.scaler
            else latest_row
        )

        # ── LSTM sequence ─────────────────────────────────────
        lstm_model = self.lstm_models.get(ticker)
        lstm_cols  = (
            lstm_model.feature_cols
            if lstm_model and lstm_model.feature_cols
            else feat_cols
        )
        df_lstm    = df.dropna(subset=lstm_cols)
        seq_data   = df_lstm[lstm_cols].iloc[-sequence_len:].values
        seq_scaled = (
            lstm_model.scaler.transform(seq_data)
            if lstm_model and lstm_model.scaler
            else seq_data
        )
        lstm_X = seq_scaled[np.newaxis, :, :].astype(np.float32)

        # ── CNN chart image ───────────────────────────────────
        cnn_X = None
        cnn_model = self.cnn_models.get(ticker)
        if cnn_model:
            try:
                if self.chart_gen is None:
                    self.chart_gen = ChartGenerator(window_days=30)
                window_df = df[
                    ["open", "high", "low", "close", "volume"]
                ].dropna().iloc[-30:]
                img = self.chart_gen.ohlcv_to_image(window_df)
                cnn_X = img.transpose(2, 0, 1)  # (3, 224, 224)
            except Exception as e:
                log.warning(f"[{ticker}] CNN image failed: {e}")

        return xgb_X.astype(np.float32), lstm_X, cnn_X

    def generate_signal(self, ticker: str) -> TradingSignal:
        """Generate ensemble signal for one ticker."""
        xgb_prob  = 0.5
        lstm_prob = 0.5
        cnn_prob  = 0.5
        has_xgb   = ticker in self.xgb_models
        has_lstm  = ticker in self.lstm_models
        has_cnn   = ticker in self.cnn_models

        if not has_xgb and not has_lstm and not has_cnn:
            return TradingSignal(
                ticker=ticker, signal="NEUTRAL",
                confidence=0.0, xgb_prob=0.5,
                lstm_prob=0.5, cnn_prob=0.5,
                ensemble_prob=0.5, agreement=False,
                xgb_weight=0.34, lstm_weight=0.33,
                cnn_weight=0.33,
            )

        try:
            xgb_X, lstm_X, cnn_X = self._get_latest_features(ticker)
        except Exception as e:
            log.error(f"[{ticker}] Feature load failed: {e}")
            return TradingSignal(
                ticker=ticker, signal="NEUTRAL",
                confidence=0.0, xgb_prob=0.5,
                lstm_prob=0.5, cnn_prob=0.5,
                ensemble_prob=0.5, agreement=False,
                xgb_weight=0.34, lstm_weight=0.33,
                cnn_weight=0.33,
            )

        if has_xgb:
            try:
                xgb_prob = float(
                    self.xgb_models[ticker].predict_proba(xgb_X)[0]
                )
            except Exception as e:
                log.warning(f"[{ticker}] XGBoost predict: {e}")
                has_xgb = False

        if has_lstm:
            try:
                lstm_prob = float(
                    self.lstm_models[ticker].predict_proba(lstm_X)[0]
                )
            except Exception as e:
                log.warning(f"[{ticker}] LSTM predict: {e}")
                has_lstm = False

        if has_cnn and cnn_X is not None:
            try:
                cnn_prob = self.cnn_models[ticker].predict_proba_image(
                    cnn_X
                )
            except Exception as e:
                log.warning(f"[{ticker}] CNN predict: {e}")
                has_cnn = False

        # Dynamic weights
        w = self.weights.get(
            ticker,
            {"xgboost": 0.34, "lstm": 0.33, "cnn": 0.33}
        )
        xgb_w  = w["xgboost"] if has_xgb  else 0.0
        lstm_w = w["lstm"]    if has_lstm else 0.0
        cnn_w  = w["cnn"]     if has_cnn  else 0.0

        total_w = xgb_w + lstm_w + cnn_w
        if total_w > 0:
            xgb_w  /= total_w
            lstm_w /= total_w
            cnn_w  /= total_w

        ensemble_prob = (
            xgb_w  * xgb_prob  +
            lstm_w * lstm_prob +
            cnn_w  * cnn_prob
        )

        # Agreement — all available models point same direction
        directions = []
        if has_xgb:
            directions.append("up" if xgb_prob  > 0.5 else "down")
        if has_lstm:
            directions.append("up" if lstm_prob > 0.5 else "down")
        if has_cnn:
            directions.append("up" if cnn_prob  > 0.5 else "down")

        agreement = len(set(directions)) == 1

        if ensemble_prob > BULLISH_THRESHOLD and agreement:
            signal     = "BULLISH"
            confidence = (ensemble_prob - 0.5) * 2
        elif ensemble_prob < BEARISH_THRESHOLD and agreement:
            signal     = "BEARISH"
            confidence = (0.5 - ensemble_prob) * 2
        else:
            signal     = "NEUTRAL"
            confidence = abs(ensemble_prob - 0.5) * 2

        confidence = round(min(confidence, 1.0), 4)

        return TradingSignal(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            xgb_prob=round(xgb_prob, 4),
            lstm_prob=round(lstm_prob, 4),
            cnn_prob=round(cnn_prob, 4),
            ensemble_prob=round(ensemble_prob, 4),
            agreement=agreement,
            xgb_weight=round(xgb_w, 4),
            lstm_weight=round(lstm_w, 4),
            cnn_weight=round(cnn_w, 4),
        )

    def generate_signals(
        self,
        tickers: list[str] = None,
    ) -> dict[str, TradingSignal]:
        """Generate signals for all tickers."""
        tickers = tickers or list(self.xgb_models.keys())
        signals = {}

        for ticker in tickers:
            try:
                signals[ticker] = self.generate_signal(ticker)
            except Exception as e:
                log.error(f"[{ticker}] Signal failed: {e}")

        return signals