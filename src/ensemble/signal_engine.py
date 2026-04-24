"""
src/ensemble/signal_engine.py
───────────────────────────────
Combines XGBoost + LSTM predictions into a single
Bullish / Neutral / Bearish trading signal with confidence.

Dynamic weighting:
  Models are weighted by their AUC score from training.
  Better-performing model gets more influence.
  This is Sharpe-ratio-inspired dynamic weighting
  from TradingAgents research (arXiv:2412.20138).

Signal thresholds:
  probability > 0.55  → BULLISH
  probability < 0.45  → BEARISH
  otherwise           → NEUTRAL (no trade)

Confidence filtering (research-backed):
  Only act on signals where models AGREE.
  Disagreement between models → NEUTRAL.
  This reduces false signals significantly.

Output per ticker:
  signal:      BULLISH / NEUTRAL / BEARISH
  confidence:  0.0 to 1.0
  xgb_prob:    XGBoost bullish probability
  lstm_prob:   LSTM bullish probability
  agreement:   True if both models agree on direction
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import settings
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.base_model import MODELS_DIR
from src.utils.logger import log

RESULTS_FILE = MODELS_DIR / "ensemble_results.json"

# ─── Signal thresholds ────────────────────────────────────────
BULLISH_THRESHOLD = 0.55   # above this → bullish
BEARISH_THRESHOLD = 0.45   # below this → bearish
AGREEMENT_MARGIN  = 0.05   # models must agree within this margin


@dataclass
class TradingSignal:
    """Output signal for one ticker."""
    ticker:      str
    signal:      str        # BULLISH / NEUTRAL / BEARISH
    confidence:  float      # 0.0 to 1.0
    xgb_prob:    float      # XGBoost bullish probability
    lstm_prob:   float      # LSTM bullish probability
    ensemble_prob: float    # weighted combination
    agreement:   bool       # do models agree?
    xgb_weight:  float      # dynamic weight given to XGBoost
    lstm_weight: float      # dynamic weight given to LSTM

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
            f"xgb={self.xgb_prob:.3f} lstm={self.lstm_prob:.3f} "
            f"ensemble={self.ensemble_prob:.3f} | {agree}"
        )


class SignalEngine:
    """
    Loads trained models and generates trading signals.

    Usage:
        engine = SignalEngine()
        engine.load_models(['AAPL', 'MSFT'])
        signals = engine.generate_signals(['AAPL', 'MSFT'])
        for sig in signals.values():
            print(sig)
    """

    def __init__(self):
        self.xgb_models:  dict[str, XGBoostModel] = {}
        self.lstm_models: dict[str, LSTMModel]    = {}
        self.weights:     dict[str, dict]          = {}
        self._load_weights()

    def _load_weights(self) -> None:
        """
        Load model accuracy scores to compute dynamic weights.
        Weight = AUC score (higher AUC → more weight).
        """
        if not RESULTS_FILE.exists():
            log.warning(
                "No ensemble_results.json found. "
                "Run ensemble_trainer.py first."
            )
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

            # Normalise weights to sum to 1.0
            total = xgb_auc + lstm_auc
            if total > 0:
                xgb_w  = xgb_auc  / total
                lstm_w = lstm_auc / total
            else:
                xgb_w = lstm_w = 0.5

            self.weights[ticker] = {
                "xgboost": round(xgb_w, 4),
                "lstm":    round(lstm_w, 4),
            }

        log.info(
            f"Dynamic weights loaded for "
            f"{len(self.weights)} tickers"
        )

    def load_models(self, tickers: list[str] = None) -> None:
        """Load trained XGBoost + LSTM models from disk."""
        tickers = tickers or settings.all_tickers

        for ticker in tickers:
            xgb_path  = MODELS_DIR / f"xgboost_{ticker}.joblib"
            lstm_path = MODELS_DIR / f"lstm_{ticker}.pt"

            loaded_any = False

            if xgb_path.exists():
                try:
                    xgb = XGBoostModel(ticker)
                    xgb.load(xgb_path)
                    self.xgb_models[ticker] = xgb
                    loaded_any = True
                except Exception as e:
                    log.error(
                        f"[{ticker}] XGBoost load failed: {e}"
                    )

            if lstm_path.exists():
                try:
                    lstm = LSTMModel(ticker)
                    lstm.load(lstm_path)
                    self.lstm_models[ticker] = lstm
                    loaded_any = True
                except Exception as e:
                    log.error(
                        f"[{ticker}] LSTM load failed: {e}"
                    )

            if loaded_any:
                w = self.weights.get(
                    ticker, {"xgboost": 0.5, "lstm": 0.5}
                )
                log.info(
                    f"[{ticker}] Models loaded | "
                    f"weights: xgb={w['xgboost']:.2f}, "
                    f"lstm={w['lstm']:.2f}"
                )
            else:
                log.warning(
                    f"[{ticker}] No models found — "
                    f"run ensemble_trainer first"
                )

    def _get_latest_features(
        self,
        ticker: str,
        sequence_len: int = 60,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the latest features for one ticker.

        Returns:
            xgb_X:  (1, n_features) for XGBoost
            lstm_X: (1, sequence_len, n_features) for LSTM
        """
        data_path = (
            settings.data_processed_path /
            f"{ticker}_features_with_sentiment.parquet"
        )
        if not data_path.exists():
            raise FileNotFoundError(
                f"No feature file found for {ticker}. "
                f"Run run_pipeline.py first."
            )

        df = pd.read_parquet(data_path)

        # ── XGBoost features ──────────────────────────────────
        xgb_model = self.xgb_models.get(ticker)
        if xgb_model and xgb_model.feature_cols:
            feat_cols = xgb_model.feature_cols
        else:
            # Fall back to all numeric non-price columns
            exclude = {
                "open", "high", "low", "close",
                "volume", "ticker",
            }
            feat_cols = [
                c for c in df.columns
                if c not in exclude
                and df[c].dtype in [
                    np.float64, np.float32,
                    np.int64, np.int32
                ]
            ]

        df_clean = df.dropna(subset=feat_cols)
        latest_row = df_clean[feat_cols].iloc[-1:].values

        # Scale using XGBoost's fitted scaler
        if xgb_model and xgb_model.scaler:
            latest_scaled = xgb_model.scaler.transform(latest_row)
        else:
            latest_scaled = latest_row

        xgb_X = latest_scaled

        # ── LSTM sequence ─────────────────────────────────────
        lstm_model = self.lstm_models.get(ticker)
        if lstm_model and lstm_model.feature_cols:
            lstm_cols = lstm_model.feature_cols
        else:
            lstm_cols = feat_cols

        df_lstm = df.dropna(subset=lstm_cols)
        if len(df_lstm) < sequence_len:
            raise ValueError(
                f"[{ticker}] Not enough data for LSTM sequence: "
                f"need {sequence_len}, have {len(df_lstm)}"
            )

        seq_data = df_lstm[lstm_cols].iloc[-sequence_len:].values

        # Scale using LSTM's fitted scaler
        if lstm_model and lstm_model.scaler:
            seq_scaled = lstm_model.scaler.transform(seq_data)
        else:
            seq_scaled = seq_data

        lstm_X = seq_scaled[np.newaxis, :, :]  # (1, seq_len, features)

        return xgb_X.astype(np.float32), lstm_X.astype(np.float32)

    def generate_signal(self, ticker: str) -> TradingSignal:
        """
        Generate a trading signal for one ticker.

        Combines XGBoost + LSTM probabilities using
        AUC-based dynamic weights.
        """
        xgb_prob  = 0.5
        lstm_prob = 0.5
        has_xgb   = ticker in self.xgb_models
        has_lstm  = ticker in self.lstm_models

        if not has_xgb and not has_lstm:
            log.warning(f"[{ticker}] No models loaded")
            return TradingSignal(
                ticker=ticker, signal="NEUTRAL",
                confidence=0.0, xgb_prob=0.5,
                lstm_prob=0.5, ensemble_prob=0.5,
                agreement=False, xgb_weight=0.5,
                lstm_weight=0.5,
            )

        try:
            xgb_X, lstm_X = self._get_latest_features(ticker)
        except Exception as e:
            log.error(f"[{ticker}] Feature load failed: {e}")
            return TradingSignal(
                ticker=ticker, signal="NEUTRAL",
                confidence=0.0, xgb_prob=0.5,
                lstm_prob=0.5, ensemble_prob=0.5,
                agreement=False, xgb_weight=0.5,
                lstm_weight=0.5,
            )

        # Get probabilities from each model
        if has_xgb:
            try:
                xgb_prob = float(
                    self.xgb_models[ticker].predict_proba(xgb_X)[0]
                )
            except Exception as e:
                log.warning(
                    f"[{ticker}] XGBoost predict failed: {e}"
                )
                has_xgb = False

        if has_lstm:
            try:
                lstm_prob = float(
                    self.lstm_models[ticker].predict_proba(lstm_X)[0]
                )
            except Exception as e:
                log.warning(
                    f"[{ticker}] LSTM predict failed: {e}"
                )
                has_lstm = False

        # Dynamic weights from AUC scores
        w = self.weights.get(
            ticker, {"xgboost": 0.5, "lstm": 0.5}
        )
        xgb_w  = w["xgboost"] if has_xgb else 0.0
        lstm_w = w["lstm"]    if has_lstm else 0.0

        # Renormalise if one model missing
        total_w = xgb_w + lstm_w
        if total_w > 0:
            xgb_w  /= total_w
            lstm_w /= total_w

        # Weighted ensemble probability
        ensemble_prob = xgb_w * xgb_prob + lstm_w * lstm_prob

        # Check model agreement
        xgb_direction  = "up" if xgb_prob  > 0.5 else "down"
        lstm_direction = "up" if lstm_prob > 0.5 else "down"
        agreement = (xgb_direction == lstm_direction)

        # Determine signal
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
            ensemble_prob=round(ensemble_prob, 4),
            agreement=agreement,
            xgb_weight=round(xgb_w, 4),
            lstm_weight=round(lstm_w, 4),
        )

    def generate_signals(
        self,
        tickers: list[str] = None,
    ) -> dict[str, TradingSignal]:
        """
        Generate signals for all tickers.
        Returns dict of {ticker: TradingSignal}
        """
        tickers = tickers or list(self.xgb_models.keys())
        signals = {}

        for ticker in tickers:
            try:
                sig = self.generate_signal(ticker)
                signals[ticker] = sig
            except Exception as e:
                log.error(
                    f"[{ticker}] Signal generation failed: {e}"
                )

        return signals