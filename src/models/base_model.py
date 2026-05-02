"""
src/models/base_model.py
─────────────────────────
Base class that all prediction models inherit from.

Defines the shared interface:
  - prepare_data()   splits features into train/test sets
  - train()          fits the model
  - predict()        returns probability scores
  - evaluate()       calculates accuracy metrics
  - save() / load()  persists model to disk

Target variable:
  Binary: will close be higher in target_days? (1=up, 0=down)

Train/Test split:
  TIME-BASED — never random (prevents look-ahead bias)
  Train: first 80% of dates
  Test:  last 20% (most recent data)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config.settings import settings
from src.utils.logger import log

# Where trained models are saved
MODELS_DIR = settings.data_processed_path.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class BaseStockModel:
    """
    Base class for all stock prediction models.

    All models predict: will this stock be higher in 5 days?
    Output: probability between 0.0 (bearish) and 1.0 (bullish)
    """

    def __init__(
        self,
        model_name: str,
        ticker: str,
        target_days: int = 5,
        test_size: float = 0.2,
    ):
        self.model_name   = model_name
        self.ticker       = ticker
        self.target_days  = target_days
        self.test_size    = test_size
        self.model        = None
        self.scaler       = StandardScaler()
        self.feature_cols: list[str] = []
        self.is_trained   = False

    # ── Data Preparation ──────────────────────────────────────

    def prepare_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare feature matrix X and target vector y.

        Target: 1 if close price is higher in target_days, else 0
        Split:  Time-based (no random shuffling)
        """
        df = df.copy()

        future_return = (
            df["close"].shift(-self.target_days) / df["close"] - 1
        )
        df["target"] = (future_return > 0).astype(int)
        df = df.dropna(subset=["target"])

        exclude = {
            # Price columns
            "open", "high", "low", "close", "volume",
            "ticker", "target",
            # Raw moving averages (use price_to_sma_X relative distances instead)
            "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_9", "ema_21", "ema_55",
            # Raw volume profile levels (use vp_dist_to_X relative distances instead)
            "vp_poc", "vp_val_high", "vp_val_low",
            # Raw volume/momentum aggregates (use relative versions)
            "obv_ema", "force_index_ema", "vol_sma_20",
        } 
        self.feature_cols = [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in [
                np.float64, np.float32, np.int64, np.int32
            ]
        ]

        df = df.dropna(subset=self.feature_cols)

        X = df[self.feature_cols].values
        y = df["target"].values

        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        log.info(
            f"[{self.ticker}] {self.model_name} data prepared: "
            f"train={len(X_train)}, test={len(X_test)}, "
            f"features={len(self.feature_cols)}, "
            f"target_days={self.target_days}"
        )

        return X_train, X_test, y_train, y_test

    # ── Evaluation ────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Evaluate model on test set.
        Returns dict of performance metrics.
        """
        model_ready = (
            self.model is not None
            or getattr(self, "network", None) is not None
        )
        if not model_ready:
            raise RuntimeError("Model not trained yet")

        y_pred  = self.predict_classes(X_test)
        y_proba = self.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = 0.5

        metrics = {
            "model":        self.model_name,
            "ticker":       self.ticker,
            "accuracy":     round(accuracy, 4),
            "auc_roc":      round(auc, 4),
            "test_samples": len(y_test),
            "bullish_pct":  round(float(y_test.mean()), 4),
        }

        log.info(
            f"[{self.ticker}] {self.model_name} — "
            f"Accuracy: {accuracy:.1%} | AUC: {auc:.3f}"
        )

        return metrics

    # ── Methods subclasses must implement ─────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of being bullish (0.0 to 1.0)"""
        raise NotImplementedError

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class (0 or 1)"""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def save(self) -> Path:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError

    # ── Convenience: full pipeline ────────────────────────────

    def fit_and_evaluate(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        One-call method:
        1. Prepare data
        2. Train model
        3. Evaluate on test set
        4. Return metrics
        """
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        self.train(X_train, y_train)
        metrics = self.evaluate(X_test, y_test)
        self.is_trained = True
        return metrics