"""
src/models/xgboost_model.py
────────────────────────────
XGBoost model for stock direction prediction.

Why XGBoost first:
  - Trains in under 2 minutes on CPU
  - Handles the 103 tabular features perfectly
  - Gives feature importance — tells us which indicators matter most
  - Research (Bocconi 2025): price features >60% importance,
    RSI/BB ~15%, sentiment adds meaningful signal

Hyperparameters are tuned for financial time series:
  - Conservative learning rate (0.05) to avoid overfitting
  - Max depth 4 — shallow trees generalise better on noisy data
  - Subsample 0.8 — prevents memorising training data
  - Early stopping — stops when validation score stops improving
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from src.models.base_model import BaseStockModel, MODELS_DIR
from src.utils.logger import log


class XGBoostModel(BaseStockModel):
    """
    XGBoost classifier for stock direction prediction.

    Predicts: will price be higher in target_days?
    Output: probability 0.0 (bearish) to 1.0 (bullish)
    """

    def __init__(
        self,
        ticker: str,
        target_days: int = 5,
        test_size: float = 0.2,
    ):
        super().__init__(
            model_name="XGBoost",
            ticker=ticker,
            target_days=target_days,
            test_size=test_size,
        )

        # XGBoost hyperparameters
        self.model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,        # use all CPU cores
            verbosity=0,
        )

    # ── Training ──────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Train XGBoost with early stopping.
        Uses last 15% of training data as validation set.
        """
        log.info(
            f"[{self.ticker}] Training XGBoost on "
            f"{len(X_train)} samples, "
            f"{X_train.shape[1]} features..."
        )

        # Validation split from end of training data
        val_size  = int(len(X_train) * 0.15)
        X_tr      = X_train[:-val_size]
        y_tr      = y_train[:-val_size]
        X_val     = X_train[-val_size:]
        y_val     = y_train[-val_size:]

        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = self.model.best_iteration
        log.info(
            f"[{self.ticker}] XGBoost trained — "
            f"best iteration: {best_iter}"
        )

    # ── Prediction ────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns bullish probability for each sample."""
        return self.model.predict_proba(X)[:, 1]

    # ── Feature Importance ────────────────────────────────────

    def get_feature_importance(
        self,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Returns the top N most important features.
        This tells us which indicators the model relies on most.
        """
        if not self.feature_cols:
            raise RuntimeError("Model not trained yet")

        importance = self.model.feature_importances_
        df = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": importance,
        })
        df = df.sort_values("importance", ascending=False)
        df = df.head(top_n).reset_index(drop=True)
        df["importance"] = df["importance"].round(4)
        return df

    # ── Save / Load ───────────────────────────────────────────

    def save(self) -> Path:
        """Save trained model to disk."""
        path = MODELS_DIR / f"xgboost_{self.ticker}.joblib"
        joblib.dump({
            "model":        self.model,
            "scaler":       self.scaler,
            "feature_cols": self.feature_cols,
            "ticker":       self.ticker,
            "target_days":  self.target_days,
        }, path)
        log.info(f"[{self.ticker}] XGBoost saved → {path.name}")
        return path

    def load(self, path: Path = None) -> None:
        """Load trained model from disk."""
        path = path or MODELS_DIR / f"xgboost_{self.ticker}.joblib"
        data = joblib.load(path)
        self.model        = data["model"]
        self.scaler       = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.ticker       = data["ticker"]
        self.target_days  = data["target_days"]
        self.is_trained   = True
        log.info(f"[{self.ticker}] XGBoost loaded from {path.name}")