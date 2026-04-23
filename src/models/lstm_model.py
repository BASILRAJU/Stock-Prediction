"""
src/models/lstm_model.py
─────────────────────────
LSTM model for stock direction prediction.

Reads 60-day sequences of 103 features to predict
whether price will be higher in target_days.

Research basis:
  PLSTM-AL (ScienceDirect 2025): LSTM + attention
  outperforms vanilla LSTM, CNN, GRU on 5 country indices.

Architecture:
  Input:  (batch, 60 days, 103 features)
  LSTM:   2 stacked layers, hidden_size=128, dropout=0.2
  Dense:  128 → 32 → 1 with sigmoid
  Output: bullish probability 0.0 to 1.0

Training:
  Adam optimizer, learning rate 0.0005
  Early stopping after 30 epochs of no improvement
  Max 150 epochs, batch size 32
  Gradient clipping to prevent exploding gradients
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseStockModel, MODELS_DIR
from src.utils.logger import log


# ─── LSTM Network ─────────────────────────────────────────────

class LSTMNetwork(nn.Module):
    """
    Two-layer LSTM with dropout for sequence prediction.
    Input shape:  (batch, sequence_len, features)
    Output:       bullish probability (0.0 to 1.0)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Stacked LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out    = out[:, -1, :]        # take last timestep
        out    = self.dropout(out)
        out    = self.relu(self.fc1(out))
        out    = self.sigmoid(self.fc2(out))
        return out.squeeze(1)


# ─── LSTM Model Wrapper ───────────────────────────────────────

class LSTMModel(BaseStockModel):
    """
    LSTM classifier for stock direction prediction.

    Key advantage over XGBoost:
      Reads temporal sequences.
      "RSI rose for 3 weeks then crossed 70" is a pattern
      XGBoost cannot see but LSTM catches naturally.
    """

    def __init__(
        self,
        ticker: str,
        target_days: int = 5,
        test_size: float = 0.2,
        sequence_len: int = 60,
        hidden_size: int = 128,
        dropout: float = 0.2,
        learning_rate: float = 0.0005,
        batch_size: int = 32,
        max_epochs: int = 150,
        patience: int = 30,
    ):
        super().__init__(
            model_name="LSTM",
            ticker=ticker,
            target_days=target_days,
            test_size=test_size,
        )
        self.sequence_len  = sequence_len
        self.hidden_size   = hidden_size
        self.dropout       = dropout
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.max_epochs    = max_epochs
        self.patience      = patience
        self.network       = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        log.info(f"[{ticker}] LSTM will train on: {self.device}")

    # ── Data Preparation ──────────────────────────────────────

    def prepare_data(
        self,
        df: pd.DataFrame,
    ) -> tuple:
        """
        Prepare SEQUENCE data for LSTM.
        Creates overlapping 60-day windows.

        Returns:
            X_train, X_test, y_train, y_test
            Shape: (samples, sequence_len, features)
        """
        df = df.copy()

        future_return = (
            df["close"].shift(-self.target_days) / df["close"] - 1
        )
        df["target"] = (future_return > 0).astype(int)
        df = df.dropna(subset=["target"])

        exclude = {
            "open", "high", "low", "close", "volume",
            "ticker", "target",
        }
        self.feature_cols = [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in [
                np.float64, np.float32, np.int64, np.int32
            ]
        ]
        df = df.dropna(subset=self.feature_cols)

        feature_data = df[self.feature_cols].values
        target_data  = df["target"].values

        # Fit scaler on training portion only
        split_idx = int(len(feature_data) * (1 - self.test_size))
        self.scaler.fit(feature_data[:split_idx])
        feature_scaled = self.scaler.transform(feature_data)

        # Build sequences
        X, y = [], []
        for i in range(self.sequence_len, len(feature_scaled)):
            X.append(feature_scaled[i - self.sequence_len:i])
            y.append(target_data[i])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        seq_split = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:seq_split], X[seq_split:]
        y_train, y_test = y[:seq_split], y[seq_split:]

        log.info(
            f"[{self.ticker}] LSTM sequences: "
            f"train={len(X_train)}, test={len(X_test)}, "
            f"shape=({self.sequence_len}, {len(self.feature_cols)})"
        )
        return X_train, X_test, y_train, y_test

    # ── Training ──────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Train LSTM with early stopping."""
        input_size = X_train.shape[2]

        self.network = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        criterion = nn.BCELoss()

        # Validation split
        val_size  = int(len(X_train) * 0.15)
        X_tr      = X_train[:-val_size]
        y_tr      = y_train[:-val_size]
        X_val     = X_train[-val_size:]
        y_val     = y_train[-val_size:]

        train_dataset = TensorDataset(
            torch.FloatTensor(X_tr),
            torch.FloatTensor(y_tr),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,          # never shuffle time series
        )

        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        best_val_loss  = float("inf")
        patience_count = 0
        best_weights   = None

        log.info(
            f"[{self.ticker}] Training LSTM: "
            f"{len(X_tr)} train, {len(X_val)} val, "
            f"max {self.max_epochs} epochs..."
        )

        for epoch in range(1, self.max_epochs + 1):
            self.network.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.network(X_batch)
                loss  = criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), 1.0
                )
                optimizer.step()
                train_loss += loss.item()

            self.network.eval()
            with torch.no_grad():
                val_preds = self.network(X_val_t)
                val_loss  = criterion(val_preds, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                best_weights   = {
                    k: v.clone()
                    for k, v in self.network.state_dict().items()
                }
            else:
                patience_count += 1

            if epoch % 10 == 0:
                log.info(
                    f"[{self.ticker}] Epoch {epoch}: "
                    f"train_loss={train_loss/len(train_loader):.4f}, "
                    f"val_loss={val_loss:.4f}"
                )

            if patience_count >= self.patience:
                log.info(
                    f"[{self.ticker}] Early stopping at epoch {epoch} "
                    f"(best val_loss={best_val_loss:.4f})"
                )
                break

        if best_weights:
            self.network.load_state_dict(best_weights)

        log.info(f"[{self.ticker}] LSTM training complete")

    # ── Prediction ────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns bullish probability for each sequence."""
        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            probs    = self.network(X_tensor)
        return probs.cpu().numpy()

    # ── Save / Load ───────────────────────────────────────────

    def save(self) -> Path:
        """Save trained LSTM to disk."""
        path = MODELS_DIR / f"lstm_{self.ticker}.pt"
        torch.save({
            "network_state":  self.network.state_dict(),
            "network_config": {
                "input_size":  len(self.feature_cols),
                "hidden_size": self.hidden_size,
                "dropout":     self.dropout,
            },
            "scaler":       self.scaler,
            "feature_cols": self.feature_cols,
            "ticker":       self.ticker,
            "target_days":  self.target_days,
            "sequence_len": self.sequence_len,
        }, path)
        log.info(f"[{self.ticker}] LSTM saved → {path.name}")
        return path

    def load(self, path: Path = None) -> None:
        """Load trained LSTM from disk."""
        path = path or MODELS_DIR / f"lstm_{self.ticker}.pt"
        data = torch.load(
            path, map_location=self.device, weights_only=False
        )

        cfg = data["network_config"]
        self.network = LSTMNetwork(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            dropout=cfg["dropout"],
        ).to(self.device)
        self.network.load_state_dict(data["network_state"])

        self.scaler       = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.ticker       = data["ticker"]
        self.target_days  = data["target_days"]
        self.sequence_len = data["sequence_len"]
        self.is_trained   = True
        log.info(f"[{self.ticker}] LSTM loaded from {path.name}")