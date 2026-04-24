"""
src/models/cnn_model.py
────────────────────────
CNN model that reads candlestick chart images to predict
stock direction.

Architecture:
  Uses pretrained ResNet18 (trained on ImageNet).
  We replace the final layer to output 1 probability.
  Fine-tune only the last 2 layers on our chart images.
  This is called "transfer learning" — ResNet already knows
  how to detect edges, shapes, and patterns from ImageNet.
  We teach it to apply that knowledge to candlestick charts.

Why transfer learning works for charts:
  - Candlestick patterns are visual shapes (like ImageNet objects)
  - ResNet can detect: wicks (lines), bodies (rectangles),
    gaps (white space), volume spikes (tall bars)
  - Fine-tuning takes 10-20 minutes vs training from scratch
    which would take hours

Research:
  AlexNet/VGG on chart images: 60-65% accuracy (2020)
  ResNet18 fine-tuned: 65-70% accuracy (2023)
  ViT (Vision Transformer): 68-73% accuracy (2024)
  We use ResNet18 — best speed/accuracy tradeoff on CPU
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseStockModel, MODELS_DIR
from src.utils.logger import log

try:
    import torchvision.models as tv_models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# ─── ResNet18 for Chart Images ────────────────────────────────

class ChartCNN(nn.Module):
    """
    ResNet18 fine-tuned for candlestick chart classification.

    Input:  (batch, 3, 224, 224) — RGB chart images
    Output: bullish probability (0.0 to 1.0)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("Run: pip install torchvision")

        # Load pretrained ResNet18
        self.resnet = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Freeze early layers — keep ImageNet features
        # Only train the last layer block + classifier
        for name, param in self.resnet.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Replace final layer: 512 → 1 (binary classification)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x).squeeze(1)


# ─── CNN Model Wrapper ────────────────────────────────────────

class CNNModel(BaseStockModel):
    """
    CNN classifier using candlestick chart images.

    Unlike XGBoost (reads numbers) and LSTM (reads sequences),
    the CNN reads the chart the same way a human trader does.

    Usage:
        from src.models.chart_generator import ChartGenerator
        gen = ChartGenerator()
        images, labels, dates = gen.generate_dataset("AAPL", df)

        cnn = CNNModel("AAPL")
        cnn.train_on_images(images, labels)
        metrics = cnn.evaluate_on_images(images, labels, dates)
    """

    def __init__(
        self,
        ticker: str,
        target_days: int = 5,
        test_size: float = 0.2,
        dropout: float = 0.3,
        learning_rate: float = 0.0001,
        batch_size: int = 32,
        max_epochs: int = 30,
        patience: int = 8,
    ):
        super().__init__(
            model_name="CNN",
            ticker=ticker,
            target_days=target_days,
            test_size=test_size,
        )
        self.dropout       = dropout
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.max_epochs    = max_epochs
        self.patience      = patience
        self.network       = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        log.info(f"[{ticker}] CNN will train on: {self.device}")

    # ── Training on images ────────────────────────────────────

    def train_on_images(
        self,
        images: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Train CNN on pre-generated chart images.

        Args:
            images: shape (N, 3, 224, 224) float32
            labels: shape (N,) int64
        """
        self.network = ChartCNN(dropout=self.dropout).to(self.device)

        optimizer = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad,
                self.network.parameters()
            ),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        criterion = nn.BCELoss()

        # Time-based split
        split = int(len(images) * (1 - self.test_size))
        val_split = int(split * 0.85)

        X_train = torch.FloatTensor(images[:val_split])
        y_train = torch.FloatTensor(labels[:val_split])
        X_val   = torch.FloatTensor(images[val_split:split])
        y_val   = torch.FloatTensor(labels[val_split:split])

        train_ds     = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        best_val_loss  = float("inf")
        patience_count = 0
        best_weights   = None

        log.info(
            f"[{self.ticker}] Training CNN: "
            f"{len(X_train)} train images, "
            f"{len(X_val)} val images, "
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
                optimizer.step()
                train_loss += loss.item()

            self.network.eval()
            with torch.no_grad():
                val_preds = self.network(X_val)
                val_loss  = criterion(val_preds, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                best_weights   = {
                    k: v.clone()
                    for k, v in self.network.state_dict().items()
                }
            else:
                patience_count += 1

            if epoch % 5 == 0:
                log.info(
                    f"[{self.ticker}] CNN Epoch {epoch}: "
                    f"train={train_loss/len(train_loader):.4f} "
                    f"val={val_loss:.4f}"
                )

            if patience_count >= self.patience:
                log.info(
                    f"[{self.ticker}] CNN early stopping "
                    f"at epoch {epoch}"
                )
                break

        if best_weights:
            self.network.load_state_dict(best_weights)
        log.info(f"[{self.ticker}] CNN training complete")

    def evaluate_on_images(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        dates: list,
    ) -> dict:
        """Evaluate CNN on held-out test images."""
        from sklearn.metrics import accuracy_score, roc_auc_score

        split    = int(len(images) * (1 - self.test_size))
        X_test   = torch.FloatTensor(images[split:]).to(self.device)
        y_test   = labels[split:]

        self.network.eval()
        with torch.no_grad():
            probs = self.network(X_test).cpu().numpy()

        preds    = (probs > 0.5).astype(int)
        accuracy = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = 0.5

        metrics = {
            "model":        "CNN",
            "ticker":       self.ticker,
            "accuracy":     round(accuracy, 4),
            "auc_roc":      round(auc, 4),
            "test_samples": len(y_test),
        }

        log.info(
            f"[{self.ticker}] CNN — "
            f"Accuracy: {accuracy:.1%} | AUC: {auc:.3f} | "
            f"Test samples: {len(y_test)}"
        )
        return metrics

    def predict_proba_image(
        self,
        image: np.ndarray,
    ) -> float:
        """
        Predict bullish probability from one chart image.
        image shape: (3, 224, 224)
        """
        self.network.eval()
        with torch.no_grad():
            x = torch.FloatTensor(image).unsqueeze(0).to(self.device)
            prob = self.network(x).item()
        return prob

    # ── These override base class but CNN uses image interface ─

    def train(self, X_train, y_train):
        pass  # use train_on_images instead

    def predict_proba(self, X):
        return np.array([0.5])  # placeholder

    # ── Save / Load ───────────────────────────────────────────

    def save(self) -> Path:
        path = MODELS_DIR / f"cnn_{self.ticker}.pt"
        torch.save({
            "network_state": self.network.state_dict(),
            "ticker":        self.ticker,
            "target_days":   self.target_days,
            "dropout":       self.dropout,
        }, path)
        log.info(f"[{self.ticker}] CNN saved → {path.name}")
        return path

    def load(self, path: Path = None) -> None:
        path = path or MODELS_DIR / f"cnn_{self.ticker}.pt"
        data = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self.network = ChartCNN(
            dropout=data.get("dropout", 0.3)
        ).to(self.device)
        self.network.load_state_dict(data["network_state"])
        self.ticker      = data["ticker"]
        self.target_days = data["target_days"]
        self.is_trained  = True
        log.info(f"[{self.ticker}] CNN loaded from {path.name}")