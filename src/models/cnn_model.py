"""
src/models/cnn_model.py
────────────────────────
Pattern-aware CNN for stock direction prediction.

Architecture:
  ResNet18 backbone (ImageNet pretrained) →
    Branch 1: Direction prediction (will price go up?)
    Branch 2: Pattern prediction (which patterns present?)

Multi-task learning improves both tasks because:
  - Pattern detection forces the model to learn semantic features
  - Direction task benefits from this richer representation
  - Auxiliary patterns act as a regularizer

Training:
  Total loss = direction_loss + 0.3 * pattern_loss

Research:
  Multi-task CNN for finance: Sezer et al (2018-2024) consistently
  show 3-7% accuracy improvements over single-task training.
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


N_PATTERNS = 10  # matches pattern_detector.PATTERN_NAMES


# ─── Pattern-Aware CNN Architecture ───────────────────────────

class ChartCNN(nn.Module):
    """
    Multi-task ResNet18 for chart classification.

    Input:  (batch, 3, 224, 224) RGB chart images
    Output: (direction_prob, pattern_probs)
            direction: 0-1 bullish probability
            patterns: 10 binary flags
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        if not TORCHVISION_AVAILABLE:
            raise ImportError("Run: pip install torchvision")

        # Pretrained ResNet18 backbone
        self.resnet = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Freeze early layers, keep ImageNet features
        for name, param in self.resnet.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        in_features = self.resnet.fc.in_features

        # Replace classifier with shared trunk
        self.resnet.fc = nn.Identity()
        self.shared_trunk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Direction head (main task)
        self.direction_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Pattern head (auxiliary task) — multi-label
        self.pattern_head = nn.Sequential(
            nn.Linear(128, N_PATTERNS),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features  = self.resnet(x)
        shared    = self.shared_trunk(features)
        direction = self.direction_head(shared).squeeze(1)
        patterns  = self.pattern_head(shared)
        return direction, patterns


# ─── CNN Model Wrapper ────────────────────────────────────────

class CNNModel(BaseStockModel):
    """
    Pattern-aware CNN wrapper.

    Trains on chart images with both direction and pattern labels.
    Inference returns only direction probability.
    """

    def __init__(
        self,
        ticker: str,
        target_days: int   = 5,
        test_size:   float = 0.2,
        dropout:     float = 0.3,
        learning_rate: float = 0.0001,
        batch_size:    int   = 32,
        max_epochs:    int   = 30,
        patience:      int   = 8,
        pattern_weight: float = 0.3,
    ):
        super().__init__(
            model_name="CNN",
            ticker=ticker,
            target_days=target_days,
            test_size=test_size,
        )
        self.dropout        = dropout
        self.learning_rate  = learning_rate
        self.batch_size     = batch_size
        self.max_epochs     = max_epochs
        self.patience       = patience
        self.pattern_weight = pattern_weight
        self.network        = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        log.info(f"[{ticker}] CNN will train on: {self.device}")

    def train_on_images(
        self,
        images:   np.ndarray,
        labels:   np.ndarray,
        patterns: np.ndarray = None,
    ) -> None:
        """
        Train CNN on chart images with optional pattern labels.

        Args:
            images:   shape (N, 3, 224, 224) float32
            labels:   shape (N,) int64 — direction label
            patterns: shape (N, 10) float32 — pattern flags
                      If None, single-task training (no patterns)
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
        direction_loss_fn = nn.BCELoss()
        pattern_loss_fn   = nn.BCELoss()

        # Create dummy patterns if not provided
        if patterns is None or len(patterns) != len(images):
            patterns = np.zeros(
                (len(images), N_PATTERNS), dtype=np.float32
            )

        # Time-based split
        split = int(len(images) * (1 - self.test_size))
        val_split = int(split * 0.85)

        X_train = torch.FloatTensor(images[:val_split])
        y_train = torch.FloatTensor(labels[:val_split])
        p_train = torch.FloatTensor(patterns[:val_split])

        X_val   = torch.FloatTensor(images[val_split:split])
        y_val   = torch.FloatTensor(labels[val_split:split])
        p_val   = torch.FloatTensor(patterns[val_split:split])

        train_ds = TensorDataset(X_train, y_train, p_train)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        p_val = p_val.to(self.device)

        best_val_loss  = float("inf")
        patience_count = 0
        best_weights   = None

        log.info(
            f"[{self.ticker}] Training Pattern-Aware CNN: "
            f"{len(X_train)} train, {len(X_val)} val, "
            f"max {self.max_epochs} epochs..."
        )

        for epoch in range(1, self.max_epochs + 1):
            self.network.train()
            total_train_loss = 0.0

            for X_batch, y_batch, p_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                p_batch = p_batch.to(self.device)

                optimizer.zero_grad()
                dir_pred, pat_pred = self.network(X_batch)

                dir_loss = direction_loss_fn(dir_pred, y_batch)
                pat_loss = pattern_loss_fn(pat_pred, p_batch)
                loss = dir_loss + self.pattern_weight * pat_loss

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            self.network.eval()
            with torch.no_grad():
                val_dir, val_pat = self.network(X_val)
                val_dir_loss = direction_loss_fn(val_dir, y_val).item()
                val_pat_loss = pattern_loss_fn(val_pat, p_val).item()
                val_loss = (
                    val_dir_loss +
                    self.pattern_weight * val_pat_loss
                )

            if val_dir_loss < best_val_loss:
                best_val_loss  = val_dir_loss
                patience_count = 0
                best_weights = {
                    k: v.clone()
                    for k, v in self.network.state_dict().items()
                }
            else:
                patience_count += 1

            if epoch % 5 == 0:
                log.info(
                    f"[{self.ticker}] CNN Epoch {epoch}: "
                    f"train={total_train_loss/len(train_loader):.4f} "
                    f"val_dir={val_dir_loss:.4f} "
                    f"val_pat={val_pat_loss:.4f}"
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
        dates:  list,
    ) -> dict:
        """Evaluate CNN on held-out test images (direction only)."""
        from sklearn.metrics import accuracy_score, roc_auc_score

        split  = int(len(images) * (1 - self.test_size))
        X_test = torch.FloatTensor(images[split:]).to(self.device)
        y_test = labels[split:]

        self.network.eval()
        with torch.no_grad():
            dir_probs, _ = self.network(X_test)
            probs = dir_probs.cpu().numpy()

        preds = (probs > 0.5).astype(int)
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

    def predict_proba_image(self, image: np.ndarray) -> float:
        """Predict bullish probability from one chart image."""
        self.network.eval()
        with torch.no_grad():
            x = torch.FloatTensor(image).unsqueeze(0).to(self.device)
            dir_prob, _ = self.network(x)
            return dir_prob.item()

    def predict_with_patterns(
        self,
        image: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return both direction prob AND pattern probabilities."""
        self.network.eval()
        with torch.no_grad():
            x = torch.FloatTensor(image).unsqueeze(0).to(self.device)
            dir_prob, pat_probs = self.network(x)
            return dir_prob.item(), pat_probs.cpu().numpy()[0]

    def train(self, X_train, y_train):
        pass

    def predict_proba(self, X):
        return np.array([0.5])

    def save(self) -> Path:
        path = MODELS_DIR / f"cnn_{self.ticker}.pt"
        torch.save({
            "network_state": self.network.state_dict(),
            "ticker":        self.ticker,
            "target_days":   self.target_days,
            "dropout":       self.dropout,
            "version":       "pattern_aware_v1",
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

        # Handle loading old-architecture weights
        try:
            self.network.load_state_dict(data["network_state"])
        except RuntimeError as e:
            log.warning(
                f"[{self.ticker}] Old CNN architecture detected — "
                f"will need retraining: {e}"
            )
            raise

        self.ticker      = data["ticker"]
        self.target_days = data["target_days"]
        self.is_trained  = True
        log.info(f"[{self.ticker}] CNN loaded from {path.name}")