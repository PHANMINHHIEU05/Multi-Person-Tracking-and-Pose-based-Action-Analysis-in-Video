"""
train_final.py – BƯỚC 3: LSTM-BASED ACTION RECOGNITION TRAINING
================================================================
Bi-GRU + Attention Mechanism, tối ưu cho RTX 3050 Laptop GPU.

Kiến trúc:
  - Bidirectional GRU (3 lớp, 128 hidden units)
  - Temporal Attention Mechanism (tập trung vào frame quan trọng)
  - Dropout + Layer Normalization

Chiến thuật tối ưu:
  - Mixed Precision Training (torch.cuda.amp) — tiết kiệm VRAM
  - Label Smoothing — chống overconfident
  - ReduceLROnPlateau — tự giảm LR
  - Early Stopping — chống overfitting
  - Class Weight Balancing — xử lý imbalanced data
  - Gradient Clipping — ổn định training

Output:
  - best_action_model.pth
  - Biểu đồ Loss/Accuracy

Usage:
    python train_final.py
    python train_final.py --epochs 150 --batch_size 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "./data/train_ready"
SAVE_DIR = "./runs/train"

LABEL_MAP = {
    0: "Fall",
    1: "Walking",
    2: "Bending",
    3: "Sitting_Standing",
    4: "Jogging_Jumping",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Bi-GRU + Attention for Action Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    p.add_argument("--save_dir", type=str, default=SAVE_DIR)

    # Model
    p.add_argument("--input_dim", type=int, default=68,
                   help="17 keypoints * 4 features (x, y, v, a)")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=8,
                   help="Number of attention heads (hidden_dim*2 must be divisible)")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--use_lstm", action="store_true",
                   help="Use Bi-LSTM instead of Bi-GRU")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-4,
                   help="AdamW weight decay — strong regularization")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=15,
                   help="Early stopping patience")
    p.add_argument("--lr_patience", type=int, default=8,
                   help="ReduceLROnPlateau patience")
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--cosine_lr", action="store_true", default=True,
                   help="Use CosineAnnealingWarmRestarts instead of ReduceLROnPlateau")
    p.add_argument("--cosine_t0", type=int, default=30,
                   help="T_0 for CosineAnnealingWarmRestarts")

    # GPU
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Mixed Precision Training (use --no-amp to disable)")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cuda | cpu")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.15)

    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PoseSequenceDataset(Dataset):
    """Dataset cho chuỗi pose keypoints đã xử lý."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 augment: bool = False):
        """
        Args:
            X: (N, seq_len, feature_dim) — đã flatten từ (N, 128, 34)
            y: (N,) — nhãn
            augment: Có áp dụng data augmentation không
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()  # (128, 34)
        y = self.y[idx]

        if self.augment:
            x = self._augment(x)

        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Data augmentation cho pose sequences:
          - Gaussian noise (nhỏ)
          - Temporal jitter (hoán đổi nhẹ thứ tự frame)
          - Random scaling
        """
        # Tạo mask: không augment padding frames (toàn zero)
        non_pad = x.abs().sum(dim=-1) > 0  # (128,)

        # (1) Gaussian noise — chỉ thêm noise vào frame hợp lệ
        if random.random() < 0.5:
            noise = torch.randn_like(x) * 0.005
            noise[~non_pad] = 0
            x = x + noise
            x = x.clamp(0, 1)

        # (2) Random scaling — scale toạ độ nhẹ
        if random.random() < 0.3:
            scale = 0.95 + random.random() * 0.1  # [0.95, 1.05]
            x[non_pad] = (x[non_pad] * scale).clamp(0, 1)

        # (3) Temporal shift — dịch toàn bộ chuỗi 1-3 frames
        if random.random() < 0.3:
            shift = random.randint(-3, 3)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)
                # Zero out phần bị tràn
                if shift > 0:
                    x[:shift] = 0
                else:
                    x[shift:] = 0

        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Model: Bi-GRU + Multi-Head Self-Attention
# ─────────────────────────────────────────────────────────────────────────────
class SelfAttentionPooling(nn.Module):
    """
    Transformer-style Self-Attention block + Weighted Pooling.

    Khác Temporal Attention thông thường, Self-Attention cho phép
    mỗi frame chú ý tới TẤT CẢ các frame khác — nhập bắt được
    context toàn sự kiện (v.d. so sánh tư thế trước và sau khi ngã).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Feed-Forward sublayer (giống Transformer encoder)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Learned pooling: 1 vector trọng số mỗi frame → context vector
        self.pool_fc = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x   : (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) — True = frame hợp lệ, False = padding
        Returns:
            context     : (batch, hidden_dim)
            attn_weights: (batch, seq_len)
        """
        # key_padding_mask của PyTorch: True = ignore (ngược mask của mình)
        kpm = (~mask) if mask is not None else None  # (batch, seq_len)

        # ── Self-Attention + residual + norm ─────────────────────────────
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=kpm)
        x = self.norm1(x + attn_out)

        # ── Feed-Forward + residual + norm ──────────────────────────────
        x = self.norm2(x + self.ff(x))

        # ── Weighted Pooling ────────────────────────────────────────
        scores = self.pool_fc(x).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)   # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)

        return context, weights


class ActionRecognitionModel(nn.Module):
    """
    Bi-GRU (3 lớp) + Multi-Head Self-Attention + MLP classifier.

    Kiến trúc:
        Input (batch, 128, 68)        — 17 kpts × 4 features (x,y,v,a)
        → LayerNorm
        → Input Projection: 68 → hidden_dim
        → Bi-GRU (3 lớp, 128 hidden → 256 bidirectional)
        → LayerNorm
        → Self-Attention Pooling (8 heads) → (batch, 256)
        → MLP Head: 256 → 128 → num_classes
    """

    def __init__(self, input_dim: int = 68, hidden_dim: int = 128,
                 num_layers: int = 3, num_classes: int = 2,
                 num_heads: int = 8, dropout: float = 0.4,
                 use_lstm: bool = False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_lstm = use_lstm
        bidirectional_dim = hidden_dim * 2  # 256

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # Input projection — nâng chiều feature
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Recurrent layers
        RNNClass = nn.LSTM if use_lstm else nn.GRU
        self.rnn = RNNClass(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Layer normalization after RNN
        self.rnn_norm = nn.LayerNorm(bidirectional_dim)

        # Multi-Head Self-Attention Pooling
        # bidirectional_dim=256, num_heads=8 → head_dim=32 ✓
        self.attention = SelfAttentionPooling(
            hidden_dim=bidirectional_dim,
            num_heads=num_heads,
            dropout=dropout * 0.25,
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(bidirectional_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming initialization."""
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim=68)

        Returns:
            logits      : (batch, num_classes)
            attn_weights: (batch, seq_len)
        """
        # Mask padding frames (frame toàn zero = padding)
        mask = x.abs().sum(dim=-1) > 0  # (batch, seq_len)

        # Input normalization + projection
        x = self.input_norm(x)
        x = self.input_proj(x)   # (batch, seq_len, hidden_dim)

        # RNN forward
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim*2)
        rnn_out = self.rnn_norm(rnn_out)

        # Self-Attention Pooling
        context, attn_weights = self.attention(rnn_out, mask)

        # Classification
        logits = self.classifier(context)  # (batch, num_classes)

        return logits, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
#  Training Utilities
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """Dừng training khi val metric không cải thiện."""

    def __init__(self, patience: int = 20, mode: str = "max",
                 min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Tính class weights dựa trên inverse frequency."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = total / (len(classes) * counts)
    weight_tensor = torch.zeros(max(classes) + 1)
    for c, w in zip(classes, weights):
        weight_tensor[c] = w
    return weight_tensor


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Tạo WeightedRandomSampler cho imbalanced dataset."""
    class_weights = compute_class_weights(y)
    sample_weights = class_weights[y]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: GradScaler, device: torch.device,
                    grad_clip: float, use_amp: bool) -> Tuple[float, float]:
    """Train 1 epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda"):
                logits, _ = model(X_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device,
             use_amp: bool) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate on validation set. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type="cuda"):
                logits, _ = model(X_batch)
                loss = criterion(logits, y_batch)
        else:
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: Dict[str, List[float]], save_dir: str):
    """Vẽ biểu đồ Loss/Accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Đánh dấu best epoch
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_val_loss = min(history["val_loss"])
    ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5)
    ax.annotate(f"Best: {best_val_loss:.4f}\n(epoch {best_epoch})",
                xy=(best_epoch, best_val_loss),
                xytext=(best_epoch + 5, best_val_loss + 0.05),
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=10, color="green")

    # ── Accuracy ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    ax.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    best_acc_epoch = np.argmax(history["val_acc"]) + 1
    best_val_acc = max(history["val_acc"])
    ax.axvline(x=best_acc_epoch, color="green", linestyle="--", alpha=0.5)
    ax.annotate(f"Best: {best_val_acc:.2%}\n(epoch {best_acc_epoch})",
                xy=(best_acc_epoch, best_val_acc),
                xytext=(best_acc_epoch + 5, best_val_acc - 0.1),
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=10, color="green")

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Biểu đồ đã lưu: {path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str], save_dir: str):
    """Vẽ Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    # Ghi số vào ô
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confusion matrix đã lưu: {path}")


def plot_lr_history(lr_history: List[float], save_dir: str):
    """Vẽ biểu đồ Learning Rate."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(lr_history) + 1), lr_history, "g-", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "lr_schedule.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    # ── Device setup ──────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_amp = args.amp and device.type == "cuda"

    print("=" * 70)
    print("  BƯỚC 3: TRAINING — Bi-GRU + Attention Action Recognition")
    print("=" * 70)
    print(f"\n  🖥️  Device         : {device}")
    if device.type == "cuda":
        print(f"  🎮 GPU            : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  💾 VRAM           : {vram:.1f} GB")
    print(f"  ⚡ Mixed Precision : {'ON' if use_amp else 'OFF'}")
    print(f"  🏗️  Architecture   : {'Bi-LSTM' if args.use_lstm else 'Bi-GRU'} + Multi-Head Self-Attention")
    print(f"  📦 Batch Size      : {args.batch_size}")
    print(f"  📀 Hidden Dim      : {args.hidden_dim}  |  Heads: {args.num_heads}")
    print(f"  📚 Num Layers      : {args.num_layers}")
    print(f"  🎯 Label Smoothing : {args.label_smoothing}")
    print(f"  ⚖️  Weight Decay    : {args.weight_decay}")
    print(f"  ⏳ Max Epochs      : {args.epochs}")
    print(f"  🛑 Early Stopping  : patience={args.patience}")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  LOADING DATA")
    print(f"{'─'*50}")

    X = np.load(os.path.join(args.data_dir, "X_train.npy"))  # (N, 128, 34)
    y = np.load(os.path.join(args.data_dir, "y_train.npy"))  # (N,)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {np.unique(y)}")

    # Auto-detect num_classes
    num_classes = len(np.unique(y))
    if num_classes != args.num_classes:
        print(f"  [INFO] Auto-setting num_classes = {num_classes}")
        args.num_classes = num_classes

    # ── Train/Val split (stratified) ──────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_ratio, random_state=args.seed,
        stratify=y,
    )

    print(f"\n  Train: {len(X_train)} samples")
    print(f"  Val  : {len(X_val)} samples")

    # Phân bố nhãn
    for split_name, split_y in [("Train", y_train), ("Val", y_val)]:
        classes, counts = np.unique(split_y, return_counts=True)
        dist = ", ".join(f"{LABEL_MAP.get(c, c)}: {n}" for c, n in zip(classes, counts))
        print(f"  {split_name} distribution: {dist}")

    # ── Datasets & Loaders ────────────────────────────────────────────────
    train_dataset = PoseSequenceDataset(X_train, y_train, augment=True)
    val_dataset = PoseSequenceDataset(X_val, y_val, augment=False)

    # Weighted sampler cho imbalanced data
    sampler = make_weighted_sampler(y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = ActionRecognitionModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_lstm=args.use_lstm,
    ).to(device)

    # Đếm parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  🧠 Model Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────
    class_weights = compute_class_weights(y_train).to(device)
    print(f"  ⚖️  Class Weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.cosine_lr:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.cosine_t0,
            T_mult=2,
            eta_min=args.min_lr,
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

    scaler = GradScaler("cuda", enabled=use_amp)
    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    # ── Training ──────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_action_model.pth")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_val_f1 = 0.0

    print(f"\n{'─'*50}")
    print("  TRAINING STARTS")
    print(f"{'─'*50}")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, args.grad_clip, use_amp,
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device, use_amp,
        )

        # Metrics
        val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - t_epoch

        # Print progress
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            marker = " ★ BEST"

            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_loss": val_loss,
                "args": vars(args),
            }, best_model_path)

        if epoch % 5 == 0 or epoch <= 3 or marker:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.2e} | {epoch_time:.1f}s{marker}")

        # Scheduler step (monitoring val_acc)
        if args.cosine_lr:
            scheduler.step(epoch)  # CosineAnnealingWarmRestarts dùng epoch index
        else:
            scheduler.step(val_acc)  # ReduceLROnPlateau dùng metric

        # Early stopping
        if early_stopping(val_acc):
            print(f"\n  🛑 Early Stopping tại epoch {epoch}! "
                  f"Val Acc không cải thiện sau {args.patience} epochs.")
            break

    total_time = time.time() - t_start

    # ── Final Evaluation ──────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  ĐÁNH GIÁ CUỐI CÙNG")
    print(f"{'─'*50}")

    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, final_acc, final_preds, final_labels = evaluate(
        model, val_loader, criterion, device, use_amp,
    )

    class_names = [LABEL_MAP.get(i, f"Class_{i}") for i in range(args.num_classes)]

    print(f"\n  Best Epoch    : {best_epoch}")
    print(f"  Best Val Acc  : {best_val_acc:.4f} ({best_val_acc:.2%})")
    print(f"  Best Val F1   : {best_val_f1:.4f}")
    print(f"  Training Time : {total_time:.1f}s ({total_time/60:.1f} min)")

    print(f"\n  Classification Report:")
    report = classification_report(
        final_labels, final_preds,
        target_names=class_names,
        digits=4,
    )
    print(report)

    # ── Visualization ─────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  VISUALIZATION")
    print(f"{'─'*50}")

    plot_training_curves(history, args.save_dir)
    plot_confusion_matrix(final_labels, final_preds, class_names, args.save_dir)
    plot_lr_history(history["lr"], args.save_dir)

    # ── Save training history ─────────────────────────────────────────────
    history_path = os.path.join(args.save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ✅ TRAINING HOÀN TẤT!")
    print(f"{'='*70}")
    print(f"  📁 Model saved    : {best_model_path}")
    print(f"  📊 Plots saved    : {args.save_dir}/")
    print(f"  📈 Best Val Acc   : {best_val_acc:.2%}")
    print(f"  📈 Best Val F1    : {best_val_f1:.4f}")
    print(f"  ⏰ Total Time     : {total_time/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
