"""
train_professional_v3.py – Production Training for Real-World Fall Detection
=============================================================================
Architecture : Bi-GRU (3 layers, 128 hidden) + Multi-Head Self-Attention
Loss         : Focal Loss (γ=2.0) with class weights
Optimiser    : AdamW + CosineAnnealingWarmRestarts
Input        : (N, 128, 69) — 17 keypoints × 4 features + 1 aspect ratio

Default classes (overridden by data_dir/label_map.json when available):
    0: Fall   1: Walking   2: Sitting_Quickly   3: Bending   4: Lying_Down

Output: ./runs/train_v3/final_safe_system.pth

Usage:
    python train_professional_v3.py
    python train_professional_v3.py --no-amp --epochs 500 --patience 80
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "./data/train_ready_horizontal"
SAVE_DIR = "./runs/train_horizontal"

DEFAULT_LABEL_MAP = {
    0: "Fall",
    1: "Walking",
    2: "Sitting_Quickly",
    3: "Bending",
    4: "Lying_Down",
}


def load_label_map(data_dir: str, classes: np.ndarray) -> Dict[int, str]:
    """Load class names from dataset if present, else fallback to defaults."""
    lm_path = os.path.join(data_dir, "label_map.json")
    label_map: Dict[int, str] = dict(DEFAULT_LABEL_MAP)

    if os.path.exists(lm_path):
        try:
            with open(lm_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            parsed: Dict[int, str] = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        parsed[int(k)] = str(v)
                    except (TypeError, ValueError):
                        continue
            if parsed:
                label_map = parsed
                print(f"  [LABEL_MAP] Loaded from {lm_path}")
            else:
                print(f"  [LABEL_MAP] Invalid {lm_path}, fallback to default")
        except Exception as e:
            print(f"  [LABEL_MAP] Failed to load ({e}), fallback to default")
    else:
        print("  [LABEL_MAP] label_map.json not found, using default")

    resolved = {int(c): label_map.get(int(c), f"C{int(c)}") for c in classes}
    print(f"  [LABEL_MAP] Active: {resolved}")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Bi-GRU + Self-Attention (v3 — Focal Loss)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    p.add_argument("--save_dir", type=str, default=SAVE_DIR)

    # Model config:
    # Moi mau dau vao la 128 frame, moi frame co 69 feature tu khung xuong.
    # 69 = 17 keypoints * (x, y, velocity, acceleration) + 1 bbox aspect ratio.
    p.add_argument("--input_dim", type=int, default=69,
                   help="17 kpts × 4 + 1 aspect ratio")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.4)

    # Focal Loss giup model tap trung hon vao mau kho / lop it du lieu.
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="Focusing parameter for Focal Loss")

    # Training config chinh cua ban train cuoi:
    # AdamW + CosineAnnealingWarmRestarts + EarlyStopping.
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=2e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=15,
                   help="Early stopping patience")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--cosine_t0", type=int, default=50,
                   help="T_0 for CosineAnnealingWarmRestarts")
    p.add_argument("--min_lr", type=float, default=1e-6)

    # GPU / AMP
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Mixed Precision Training (--no-amp to disable)")
    p.add_argument("--device", type=str, default="auto")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.15)

    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss:  FL(p_t) = -α_t (1 - p_t)^γ  log(p_t)

    Handles class imbalance better than CE because it down-weights
    well-classified (easy) examples and focuses on hard ones.
    """

    def __init__(self, weight: torch.Tensor | None = None,
                 gamma: float = 2.0,
                 label_smoothing: float = 0.0,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight                      # class weights (α)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """inputs: (B, C) logits,  targets: (B,) class indices."""
        # Cross Entropy tinh loi phan loai co ban.
        # self.weight la class weight, dung de bu cho lop it mau nhu Standing.
        ce = F.cross_entropy(inputs, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing,
                             reduction="none")
        # pt cang cao nghia la model cang doan dung va chac chan.
        pt = torch.exp(-ce)                        # p_t for correct class
        # Mau de se bi giam trong so; mau kho giu loss cao hon de model hoc ky.
        focal = ((1.0 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


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
class PoseDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        if self.augment:
            x = self._augment(x)
        return x, self.y[idx]

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        # non_pad danh dau frame co du lieu that, tranh them nhieu vao frame padding.
        non_pad = x.abs().sum(dim=-1) > 0
        # Them nhieu nhe vao feature khung xuong de model chiu duoc pose rung/sai so.
        if random.random() < 0.5:
            noise = torch.randn_like(x) * 0.005
            noise[~non_pad] = 0
            x = x + noise
        # Scale nhe de gia lap nguoi dung gan/xa camera khac nhau.
        if random.random() < 0.3:
            scale = 0.95 + random.random() * 0.1
            x[non_pad] *= scale
        # Dich chuoi theo thoi gian de model bot phu thuoc vao dung thoi diem bat dau hanh dong.
        if random.random() < 0.3:
            shift = random.randint(-3, 3)
            if shift:
                x = torch.roll(x, shifts=shift, dims=0)
                if shift > 0:
                    x[:shift] = 0
                else:
                    x[shift:] = 0
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────
class SelfAttentionPooling(nn.Module):
    """Self-Attention + Weighted Pooling over temporal sequence."""

    def __init__(self, hidden_dim: int, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        # Residual + LayerNorm giup attention hoc on dinh hon khi chuoi dai.
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        # pool_fc tao diem quan trong cho tung frame trong chuoi 128 frame.
        self.pool_fc = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        kpm = (~mask) if mask is not None else None
        # Self-attention cho moi frame "nhin" cac frame khac trong cung hanh dong.
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=kpm)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        # scores -> weights: frame nao quan trong hon se duoc gan trong so cao hon.
        scores = self.pool_fc(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        # context la vector tong hop ca chuoi sau khi nhan trong so attention.
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context, weights


class ActionRecognitionModel(nn.Module):
    """
    Bi-GRU (3 layers, 128 hidden) + Multi-Head Self-Attention + MLP.

    Input:  (batch, 128, 69)
    Output: (batch, num_classes), (batch, 128) attention weights
    """

    def __init__(self, input_dim: int = 69, hidden_dim: int = 128,
                 num_layers: int = 3, num_classes: int = 5,
                 num_heads: int = 8, dropout: float = 0.4):
        super().__init__()
        self.hidden_dim = hidden_dim
        bi_dim = hidden_dim * 2

        # Chuan hoa 69 feature dau vao truoc khi dua vao model.
        self.input_norm = nn.LayerNorm(input_dim)
        # Chieu 69 feature len hidden_dim=128 de GRU hoc bieu dien giau hon.
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Bi-GRU doc chuoi theo 2 chieu, hoc bien doi tu the theo thoi gian.
        self.rnn = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.rnn_norm = nn.LayerNorm(bi_dim)

        # Attention chon cac frame quan trong, vi du khoanh khac bat dau nga.
        self.attention = SelfAttentionPooling(
            hidden_dim=bi_dim, num_heads=num_heads,
            dropout=dropout * 0.25,
        )

        # MLP classifier bien vector context thanh 5 logits hanh dong.
        self.classifier = nn.Sequential(
            nn.Linear(bi_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
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
        # mask bo qua frame padding/toan 0 khi tinh attention.
        mask = x.abs().sum(dim=-1) > 0
        x = self.input_norm(x)
        x = self.input_proj(x)
        # rnn_out giu thong tin dong hoc cua ca chuoi keypoints.
        rnn_out, _ = self.rnn(x)
        rnn_out = self.rnn_norm(rnn_out)
        # attn_w co the dung de xem model dang tap trung vao frame nao.
        context, attn_w = self.attention(rnn_out, mask)
        logits = self.classifier(context)
        return logits, attn_w


# ─────────────────────────────────────────────────────────────────────────────
#  Training utilities
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 15, mode: str = "max",
                 min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (score > self.best_score + self.min_delta
                    if self.mode == "max"
                    else score < self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    w = total / (len(classes) * counts)
    wt = torch.zeros(int(classes.max()) + 1, dtype=torch.float32)
    for c, wv in zip(classes, w):
        wt[int(c)] = wv
    return wt


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    cw = compute_class_weights(y)
    sw = cw[y]
    return WeightedRandomSampler(sw, len(y), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Train / Eval loops
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, grad_clip, use_amp):
    model.train()
    total_loss = correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast(device_type="cuda"):
                logits, _ = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(-1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = correct = total = 0
    all_p, all_l = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        if use_amp:
            with autocast(device_type="cuda"):
                logits, _ = model(xb)
                loss = criterion(logits, yb)
        else:
            logits, _ = model(xb)
            loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(-1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        all_p.append(preds.cpu().numpy())
        all_l.append(yb.cpu().numpy())
    return (total_loss / total, correct / total,
            np.concatenate(all_p), np.concatenate(all_l))


# ─────────────────────────────────────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────────────────────────────────────
def plot_curves(history: dict, save_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history["train_loss"]) + 1)

    ax1.plot(ep, history["train_loss"], "b-", label="Train", lw=2)
    ax1.plot(ep, history["val_loss"], "r-", label="Val", lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, history["train_acc"], "b-", label="Train", lw=2)
    ax2.plot(ep, history["val_acc"], "r-", label="Val", lw=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1.05])

    best_e = int(np.argmax(history["val_acc"])) + 1
    best_v = max(history["val_acc"])
    ax2.axvline(best_e, color="green", ls="--", alpha=.5)
    ax2.annotate(f"Best: {best_v:.2%}\n(ep {best_e})",
                 xy=(best_e, best_v),
                 xytext=(best_e + 3, best_v - 0.08),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 fontsize=10, color="green")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true, y_pred, names, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im)
    ax.set(xticks=range(len(names)), yticks=range(len(names)),
           xticklabels=names, yticklabels=names,
           ylabel="True", xlabel="Predicted", title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = (torch.device("cuda") if args.device == "auto"
              and torch.cuda.is_available()
              else torch.device(args.device if args.device != "auto" else "cpu"))
    use_amp = args.amp and device.type == "cuda"

    print("=" * 70)
    print("  TRAIN v3 — Bi-GRU + Self-Attention + Focal Loss")
    print("=" * 70)
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM         : {vram:.1f} GB")
    print(f"  AMP          : {'ON' if use_amp else 'OFF'}")
    print(f"  Focal γ      : {args.focal_gamma}")
    print(f"  Batch        : {args.batch_size}")
    print(f"  LR           : {args.lr}")
    print(f"  Hidden       : {args.hidden_dim}  Heads: {args.num_heads}")
    print(f"  Patience     : {args.patience}")

    # ── Data ──────────────────────────────────────────────────────────────
    # Prefer merged dataset (UR Fall + sideways walking) if available
    merged_X = os.path.join(args.data_dir, "X_train_merged.npy")
    merged_y = os.path.join(args.data_dir, "y_train_merged.npy")
    if os.path.exists(merged_X):
        X = np.load(merged_X)
        y = np.load(merged_y)
        print(f"\n  [DATA] Using merged dataset")
    else:
        X = np.load(os.path.join(args.data_dir, "X_train.npy"))
        y = np.load(os.path.join(args.data_dir, "y_train.npy"))
    print(f"  X: {X.shape}   y: {y.shape}")

    # ── StandardScaler normalization ──────────────────────────────────────
    feat_mean_path = os.path.join(args.data_dir, "feat_mean.npy")
    feat_std_path  = os.path.join(args.data_dir, "feat_std.npy")
    if os.path.exists(feat_mean_path):
        feat_mean = np.load(feat_mean_path)   # (69,)
        feat_std  = np.load(feat_std_path)    # (69,)
        X = (X - feat_mean) / feat_std
        X = X.astype(np.float32)
        print(f"  [SCALER] Applied StandardScaler  mean={feat_mean.mean():.4f}  std={feat_std.mean():.4f}")
        use_scaler = True
    else:
        feat_mean = feat_std = None
        use_scaler = False
        print("  [SCALER] Not found — skipping normalization")

    classes = np.unique(y)
    expected = np.arange(len(classes))
    if not np.array_equal(classes, expected):
        raise ValueError(
            f"Class ids must be contiguous from 0..C-1, got {classes.tolist()}"
        )

    num_classes = len(classes)
    if num_classes != args.num_classes:
        print(f"  [AUTO] num_classes → {num_classes}")
        args.num_classes = num_classes

    label_map = load_label_map(args.data_dir, classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_ratio, random_state=args.seed, stratify=y)
    print(f"  Train: {len(X_train)}   Val: {len(X_val)}")
    for name, sy in [("Train", y_train), ("Val", y_val)]:
        cls, cnt = np.unique(sy, return_counts=True)
        d = ", ".join(f"{label_map.get(int(c), c)}:{n}" for c, n in zip(cls, cnt))
        print(f"  {name}: {d}")

    # ── Loaders ───────────────────────────────────────────────────────────
    train_ds = PoseDataset(X_train, y_train, augment=True)
    val_ds   = PoseDataset(X_val, y_val, augment=False)
    sampler  = make_weighted_sampler(y_train)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=args.pin_memory)

    # ── Model ─────────────────────────────────────────────────────────────
    model = ActionRecognitionModel(
        input_dim=args.input_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, num_classes=args.num_classes,
        num_heads=args.num_heads, dropout=args.dropout,
    ).to(device)

    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable params: {tp:,}")

    # ── Loss / Optim / Scheduler ──────────────────────────────────────────
    class_w = compute_class_weights(y_train).to(device)
    print(f"  Class weights: {class_w.cpu().numpy().round(3)}")

    criterion = FocalLoss(weight=class_w, gamma=args.focal_gamma,
                          label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_t0,
                                           T_mult=2, eta_min=args.min_lr)
    scaler = GradScaler("cuda", enabled=use_amp)
    early  = EarlyStopping(patience=args.patience, mode="max")

    # ── Training ──────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "final_safe_system.pth")

    history: Dict[str, List[float]] = dict(
        train_loss=[], train_acc=[], val_loss=[], val_acc=[], lr=[])

    best_acc = best_f1 = 0.0
    best_epoch = 0
    t0 = time.time()

    print(f"\n{'─'*50}\n  TRAINING\n{'─'*50}")
    for epoch in range(1, args.epochs + 1):
        te = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_dl, criterion, optimizer, scaler,
            device, args.grad_clip, use_amp)

        vl_loss, vl_acc, vl_pred, vl_true = evaluate(
            model, val_dl, criterion, device, use_amp)

        vl_f1 = f1_score(vl_true, vl_pred, average="weighted", zero_division=0)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["lr"].append(lr_now)

        mark = ""
        if vl_acc > best_acc:
            best_acc, best_f1, best_epoch = vl_acc, vl_f1, epoch
            mark = " ★"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": vl_acc, "val_f1": vl_f1, "val_loss": vl_loss,
                "args": vars(args),
                "label_map": label_map,
                "feat_mean": feat_mean,
                "feat_std":  feat_std,
            }, ckpt_path)

        if epoch % 5 == 0 or epoch <= 3 or mark:
            dt = time.time() - te
            print(f"  Ep {epoch:3d}/{args.epochs} | "
                  f"TrL {tr_loss:.4f} TrA {tr_acc:.4f} | "
                  f"VL {vl_loss:.4f} VA {vl_acc:.4f} F1 {vl_f1:.4f} | "
                  f"LR {lr_now:.2e} | {dt:.1f}s{mark}")

        scheduler.step(epoch)
        if early(vl_acc):
            print(f"\n  Early stop at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    total_t = time.time() - t0

    # ── Final evaluation ──────────────────────────────────────────────────
    print(f"\n{'─'*50}\n  FINAL EVALUATION\n{'─'*50}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_acc, final_pred, final_true = evaluate(
        model, val_dl, criterion, device, use_amp)

    names = [label_map.get(i, f"C{i}") for i in range(args.num_classes)]
    print(f"\n  Best epoch  : {best_epoch}")
    print(f"  Best val acc: {best_acc:.4f} ({best_acc:.2%})")
    print(f"  Best val F1 : {best_f1:.4f}")
    print(f"  Time        : {total_t:.0f}s ({total_t/60:.1f} min)\n")
    print(classification_report(final_true, final_pred,
                                target_names=names, digits=4))

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_curves(history, args.save_dir)
    plot_confusion(final_true, final_pred, names, args.save_dir)

    with open(os.path.join(args.save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  DONE — saved: {ckpt_path}")
    print(f"  Val Acc: {best_acc:.2%}   F1: {best_f1:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
