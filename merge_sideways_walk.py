"""
merge_sideways_walk.py
────────────────────────────────────────────────────────────────────────────
Gộp sideways_walk.npy (người đi ngang camera) vào tập huấn luyện hiện tại.
Lưu bộ dữ liệu mới vào data/train_ready_horizontal/ cùng với StandardScaler.

Usage:
    python merge_sideways_walk.py
"""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, '.')

DATA_DIR  = Path("data/train_ready_horizontal")
SW_PATH   = DATA_DIR / "sideways_walk.npy"

# ── Load existing data ─────────────────────────────────────────────────────
X_orig = np.load(DATA_DIR / "X_train.npy")   # (2823, 128, 69)
y_orig = np.load(DATA_DIR / "y_train.npy")   # (2823,)
print(f"Original: X={X_orig.shape}  y dist={dict(zip(*np.unique(y_orig,return_counts=True)))}")

# ── Load sideways walk ────────────────────────────────────────────────────
X_sw   = np.load(SW_PATH)                     # (320, 128, 69)
y_sw   = np.ones(len(X_sw), dtype=np.int64)   # label 1 = Walking
print(f"Sideways walk: X={X_sw.shape}  all label=1 (Walking)")

# ── Merge ─────────────────────────────────────────────────────────────────
X_all  = np.concatenate([X_orig, X_sw], axis=0)
y_all  = np.concatenate([y_orig, y_sw], axis=0)

# Shuffle
rng    = np.random.default_rng(42)
idx    = rng.permutation(len(X_all))
X_all  = X_all[idx]
y_all  = y_all[idx]

print(f"Merged: X={X_all.shape}  y dist={dict(zip(*np.unique(y_all,return_counts=True)))}")

# ── Save merged ───────────────────────────────────────────────────────────
np.save(DATA_DIR / "X_train_merged.npy", X_all)
np.save(DATA_DIR / "y_train_merged.npy", y_all)
print(f"Saved → {DATA_DIR}/X_train_merged.npy, y_train_merged.npy")

# ── Fit StandardScaler on train features ─────────────────────────────────
# Fit on per-feature mean/std across all samples × all time steps
print("\nFitting StandardScaler...")
N, T, F = X_all.shape
X_flat   = X_all.reshape(-1, F)          # (N*T, 69)
feat_mean = X_flat.mean(axis=0)          # (69,)
feat_std  = X_flat.std(axis=0)           # (69,)
feat_std  = np.where(feat_std < 1e-8, 1.0, feat_std)  # avoid division by zero

print(f"  feat_mean range: [{feat_mean.min():.4f}, {feat_mean.max():.4f}]")
print(f"  feat_std  range: [{feat_std.min():.4f}, {feat_std.max():.4f}]")
print(f"  ar mean={feat_mean[-1]:.3f} std={feat_std[-1]:.3f}")
print(f"  vel[0] mean={feat_mean[34]:.4f} std={feat_std[34]:.4f}")

np.save(DATA_DIR / "feat_mean.npy", feat_mean.astype(np.float32))
np.save(DATA_DIR / "feat_std.npy",  feat_std.astype(np.float32))
print(f"Saved → {DATA_DIR}/feat_mean.npy, feat_std.npy")

# Verify: normalize and check
X_norm = (X_flat - feat_mean) / feat_std
print(f"\nAfter normalization: mean={X_norm.mean():.4f} std={X_norm.std():.4f}")
print(f"  ar normalized range: [{X_norm[:,-1].min():.2f}, {X_norm[:,-1].max():.2f}]")
print(f"  vel[0] normalized range: [{X_norm[:,34].min():.2f}, {X_norm[:,34].max():.2f}]")
