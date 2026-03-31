"""
enhance_fall_diversity.py
=========================
Create a cleaner fall-diversity boosted dataset from an existing train-ready folder.

Input folder must contain:
  - X_train.npy (N, 128, 69)
  - y_train.npy (N,)
  - metadata_train.csv

Output folder will contain:
  - X_train.npy
  - y_train.npy
  - metadata_train.csv
  - feat_mean.npy
  - feat_std.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COCO_FLIP_PAIRS = [
    (1, 2),   # eyes
    (3, 4),   # ears
    (5, 6),   # shoulders
    (7, 8),   # elbows
    (9, 10),  # wrists
    (11, 12), # hips
    (13, 14), # knees
    (15, 16), # ankles
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Boost fall diversity with controlled sequence augmentations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in_dir", default="data/train_ready_unified_clean_v1")
    p.add_argument("--out_dir", default="data/train_ready_unified_clean_v2_fallboost")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_fall_count", type=int, default=1300)
    p.add_argument("--max_additional", type=int, default=800)
    return p.parse_args()


def temporal_warp(seq: np.ndarray, factor: float) -> np.ndarray:
    t, d = seq.shape
    new_t = max(16, int(round(t * factor)))
    src = np.linspace(0, t - 1, new_t, dtype=np.float32)
    mid = np.empty((new_t, d), dtype=np.float32)
    base = np.arange(t, dtype=np.float32)
    for j in range(d):
        mid[:, j] = np.interp(src, base, seq[:, j])

    dst = np.linspace(0, new_t - 1, t, dtype=np.float32)
    out = np.empty((t, d), dtype=np.float32)
    base2 = np.arange(new_t, dtype=np.float32)
    for j in range(d):
        out[:, j] = np.interp(dst, base2, mid[:, j])
    return out


def temporal_shift(seq: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return seq.copy()
    out = np.empty_like(seq)
    if shift > 0:
        out[:shift] = seq[0]
        out[shift:] = seq[:-shift]
    else:
        s = -shift
        out[-s:] = seq[-1]
        out[:-s] = seq[s:]
    return out


def mirror_pose_features(seq: np.ndarray) -> np.ndarray:
    out = seq.copy()
    pose = out[:, :68].reshape(out.shape[0], 17, 4)
    pose[:, :, 0] = -pose[:, :, 0]  # mirror x
    for l, r in COCO_FLIP_PAIRS:
        tmp = pose[:, l, :].copy()
        pose[:, l, :] = pose[:, r, :]
        pose[:, r, :] = tmp
    out[:, :68] = pose.reshape(out.shape[0], 68)
    return out


def add_noise(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = seq.copy()
    non_pad = np.abs(out[:, :68]).sum(axis=1) > 1e-8
    if non_pad.any():
        sigma = rng.uniform(0.002, 0.010)
        noise = rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
        noise[:, -1] = 0.0  # keep aspect ratio stable
        out[non_pad] += noise[non_pad]
    return out


def velocity_scale(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = seq.copy()
    vel_idx = np.arange(2, 68, 4)
    acc_idx = np.arange(3, 68, 4)
    fac = rng.uniform(0.85, 1.25)
    out[:, vel_idx] *= fac
    out[:, acc_idx] *= fac
    return out


def augment_fall(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = seq.copy()
    if rng.random() < 0.85:
        out = temporal_warp(out, rng.uniform(0.80, 1.25))
    if rng.random() < 0.65:
        out = temporal_shift(out, int(rng.integers(-8, 9)))
    if rng.random() < 0.55:
        out = mirror_pose_features(out)
    out = velocity_scale(out, rng)
    if rng.random() < 0.85:
        out = add_noise(out, rng)
    out[:, -1] = np.clip(out[:, -1], 0.0, None)  # aspect ratio non-negative
    return out.astype(np.float32)


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1]).astype(np.float32)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(in_dir / "X_train.npy").astype(np.float32)
    y = np.load(in_dir / "y_train.npy").astype(np.int64)
    meta = pd.read_csv(in_dir / "metadata_train.csv")

    rng = np.random.default_rng(args.seed)
    fall_idx = np.where(y == 0)[0]
    current_fall = len(fall_idx)
    needed = max(0, min(args.max_additional, args.target_fall_count - current_fall))

    print("=" * 72)
    print("ENHANCE FALL DIVERSITY")
    print("=" * 72)
    print(f"Input samples: {len(y)} | current Fall: {current_fall} | add requested: {needed}")

    if needed > 0 and current_fall > 0:
        chosen = rng.choice(fall_idx, size=needed, replace=True)
        X_add = np.empty((needed, X.shape[1], X.shape[2]), dtype=np.float32)
        y_add = np.zeros((needed,), dtype=np.int64)
        meta_add_rows = []

        for i, idx in enumerate(chosen):
            aug = augment_fall(X[idx], rng)
            X_add[i] = aug
            row = meta.iloc[int(idx)].to_dict()
            row["source"] = "Augment_Fall"
            row["action_id"] = f"{row.get('action_id', 'fall')}_aug{i}"
            aug_type = str(row.get("aug_type", "orig"))
            row["aug_type"] = f"{aug_type}_fallboost"
            row["label_id"] = 0
            row["label_name"] = "Fall"
            meta_add_rows.append(row)

        X = np.concatenate([X, X_add], axis=0)
        y = np.concatenate([y, y_add], axis=0)
        meta_add = pd.DataFrame(meta_add_rows)
        meta = pd.concat([meta, meta_add], ignore_index=True)

    # Shuffle
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]
    meta = meta.iloc[perm].reset_index(drop=True)

    # Save
    np.save(out_dir / "X_train.npy", X)
    np.save(out_dir / "y_train.npy", y)
    meta.to_csv(out_dir / "metadata_train.csv", index=False)
    mean, std = fit_scaler(X)
    np.save(out_dir / "feat_mean.npy", mean)
    np.save(out_dir / "feat_std.npy", std)

    u, c = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(u, c)}
    src = meta["source"].value_counts().to_dict() if "source" in meta.columns else {}
    print(f"Output samples: {len(y)}")
    print(f"Label dist: {dist}")
    print(f"Source dist: {src}")
    print(f"Saved to: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
