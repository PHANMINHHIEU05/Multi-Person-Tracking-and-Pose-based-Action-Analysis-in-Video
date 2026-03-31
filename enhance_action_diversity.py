"""
enhance_action_diversity.py
===========================
Boost diversity for confusion-prone classes in an existing train-ready dataset.

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


LABEL_NAMES = {
    0: "Fall",
    1: "Walking",
    2: "Sitting_Quickly",
    3: "Bending",
    4: "Lying_Down",
}

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
        description="Boost diversity for selected labels with controlled sequence augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in_dir", default="data/train_ready_unified_clean_v2_fallboost")
    p.add_argument("--out_dir", default="data/train_ready_unified_clean_v3_actionboost")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--labels_to_boost",
        default="2,3,4",
        help="Comma-separated label ids to augment",
    )
    p.add_argument(
        "--target_count",
        type=int,
        default=1100,
        help="Target samples for each selected label",
    )
    p.add_argument(
        "--max_additional_per_label",
        type=int,
        default=500,
        help="Safety cap for synthetic additions per selected label",
    )
    return p.parse_args()


def parse_labels(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val not in LABEL_NAMES:
            raise ValueError(f"Unknown label id: {val}")
        out.append(val)
    return sorted(set(out))


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
    pose[:, :, 0] = -pose[:, :, 0]
    for left, right in COCO_FLIP_PAIRS:
        tmp = pose[:, left, :].copy()
        pose[:, left, :] = pose[:, right, :]
        pose[:, right, :] = tmp
    out[:, :68] = pose.reshape(out.shape[0], 68)
    return out


def add_noise(seq: np.ndarray, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    out = seq.copy()
    non_pad = np.abs(out[:, :68]).sum(axis=1) > 1e-8
    if non_pad.any():
        sigma = rng.uniform(lo, hi)
        noise = rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
        noise[:, -1] = 0.0
        out[non_pad] += noise[non_pad]
    return out


def velocity_scale(
    seq: np.ndarray,
    rng: np.random.Generator,
    lo: float,
    hi: float,
) -> np.ndarray:
    out = seq.copy()
    vel_idx = np.arange(2, 68, 4)
    acc_idx = np.arange(3, 68, 4)
    fac = rng.uniform(lo, hi)
    out[:, vel_idx] *= fac
    out[:, acc_idx] *= fac
    return out


def augment_by_label(seq: np.ndarray, label: int, rng: np.random.Generator) -> np.ndarray:
    out = seq.copy()

    if label == 1:  # Walking
        if rng.random() < 0.9:
            out = temporal_warp(out, rng.uniform(0.70, 1.35))
        if rng.random() < 0.7:
            out = temporal_shift(out, int(rng.integers(-10, 11)))
        if rng.random() < 0.5:
            out = mirror_pose_features(out)
        out = velocity_scale(out, rng, 0.70, 1.35)
        if rng.random() < 0.8:
            out = add_noise(out, rng, 0.002, 0.008)
    elif label in (2, 3):  # Sitting_Quickly, Bending
        if rng.random() < 0.9:
            out = temporal_warp(out, rng.uniform(0.80, 1.20))
        if rng.random() < 0.65:
            out = temporal_shift(out, int(rng.integers(-8, 9)))
        if rng.random() < 0.6:
            out = mirror_pose_features(out)
        out = velocity_scale(out, rng, 0.80, 1.20)
        if rng.random() < 0.85:
            out = add_noise(out, rng, 0.002, 0.010)
    elif label == 4:  # Lying_Down
        if rng.random() < 0.75:
            out = temporal_warp(out, rng.uniform(0.90, 1.15))
        if rng.random() < 0.55:
            out = temporal_shift(out, int(rng.integers(-6, 7)))
        if rng.random() < 0.55:
            out = mirror_pose_features(out)
        out = velocity_scale(out, rng, 0.85, 1.15)
        if rng.random() < 0.75:
            out = add_noise(out, rng, 0.001, 0.007)
    else:  # fallback (e.g. Fall)
        if rng.random() < 0.85:
            out = temporal_warp(out, rng.uniform(0.80, 1.25))
        if rng.random() < 0.65:
            out = temporal_shift(out, int(rng.integers(-8, 9)))
        if rng.random() < 0.55:
            out = mirror_pose_features(out)
        out = velocity_scale(out, rng, 0.85, 1.25)
        if rng.random() < 0.85:
            out = add_noise(out, rng, 0.002, 0.010)

    out[:, -1] = np.clip(out[:, -1], 0.0, None)
    return out.astype(np.float32)


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1]).astype(np.float32)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    args = parse_args()
    labels = parse_labels(args.labels_to_boost)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(in_dir / "X_train.npy").astype(np.float32)
    y = np.load(in_dir / "y_train.npy").astype(np.int64)
    meta = pd.read_csv(in_dir / "metadata_train.csv")

    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("ENHANCE ACTION DIVERSITY")
    print("=" * 72)
    print(f"Input samples: {len(y)}")
    print(f"Boost labels: {labels} | target_count: {args.target_count}")

    added_total = 0
    add_X_all: list[np.ndarray] = []
    add_y_all: list[np.ndarray] = []
    add_rows_all: list[dict] = []

    for label in labels:
        idx = np.where(y == label)[0]
        current = len(idx)
        needed = max(
            0,
            min(args.max_additional_per_label, args.target_count - current),
        )
        print(
            f"Label {label} ({LABEL_NAMES[label]}): "
            f"current={current} add={needed}"
        )
        if needed <= 0 or current <= 0:
            continue

        chosen = rng.choice(idx, size=needed, replace=True)
        X_add = np.empty((needed, X.shape[1], X.shape[2]), dtype=np.float32)
        y_add = np.full((needed,), label, dtype=np.int64)

        for i, old_idx in enumerate(chosen):
            X_add[i] = augment_by_label(X[old_idx], label, rng)
            row = meta.iloc[int(old_idx)].to_dict()
            row["source"] = f"Augment_{LABEL_NAMES[label]}"
            row["action_id"] = f"{row.get('action_id', f'label{label}')}_aug{i}"
            aug_type = str(row.get("aug_type", "orig"))
            row["aug_type"] = f"{aug_type}_actionboost"
            row["label_id"] = int(label)
            row["label_name"] = LABEL_NAMES[label]
            add_rows_all.append(row)

        add_X_all.append(X_add)
        add_y_all.append(y_add)
        added_total += needed

    if added_total > 0:
        X_add_full = np.concatenate(add_X_all, axis=0)
        y_add_full = np.concatenate(add_y_all, axis=0)
        meta_add = pd.DataFrame(add_rows_all)
        X = np.concatenate([X, X_add_full], axis=0)
        y = np.concatenate([y, y_add_full], axis=0)
        meta = pd.concat([meta, meta_add], ignore_index=True)

    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]
    meta = meta.iloc[perm].reset_index(drop=True)

    np.save(out_dir / "X_train.npy", X)
    np.save(out_dir / "y_train.npy", y)
    meta.to_csv(out_dir / "metadata_train.csv", index=False)
    mean, std = fit_scaler(X)
    np.save(out_dir / "feat_mean.npy", mean)
    np.save(out_dir / "feat_std.npy", std)

    unique, counts = np.unique(y, return_counts=True)
    label_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    src_dist = meta["source"].value_counts().to_dict() if "source" in meta.columns else {}
    print(f"Added total: {added_total}")
    print(f"Output samples: {len(y)}")
    print(f"Label dist: {label_dist}")
    print(f"Source dist: {src_dist}")
    print(f"Saved to: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
