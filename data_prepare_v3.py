"""
data_prepare_v3.py – Preprocessing tối ưu cho camera NGANG (hành lang, gia đình)
==================================================================================
Chỉ dùng UR Fall dataset (cam0 – camera ngang): loại bỏ Multicam (góc cao) vì gây
nhiễu khi train cho camera ngang.

5 classes:
  0: Fall            – Ngã ngã người
  1: Walking         – Đi bộ / chạy
  2: Sitting_Quickly – Ngồi xuống nhanh
  3: Bending         – Cúi người
  4: Lying_Down      – Nằm nghỉ (hậu kỳ ngã)

Augmentation cho camera ngang:
  ● Camera tilt nhỏ (±5–10°) – camera gắn hơi cao trên tường
  ● Scale variation (0.75–1.25) – simulate người đứng gần/xa camera
  ● Horizontal flip, Gaussian noise, time-warp
  ● Hip-centred normalisation + Moving Average(5) + vel/acc/aspect_ratio

Output: ./data/train_ready_horizontal/
  X_train.npy   (N, 128, 69)
  y_train.npy   (N,)
  metadata_train.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
POSE_DIR        = "./data/processed_pose"
OUT_DIR         = "./data/train_ready_horizontal"
SEQ_LEN         = 128
STRIDE          = 16
FEATURE_DIM     = 69          # 17×4 + 1
MA_WINDOW       = 5           # Moving-average window

LABEL_MAP = {
    0: "Fall",
    1: "Walking",
    2: "Sitting_Quickly",
    3: "Bending",
    4: "Lying_Down",
}

# COCO 17-keypoint indices
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

COCO_FLIP_PAIRS = [
    (L_EYE, R_EYE), (L_EAR, R_EAR),
    (L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST),
    (L_HIP, R_HIP), (L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE),
]

# Lying-phase detection thresholds
LYING_SKEL_H_THRESH  = 0.25   # skeleton height below this → person is flat
LYING_MIN_FRAMES     = 10     # minimum contiguous lying frames to count

# ─────────────────────────────────────────────────────────────────────────────
#  UR Fall label mapping
# ─────────────────────────────────────────────────────────────────────────────
def ur_fall_label(seq_name: str) -> tuple[int | None, bool]:
    """Return (label_id, is_fall_sequence).  None → skip."""
    name = seq_name.lower()
    if name.startswith("fall"):
        return 0, True                         # will be split later
    if name.startswith("adl"):
        try:
            num = int(name.split("-")[1])
        except (IndexError, ValueError):
            return None, False
        if 1 <= num <= 8:
            return 1, False                    # Walking
        if 9 <= num <= 16:
            return 3, False                    # Bending
        if 17 <= num <= 30:
            return 2, False                    # Sitting_Quickly
        if 31 <= num <= 40:
            return 1, False                    # Walking (jogging → similar)
    return None, False


# ─────────────────────────────────────────────────────────────────────────────
#  Core preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────
def interpolate_zeros(data: np.ndarray) -> np.ndarray:
    """Linear-interpolate frames where ALL keypoints are (0,0)."""
    T = data.shape[0]
    zero_mask = np.all(data == 0, axis=(1, 2))
    if not zero_mask.any():
        return data
    result = data.copy()
    idx    = np.arange(T)
    valid  = ~zero_mask
    if valid.sum() < 2:
        return result
    for k in range(data.shape[1]):
        for c in range(data.shape[2]):
            result[:, k, c] = np.interp(idx, idx[valid], data[valid, k, c])
    return result


def temporal_ma(data: np.ndarray, window: int = MA_WINDOW) -> np.ndarray:
    """Apply moving-average filter along the temporal axis."""
    out = data.copy()
    for k in range(data.shape[1]):
        for c in range(data.shape[2]):
            out[:, k, c] = uniform_filter1d(data[:, k, c], size=window,
                                            mode="nearest")
    return out


def hip_centered_normalize(data: np.ndarray) -> np.ndarray:
    """
    Centre every frame on mid-hip and scale by skeleton height
    (nose → mid-ankle).  Returns (T, 17, 2).
    """
    result = np.zeros_like(data)
    for t in range(data.shape[0]):
        kp = data[t]
        mid_hip   = (kp[L_HIP] + kp[R_HIP]) / 2.0
        mid_ankle = (kp[L_ANKLE] + kp[R_ANKLE]) / 2.0
        skel_h    = np.linalg.norm(kp[NOSE] - mid_ankle)
        # Fallback for very flat / missing poses
        if skel_h < 0.05:
            mid_sh = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2.0
            skel_h = np.linalg.norm(mid_sh - mid_hip) * 3.0
        skel_h = max(skel_h, 0.01)
        result[t] = (kp - mid_hip) / skel_h
    return result


def bbox_aspect_ratio(data: np.ndarray) -> np.ndarray:
    """Compute per-frame bbox W/H from keypoints.  data: (T, 17, 2) raw.
    Returns (T, 1)."""
    T = data.shape[0]
    ratio = np.ones((T, 1), dtype=np.float32)
    for t in range(T):
        kp = data[t]
        valid = np.any(kp != 0, axis=1)
        if valid.sum() < 2:
            continue
        vk = kp[valid]
        w = vk[:, 0].max() - vk[:, 0].min()
        h = vk[:, 1].max() - vk[:, 1].min()
        ratio[t, 0] = w / max(h, 1e-4)
    return ratio


# ─────────────────────────────────────────────────────────────────────────────
#  Feature engineering (full pipeline for one sequence)
# ─────────────────────────────────────────────────────────────────────────────
def compute_features(data_raw: np.ndarray,
                     tilt_angle: float = 0.0,
                     scale_factor: float = 1.0,
                     do_flip: bool = False,
                     noise_sigma: float = 0.0) -> np.ndarray:
    """
    Full pipeline:  raw (T,17,2) → (T, 69).

    1. Interpolate zeros
    2. (optional) small camera tilt on raw coords
    3. (optional) scale variation (simulate distance)
    4. Bbox aspect ratio
    5. Moving-Average temporal filter
    6. Hip-centred normalisation
    7. (optional) horizontal flip
    8. Velocity + Acceleration
    9. Flatten 17×4 + append aspect ratio → (T, 69)
    10. (optional) additive Gaussian noise
    """
    # 1 — interpolate
    data = interpolate_zeros(data_raw)

    # 2 — small camera tilt (horizontal camera variation)
    if tilt_angle != 0.0:
        data = _camera_tilt_raw(data, tilt_angle)

    # 3 — scale variation (simulate different person distances)
    if scale_factor != 1.0:
        data = _scale_augment_raw(data, scale_factor)

    # 3 — aspect ratio (before normalising)
    ar = bbox_aspect_ratio(data)                  # (T, 1)

    # 4 — Moving-Average
    data = temporal_ma(data, MA_WINDOW)

    # 5 — Hip-centred normalisation
    normed = hip_centered_normalize(data)         # (T, 17, 2)

    # 6 — flip
    if do_flip:
        normed = _flip_centred(normed)

    # 7 — velocity & acceleration
    T, K, _ = normed.shape
    delta = np.zeros_like(normed)
    delta[1:] = normed[1:] - normed[:-1]
    vel = np.linalg.norm(delta, axis=-1)          # (T, 17)
    acc = np.zeros_like(vel)
    acc[1:] = np.abs(vel[1:] - vel[:-1])

    features = np.concatenate([
        normed,                                   # (T,17,2)
        vel[..., np.newaxis],                     # (T,17,1)
        acc[..., np.newaxis],                     # (T,17,1)
    ], axis=-1)                                   # (T,17,4)

    flat = features.reshape(T, -1)                # (T, 68)
    full = np.concatenate([flat, ar], axis=-1)    # (T, 69)

    # 9 — noise
    if noise_sigma > 0:
        full = full + np.random.normal(0, noise_sigma,
                                       full.shape).astype(np.float32)

    return full.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation primitives cho camera ngang
# ─────────────────────────────────────────────────────────────────────────────
def _camera_tilt_raw(data: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Simulate camera mounted slightly above/below eye level on a wall.
    Compresses Y axis around frame center by a small angle (±5–10°).
    angle_deg > 0: camera hơi nhìn xuống; < 0: hơi nhìn lên.
    """
    angle = np.radians(abs(angle_deg))
    cos_a = np.cos(angle)
    out = data.copy()
    for t in range(data.shape[0]):
        center_y = 0.5
        out[t, :, 1] = center_y + (data[t, :, 1] - center_y) * cos_a
    return np.clip(out, 0.0, 1.0)


def _scale_augment_raw(data: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scale keypoints around the skeleton centre to simulate
    a person standing at different distances from the camera.
    scale_factor < 1: người xa (nhỏ hơn); > 1: người gần (lớn hơn).
    """
    out = data.copy()
    for t in range(data.shape[0]):
        valid = ~np.all(data[t] == 0, axis=-1)
        if valid.sum() < 2:
            continue
        cx = data[t, valid, 0].mean()
        cy = data[t, valid, 1].mean()
        out[t, :, 0] = cx + (data[t, :, 0] - cx) * scale_factor
        out[t, :, 1] = cy + (data[t, :, 1] - cy) * scale_factor
    return np.clip(out, 0.0, 1.0)


def _flip_centred(normed: np.ndarray) -> np.ndarray:
    """Horizontal flip on hip-centred coords (negate x, swap L/R)."""
    out = normed.copy()
    out[:, :, 0] = -out[:, :, 0]
    for l, r in COCO_FLIP_PAIRS:
        out[:, [l, r], :] = out[:, [r, l], :]
    return out


def _time_warp(data_raw: np.ndarray,
               factor_range: tuple = (0.85, 1.15)) -> np.ndarray:
    """Resample temporal length by a random speed factor."""
    T = data_raw.shape[0]
    factor = np.random.uniform(*factor_range)
    new_T = max(5, int(T * factor))
    src = np.linspace(0, T - 1, new_T)
    warped = np.zeros((new_T, *data_raw.shape[1:]), dtype=np.float32)
    for k in range(data_raw.shape[1]):
        for c in range(data_raw.shape[2]):
            warped[:, k, c] = np.interp(src, np.arange(T), data_raw[:, k, c])
    return warped


# ─────────────────────────────────────────────────────────────────────────────
#  Lying-phase detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_lying_start(data_raw: np.ndarray) -> int | None:
    """
    Return the first frame index where the person is consistently lying
    (low skeleton height) until the end of the sequence.
    Returns None if no clear lying phase.
    """
    T = data_raw.shape[0]
    mid_ankle = (data_raw[:, L_ANKLE, :] + data_raw[:, R_ANKLE, :]) / 2.0
    skel_h = np.linalg.norm(data_raw[:, NOSE, :] - mid_ankle, axis=-1)
    skel_h = uniform_filter1d(skel_h, size=5, mode="nearest")

    # Walk backwards from the end counting contiguous "lying" frames
    count = 0
    for t in range(T - 1, -1, -1):
        if skel_h[t] < LYING_SKEL_H_THRESH:
            count += 1
        else:
            break

    if count >= LYING_MIN_FRAMES:
        return T - count
    return None


def _loop_pad(data: np.ndarray, min_len: int) -> np.ndarray:
    """Loop a short array until it reaches min_len."""
    if len(data) >= min_len:
        return data
    reps = (min_len // len(data)) + 1
    return np.tile(data, (reps,) + (1,) * (data.ndim - 1))[:min_len]


# ─────────────────────────────────────────────────────────────────────────────
#  Sliding window
# ─────────────────────────────────────────────────────────────────────────────
def sliding_window(features: np.ndarray, seq_len: int,
                   stride: int) -> list[np.ndarray]:
    """Extract overlapping windows of shape (seq_len, D).
    Short inputs are zero-padded to produce at least one window."""
    T = features.shape[0]
    D = features.shape[1:]

    if T < seq_len:
        pad = np.zeros((seq_len - T, *D), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)
        T = seq_len

    windows = []
    start = 0
    while start + seq_len <= T:
        windows.append(features[start:start + seq_len])
        start += stride
    return windows


# ─────────────────────────────────────────────────────────────────────────────
#  Build metadata (file-level)
# ─────────────────────────────────────────────────────────────────────────────
def build_file_list(pose_dir: str, multicam_csv: str,
                    horizontal_only: bool = True) -> list[dict]:
    """
    Return file-level record list.
    horizontal_only=True: chỉ dùng UR Fall (camera ngang), bỏ Multicam (góc cao).
    """
    records: list[dict] = []
    pdir = Path(pose_dir)

    # ── UR Fall (cam0 – camera ngang) ─────────────────────────────────────
    for npy in sorted(pdir.glob("ur_fall_*.npy")):
        stem = npy.stem[len("ur_fall_"):]         # e.g. adl-07-cam0-rgb
        lbl, is_fall = ur_fall_label(stem)
        if lbl is None:
            continue
        records.append(dict(source="UR_Fall", action_id=stem,
                            npy_path=str(npy), base_label=lbl,
                            is_fall=is_fall))

    # ── Multicam (cam1 – góc cao) — CHỈ dùng khi không phải horizontal mode ─
    if not horizontal_only:
        meta_path = Path(multicam_csv)
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            for _, row in df.iterrows():
                mc_lbl = int(row["label_id"])      # 0=Fall, 1=Walking
                records.append(dict(source="Multicam",
                                    action_id=row["action_id"],
                                    npy_path=row["npy_path"],
                                    base_label=mc_lbl,
                                    is_fall=(mc_lbl == 0)))
        print("[INFO] Multicam data INCLUDED (không phải horizontal-only mode)")
    else:
        print("[INFO] Multicam data EXCLUDED — camera ngang mode (UR Fall only)")
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  Per-sequence processing with augmentation variants
# ─────────────────────────────────────────────────────────────────────────────
def _make_variants(data_raw: np.ndarray, label_id: int,
                   n_aug: int, rng: np.random.Generator,
                   ) -> list[tuple[np.ndarray, str]]:
    """
    Augmentation cho camera NGANG:
      - Small tilt (±5–10°): camera trên tường hơi nghiêng
      - Scale (0.75–1.25): người đứng xa/gần
      - Horizontal flip, noise, time-warp
    """
    variants: list[tuple[np.ndarray, str]] = []

    # ── ORIGINAL ───────────────────────────────────────────────────────────
    variants.append((compute_features(data_raw), "orig"))

    # ── HORIZONTAL FLIP ────────────────────────────────────────────────────
    variants.append((compute_features(data_raw, do_flip=True), "flip"))

    # ── CAMERA TILT variants (±5–10°) ─────────────────────────────────────
    for tilt in [rng.uniform(3, 10), rng.uniform(-10, -3)]:
        variants.append((
            compute_features(data_raw, tilt_angle=tilt),
            f"tilt{tilt:.0f}",
        ))

    # ── SCALE variants (xa/gần camera) ────────────────────────────────────
    for sc in [rng.uniform(0.75, 0.9), rng.uniform(1.1, 1.25)]:
        variants.append((
            compute_features(data_raw, scale_factor=sc),
            f"sc{sc:.2f}",
        ))

    # ── EXTRA — noise + time-warp + tilt + scale + flip ────────────────────
    for i in range(n_aug):
        raw = data_raw.copy()
        tag_parts = []
        if rng.random() > 0.3:
            raw = _time_warp(raw)
            tag_parts.append("tw")
        tilt  = rng.uniform(-8, 8)  if rng.random() > 0.4 else 0.0
        sc    = rng.uniform(0.8, 1.2) if rng.random() > 0.4 else 1.0
        do_f  = rng.random() > 0.5
        ns    = rng.uniform(0.003, 0.015) if rng.random() > 0.3 else 0.0
        if tilt != 0.0: tag_parts.append("t")
        if sc != 1.0:   tag_parts.append("s")
        if do_f:        tag_parts.append("f")
        if ns > 0:      tag_parts.append("n")
        variants.append((
            compute_features(raw, tilt_angle=tilt, scale_factor=sc,
                             do_flip=do_f, noise_sigma=ns),
            f"aug{i}_{'_'.join(tag_parts) or 'base'}",
        ))

    return variants


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_list = build_file_list(args.pose_dir, args.multicam_csv,
                                horizontal_only=args.horizontal_only)
    print(f"\n[INFO] Raw sequence files found: {len(file_list)}")

    # Count per base_label to decide augmentation multiplier
    from collections import Counter
    base_counts = Counter(r["base_label"] for r in file_list)
    max_count = max(base_counts.values())
    # For Lying_Down (label 4) we'll handle it specially
    print(f"  Base label counts (before split): {dict(base_counts)}")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    info_list: list[tuple] = []

    lying_raw_segments: list[tuple[np.ndarray, str]] = []  # raw lying segments

    for rec in file_list:
        npy_path   = rec["npy_path"]
        base_label = rec["base_label"]
        is_fall    = rec["is_fall"]

        try:
            data_raw = np.load(npy_path).astype(np.float32)
        except Exception as e:
            print(f"[WARN] Cannot load {npy_path}: {e}")
            continue
        if data_raw.ndim != 3 or data_raw.shape[1] != 17 or data_raw.shape[2] != 2:
            print(f"[WARN] Bad shape {data_raw.shape} — skip {npy_path}")
            continue

        # Skip mostly-zero sequences
        zero_ratio = np.all(data_raw == 0, axis=(1, 2)).mean()
        if zero_ratio > 0.6:
            print(f"[SKIP] {Path(npy_path).name} zero_ratio={zero_ratio:.2f}")
            continue

        # ── FALL sequences: split into Fall + Lying phases ─────────────────
        if is_fall and rec["source"] == "UR_Fall":
            lying_start = detect_lying_start(data_raw)

            # --- Fall portion (everything up to lying_start or full seq) ---
            if lying_start is not None and lying_start > 30:
                fall_raw = data_raw[:lying_start]
            else:
                fall_raw = data_raw  # use whole seq as Fall

            n_aug_fall = max(1, round(max_count / max(base_counts[0], 1)) - 1)
            n_aug_fall = min(n_aug_fall, 3)
            for feat, tag in _make_variants(fall_raw, 0, n_aug_fall, rng):
                for w in sliding_window(feat, args.seq_len, args.stride):
                    X_list.append(w)
                    y_list.append(0)
                    info_list.append((rec["source"], rec["action_id"], f"fall_{tag}"))

            # --- Lying portion (if exists) ---
            if lying_start is not None:
                lying_raw = data_raw[lying_start:]
                if len(lying_raw) >= LYING_MIN_FRAMES:
                    lying_raw_segments.append((lying_raw, rec["action_id"]))

        # ── MULTICAM fall sequences → keep as Fall (no lying in short clips) ─
        elif is_fall and rec["source"] == "Multicam":
            n_aug = max(1, round(max_count / max(base_counts[0], 1)) - 1)
            n_aug = min(n_aug, 3)
            for feat, tag in _make_variants(data_raw, 0, n_aug, rng):
                for w in sliding_window(feat, args.seq_len, args.stride):
                    X_list.append(w)
                    y_list.append(0)
                    info_list.append((rec["source"], rec["action_id"], tag))

        # ── NON-FALL sequences (Walking, Sitting, Bending) ────────────────
        else:
            this_count = base_counts.get(base_label, 1)
            n_aug = max(1, round(max_count / max(this_count, 1)) - 1)
            n_aug = min(n_aug, 5)  # cap for minority classes
            for feat, tag in _make_variants(data_raw, base_label, n_aug, rng):
                for w in sliding_window(feat, args.seq_len, args.stride):
                    X_list.append(w)
                    y_list.append(base_label)
                    info_list.append((rec["source"], rec["action_id"], tag))

    # ── Process Lying_Down segments ────────────────────────────────────────
    print(f"\n[INFO] Lying segments collected: {len(lying_raw_segments)}")
    # Lying needs aggressive augmentation since data is scarce
    target_lying_windows = max(200, len(X_list) // 5)   # aim for ~20% of total
    lying_windows_so_far = 0
    aug_rounds = 0

    while lying_windows_so_far < target_lying_windows and aug_rounds < 15:
        for lying_raw, action_id in lying_raw_segments:
            # Loop-pad to at least SEQ_LEN frames
            padded = _loop_pad(lying_raw, args.seq_len + 32)

            # Time-warp variant
            if aug_rounds > 0 and rng.random() > 0.3:
                padded_tw = _time_warp(padded)
            else:
                padded_tw = padded

            tilt = rng.uniform(-8, 8) if rng.random() > 0.3 else 0.0
            sc   = rng.uniform(0.8, 1.2) if rng.random() > 0.4 else 1.0
            do_f = rng.random() > 0.5
            ns   = rng.uniform(0.003, 0.015) if aug_rounds > 0 else 0.0

            feat = compute_features(padded_tw, tilt_angle=tilt,
                                    scale_factor=sc, do_flip=do_f,
                                    noise_sigma=ns)
            for w in sliding_window(feat, args.seq_len, args.stride):
                X_list.append(w)
                y_list.append(4)  # Lying_Down
                info_list.append(("UR_Fall", action_id,
                                  f"lying_aug{aug_rounds}"))
                lying_windows_so_far += 1

        aug_rounds += 1

    print(f"  Lying_Down windows generated: {lying_windows_so_far}")

    # ── Compile arrays ─────────────────────────────────────────────────────
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    info_list = [info_list[i] for i in perm]

    # Save
    np.save(out_dir / "X_train.npy", X)
    np.save(out_dir / "y_train.npy", y)

    meta_df = pd.DataFrame(info_list,
                           columns=["source", "action_id", "aug_type"])
    meta_df["label_id"]   = y
    meta_df["label_name"] = meta_df["label_id"].map(LABEL_MAP)
    meta_df.to_csv(out_dir / "metadata_train.csv", index=False)

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  X_train : {X.shape}   y_train : {y.shape}")
    print(f"  Feature dim : {X.shape[-1]}")
    print(f"\n  Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        pct = 100 * c / len(y)
        print(f"    {LABEL_MAP[u]:20s} (label {u}): {c:5d}  ({pct:5.1f}%)")
    print(f"\n  Saved to: {out_dir}/")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("data_prepare_v3 — horizontal camera 5-class preprocessing")
    p.add_argument("--pose_dir",       default=POSE_DIR)
    p.add_argument("--multicam_csv",   default="./data/processed_pose/metadata_pose_final.csv")
    p.add_argument("--out_dir",        default=OUT_DIR)
    p.add_argument("--seq_len",        type=int, default=SEQ_LEN)
    p.add_argument("--stride",         type=int, default=STRIDE)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--horizontal_only", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Chỉ dùng UR Fall (camera ngang). Dùng --no-horizontal_only để thêm Multicam")
    main(p.parse_args())
