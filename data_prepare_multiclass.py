"""
data_prepare_multiclass.py – Preprocess data for 5-class action recognition
=============================================================================
5 classes based on available data:
  0: Fall           – UR_Fall fall-* + Multicam Fall segments
  1: Walking        – UR_Fall adl-01..08 + Multicam non-Fall segments
  2: Bending        – UR_Fall adl-09..16
  3: Sitting_Standing – UR_Fall adl-17..30 (sitting/standing + getting-up)
  4: Jogging_Jumping – UR_Fall adl-31..40

(Note: "Stairs" class excluded — no data available in either dataset)

Output: ./data/train_ready_5class/
  - X_train.npy  shape (N, 128, 68)   — 17 kpts × 4 features (x,y,vel,acc)
  - y_train.npy  shape (N,)
  - metadata_train.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
POSE_DIR   = "./data/processed_pose"
OUT_DIR    = "./data/train_ready_5class"
SEQ_LEN    = 128
STRIDE     = 32      # 75% overlap → more samples per raw sequence
FEATURE_DIM = 68     # 17 kpts × 4: x, y, velocity, acceleration

LABEL_MAP_5 = {
    0: "Fall",
    1: "Walking",
    2: "Bending",
    3: "Sitting_Standing",
    4: "Jogging_Jumping",
}

# UR_Fall ADL grouping (standard UR Fall Detection Dataset literature mapping)
# adl-01..08  → Walking        (ADL group 1)
# adl-09..16  → Bending        (ADL group 2)
# adl-17..30  → Sitting_Stand  (ADL groups 3+4: sitting/standing, getting up)
# adl-31..40  → Jogging_Jump   (ADL group 5)
def ur_fall_adl_to_label(seq_name: str) -> int | None:
    """Map UR_Fall sequence folder name to 5-class label. Returns None to skip."""
    name = seq_name.lower()
    if name.startswith("fall"):
        return 0
    if name.startswith("adl"):
        # parse number e.g. adl-07-cam0-rgb → 7
        try:
            num = int(name.split("-")[1])
        except (IndexError, ValueError):
            return None
        if 1 <= num <= 8:
            return 1   # Walking
        elif 9 <= num <= 16:
            return 2   # Bending
        elif 17 <= num <= 30:
            return 3   # Sitting_Standing
        elif 31 <= num <= 40:
            return 4   # Jogging_Jumping
    return None

# COCO keypoint flip pairs for horizontal augmentation
COCO_FLIP_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]

# ─────────────────────────────────────────────────────────────────────────────
#  Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def interpolate_zeros(data: np.ndarray) -> np.ndarray:
    """Linear-interpolate frames where all keypoints are zero."""
    # data shape: (T, 17, 2)
    T = data.shape[0]
    zero_mask = np.all(data == 0, axis=(1, 2))
    if not zero_mask.any():
        return data
    result = data.copy()
    idx = np.arange(T)
    valid = ~zero_mask
    if valid.sum() < 2:
        return result  # can't interpolate
    for k in range(data.shape[1]):
        for c in range(data.shape[2]):
            result[:, k, c] = np.interp(idx, idx[valid], data[valid, k, c])
    return result


def compute_motion_features(data: np.ndarray) -> np.ndarray:
    """
    (T, 17, 2) → (T, 17, 4) by appending scalar velocity + acceleration.
    Features per keypoint: [x, y, v, a] where v = Euclidean speed, a = |Δv|
    """
    T, K, _ = data.shape

    # Delta positions, (T, 17, 2)
    delta_pos = np.zeros_like(data)
    delta_pos[1:] = data[1:] - data[:-1]

    # Scalar velocity: Euclidean magnitude, (T, 17)
    velocity = np.linalg.norm(delta_pos, axis=-1)

    # Scalar acceleration: |Δv|, (T, 17)
    delta_vel = np.zeros_like(velocity)
    delta_vel[1:] = np.abs(velocity[1:] - velocity[:-1])

    # Zero out padded frames
    is_pad = np.all(data == 0, axis=-1)  # (T, 17)
    velocity[is_pad] = 0.0
    delta_vel[is_pad] = 0.0

    result = np.concatenate([
        data,                           # (T, 17, 2) — x, y
        velocity[..., np.newaxis],      # (T, 17, 1) — v
        delta_vel[..., np.newaxis],     # (T, 17, 1) — a
    ], axis=-1)  # (T, 17, 4)

    return result.astype(np.float32)


def horizontal_flip(data: np.ndarray) -> np.ndarray:
    """
    Flip skeleton horizontally: mirror x coord + swap L/R keypoint pairs.
    data: (T, 17, C) — C can be 2 or 4 (with scalar motion features).
    Only x coordinate (index 0) is mirrored; scalar v,a remain unchanged.
    """
    flipped = data.copy()
    flipped[:, :, 0] = 1.0 - flipped[:, :, 0]   # mirror x
    for l, r in COCO_FLIP_PAIRS:
        flipped[:, [l, r], :] = flipped[:, [r, l], :]
    return flipped


def add_noise(data: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    return data + np.random.normal(0, sigma, data.shape).astype(np.float32)


def time_warp(data: np.ndarray, factor_range=(0.85, 1.15)) -> np.ndarray:
    """Resample sequence length by random speed factor."""
    T = data.shape[0]
    factor = np.random.uniform(*factor_range)
    new_T = int(T * factor)
    if new_T < 5:
        return data
    src_idx = np.linspace(0, T - 1, new_T)
    warped = np.zeros((new_T, *data.shape[1:]), dtype=np.float32)
    for k in range(data.shape[1]):
        for c in range(data.shape[2]):
            warped[:, k, c] = np.interp(src_idx, np.arange(T), data[:, k, c])
    return warped


# ─────────────────────────────────────────────────────────────────────────────
#  Sliding Window
# ─────────────────────────────────────────────────────────────────────────────
def sliding_window(data: np.ndarray, seq_len: int, stride: int) -> list:
    """
    Extract fixed-length windows from (T, K, C) array.
    Short sequences are zero-padded at the end to produce at least one window.
    """
    T = data.shape[0]
    rest_shape = data.shape[1:]

    if T < seq_len:
        # pad with zeros
        pad = np.zeros((seq_len - T, *rest_shape), dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)
        T = seq_len

    windows = []
    start = 0
    while start + seq_len <= T:
        windows.append(data[start: start + seq_len])
        start += stride
    return windows


# ─────────────────────────────────────────────────────────────────────────────
#  Build metadata
# ─────────────────────────────────────────────────────────────────────────────
def build_metadata(pose_dir: str, multicam_csv: str) -> pd.DataFrame:
    """
    Combine UR_Fall .npy files + Multicam .npy files into single metadata df.
    Returns DataFrame with columns: action_id, label_id, label_name, npy_path
    """
    records = []
    pose_path = Path(pose_dir)

    # ── UR_Fall sequences ──────────────────────────────────────────────────
    for npy in sorted(pose_path.glob("ur_fall_*.npy")):
        # filename like  ur_fall_adl-07-cam0-rgb.npy
        stem = npy.stem[len("ur_fall_"):]   # e.g. adl-07-cam0-rgb
        label_id = ur_fall_adl_to_label(stem)
        if label_id is None:
            continue
        records.append({
            "source": "UR_Fall",
            "action_id": stem,
            "label_id": label_id,
            "label_name": LABEL_MAP_5[label_id],
            "npy_path": str(npy),
        })

    # ── Multicam sequences ─────────────────────────────────────────────────
    if os.path.exists(multicam_csv):
        # Multicam npy naming: multicam_{action_id}.npy
        # label in CSV: 1→Fall(0), 0→Walking(1)
        mc_df = pd.read_csv(multicam_csv)
        for npy in sorted(pose_path.glob("multicam_*.npy")):
            stem = npy.stem[len("multicam_"):]     # chute01_cam1_f1052-1082
            # look up label in CSV-derived metadata
            row = mc_df[mc_df["action_id"] == stem] if "action_id" in mc_df.columns else pd.DataFrame()
            if len(row) == 0:
                # fall back: infer from the pose metadata csv
                continue
            mc_lbl_raw = int(row.iloc[0]["label_id"])
            # Multicam only has Fall(0) or Walking(1)
            records.append({
                "source": "Multicam",
                "action_id": stem,
                "label_id": mc_lbl_raw,
                "label_name": LABEL_MAP_5.get(mc_lbl_raw, "Unknown"),
                "npy_path": str(npy),
            })
    else:
        # Fall back: use existing metadata_pose_final.csv
        meta_csv = Path(pose_dir) / "metadata_pose_final.csv"
        if meta_csv.exists():
            df = pd.read_csv(meta_csv)
            for _, row in df.iterrows():
                records.append({
                    "source": row["source"],
                    "action_id": row["action_id"],
                    "label_id": int(row["label_id"]),
                    "label_name": LABEL_MAP_5.get(int(row["label_id"]), "Walking"),
                    "npy_path": row["npy_path"],
                })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build file-level metadata ──────────────────────────────────────────
    meta = build_metadata(args.pose_dir, args.multicam_csv)
    print(f"\n[INFO] Total raw sequences found: {len(meta)}")
    print(meta["label_name"].value_counts().to_string())

    # ── Process each sequence ──────────────────────────────────────────────
    X_list, y_list, info_list = [], [], []

    for _, row in meta.iterrows():
        npy_path = row["npy_path"]
        label_id = int(row["label_id"])
        label_name = row["label_name"]

        # Load
        try:
            data = np.load(npy_path).astype(np.float32)  # (T, 17, 2)
        except Exception as e:
            print(f"[WARN] Cannot load {npy_path}: {e}")
            continue

        if data.ndim != 3 or data.shape[1] != 17 or data.shape[2] != 2:
            print(f"[WARN] Unexpected shape {data.shape} — skip {npy_path}")
            continue

        # Interpolate zero frames
        data = interpolate_zeros(data)

        # Skip sequences that are still mostly zero after interpolation
        zero_ratio = np.all(data == 0, axis=(1, 2)).mean()
        if zero_ratio > 0.6:
            print(f"[SKIP] {npy_path.split('/')[-1]}  zero_ratio={zero_ratio:.2f}")
            continue

        # Compute motion features → (T, 17, 4) → flatten → (T, 68)
        data_feat = compute_motion_features(data)        # (T, 17, 4)

        # Determine how many variations to generate for minority classes
        # more augmentations for classes with fewer sequences
        label_counts = meta["label_id"].value_counts().to_dict()
        max_count = max(label_counts.values())
        this_count = label_counts.get(label_id, 1)
        # reps: minority classes get up to 4 extra augmented versions
        extra_reps = max(1, min(4, round(max_count / max(this_count, 1)) - 1))

        # ── Original windows ──
        windows_orig = sliding_window(data_feat, SEQ_LEN, args.stride)
        for w in windows_orig:
            flat = w.reshape(SEQ_LEN, -1)
            X_list.append(flat)
            y_list.append(label_id)
            info_list.append((row["source"], row["action_id"], "orig"))

        # ── Horizontal flip ──
        data_flip = horizontal_flip(data_feat)
        for w in sliding_window(data_flip, SEQ_LEN, args.stride):
            flat = w.reshape(SEQ_LEN, -1)
            X_list.append(flat)
            y_list.append(label_id)
            info_list.append((row["source"], row["action_id"], "flip"))

        # ── Extra augmentations for minority classes ──
        for rep in range(extra_reps):
            # Choose random combo: noise + time_warp  ±flip
            augmented = data_feat.copy()
            if rng.random() > 0.3:
                augmented = add_noise(augmented, sigma=0.005 + rep * 0.002)
            if rng.random() > 0.3:
                augmented = time_warp(augmented)
            if rng.random() > 0.5:
                augmented = horizontal_flip(augmented)
            for w in sliding_window(augmented, SEQ_LEN, args.stride):
                flat = w.reshape(SEQ_LEN, -1)
                X_list.append(flat)
                y_list.append(label_id)
                info_list.append((row["source"], row["action_id"], f"aug{rep}"))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # ── Shuffle ──────────────────────────────────────────────────────────
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    info_list = [info_list[i] for i in perm]

    # ── Save ─────────────────────────────────────────────────────────────
    np.save(out_dir / "X_train.npy", X)
    np.save(out_dir / "y_train.npy", y)

    meta_out = pd.DataFrame(info_list, columns=["source", "action_id", "aug_type"])
    meta_out["label_id"] = y
    meta_out["label_name"] = meta_out["label_id"].map(LABEL_MAP_5)
    meta_out.to_csv(out_dir / "metadata_train.csv", index=False)

    print(f"\n[DONE] X_train: {X.shape}  y_train: {y.shape}")
    print("\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {LABEL_MAP_5[u]:20s} (label {u}): {c:4d} samples")

    print(f"\nOutput saved to: {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("5-class action recognition preprocessing")
    p.add_argument("--pose_dir",     default=POSE_DIR)
    p.add_argument("--multicam_csv", default="./data/processed_pose/metadata_pose_final.csv")
    p.add_argument("--out_dir",      default=OUT_DIR)
    p.add_argument("--seq_len",      type=int, default=SEQ_LEN)
    p.add_argument("--stride",       type=int, default=STRIDE)
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()
    SEQ_LEN   = args.seq_len
    main(args)
