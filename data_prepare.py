"""
data_prepare.py – BƯỚC 2: KHẮC PHỤC & TIỀN XỬ LÝ DỮ LIỆU (v2 NÂNG CẤP)
==========================================================================
Pipeline xử lý:
  1. Loại bỏ file toàn zero (không thể khôi phục)
  2. Interpolation — lấp zero keypoints bằng nội suy tuyến tính
  3. Normalization — đảm bảo [0, 1]
  4. Motion Features — tính Vận tốc (v) và Gia tốc (a) → (T, 17, 4)
  5. Horizontal Flip Augmentation — lật gương nhân đôi dữ liệu
  6. Sliding Window — cắt thành đoạn 128 frames, overlap 50%
  7. Padding — post-pad mẫu ngắn hơn 128 frames

Output:
  - ./data/train_ready/X_train.npy   shape (N, 128, 68)  [17 kpts × 4 feats]
  - ./data/train_ready/y_train.npy   shape (N,)
  - ./data/train_ready/metadata_train.csv

Usage:
    python data_prepare.py
"""

from __future__ import annotations

import glob
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
POSE_DIR = "./data/processed_pose"
METADATA_CSV = os.path.join(POSE_DIR, "metadata_pose_final.csv")
OUTPUT_DIR = "./data/train_ready"

SEQ_LEN = 128          # Chiều dài chuỗi đích
STRIDE = 32            # Bước nhảy (overlap = 75%) — tăng số mẫu đầng kể
NUM_KEYPOINTS = 17
NUM_COORDS = 2         # (x, y) — chiều dài raw input
NUM_MOTION_FEATS = 4   # (x, y, v, a) — sau khi thêm motion features
FEATURE_DIM = NUM_KEYPOINTS * NUM_MOTION_FEATS  # 68

ZERO_THRESHOLD = 0.90  # Loại file có >90% frame toàn zero (không đủ data nội suy)

# COCO 17 keypoints — cặp left/right để swap khi lật gương
# (left_eye,right_eye), (left_ear,right_ear), (left_shoulder,right_shoulder),
# (left_elbow,right_elbow), (left_wrist,right_wrist),
# (left_hip,right_hip), (left_knee,right_knee), (left_ankle,right_ankle)
COCO_FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

LABEL_MAP = {
    0: "Fall",
    1: "Walking",
    2: "Stairs",
    3: "Sitting_Standing",
    4: "Bending",
    5: "Jogging_Jumping",
}


def infer_label_from_filename(fname: str) -> Tuple[int, str]:
    """Suy ra label từ tên file."""
    name = os.path.basename(fname).replace(".npy", "")
    if name.startswith("ur_fall_fall"):
        return 0, "Fall"
    elif name.startswith("ur_fall_adl"):
        return 1, "Walking"
    return -1, "Unknown"


def build_full_metadata() -> pd.DataFrame:
    """Xây dựng metadata đầy đủ bao gồm cả UR_Fall."""
    meta_rows = []
    path_to_row = {}

    if os.path.exists(METADATA_CSV):
        df_csv = pd.read_csv(METADATA_CSV)
        for _, row in df_csv.iterrows():
            npy_path = os.path.normpath(row["npy_path"])
            path_to_row[npy_path] = row.to_dict()

    all_npy = sorted(glob.glob(os.path.join(POSE_DIR, "*.npy")))

    for fpath in all_npy:
        norm_path = os.path.normpath(fpath)
        if norm_path in path_to_row:
            meta_rows.append(path_to_row[norm_path])
        else:
            label_id, label_name = infer_label_from_filename(fpath)
            name = os.path.basename(fpath).replace(".npy", "")
            source = "UR_Fall" if "ur_fall" in name else "Unknown"
            meta_rows.append({
                "source": source,
                "action_id": name,
                "label_name": label_name,
                "label_id": label_id,
                "npy_path": fpath,
            })

    return pd.DataFrame(meta_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: Interpolation — Lấp zero keypoints
# ─────────────────────────────────────────────────────────────────────────────
def interpolate_zeros(data: np.ndarray) -> np.ndarray:
    """
    Nội suy tuyến tính cho các keypoints bị (0,0).

    data: shape (T, 17, 2)
    Với mỗi keypoint riêng biệt:
      - Tìm các frame có giá trị = 0
      - Nội suy tuyến tính dựa trên frame trước & sau gần nhất
      - Nếu đầu chuỗi bị zero → forward-fill từ frame hợp lệ đầu tiên
      - Nếu cuối chuỗi bị zero → backward-fill từ frame hợp lệ cuối

    Returns: data đã được nội suy, shape giữ nguyên
    """
    result = data.copy()
    T, K, C = result.shape  # T frames, 17 keypoints, 2 coords

    for k in range(K):
        for c in range(C):
            series = result[:, k, c]

            # Xác định frame nào bị "zero" cho keypoint này
            # Một keypoint (x,y) bị zero khi CẢ x VÀ y đều = 0
            is_zero = np.all(result[:, k, :] == 0, axis=-1)  # (T,)

            if not np.any(is_zero):
                continue  # Keypoint này OK

            if np.all(is_zero):
                continue  # Toàn bộ bị zero, không nội suy được

            # Lấy index các frame hợp lệ và frame bị zero
            valid_idx = np.where(~is_zero)[0]
            zero_idx = np.where(is_zero)[0]

            # Nội suy bằng np.interp (linear interpolation)
            series_valid = series[valid_idx]
            result[zero_idx, k, c] = np.interp(zero_idx, valid_idx, series_valid)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: Normalization — Đảm bảo giá trị nằm trong [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
def normalize_to_01(data: np.ndarray) -> np.ndarray:
    """Clip giá trị về [0, 1]. Data đã được normalize từ extract_pose.py."""
    return np.clip(data, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: Motion Features — Tính Vận tốc & Gia tốc
# ─────────────────────────────────────────────────────────────────────────────
def compute_motion_features(data: np.ndarray) -> np.ndarray:
    """
    Tính toán Vận tốc (v) và Gia tốc (a) cho từng keypoint.

    data: shape (T, 17, 2)  — (x, y) đã normalize về [0,1]
    Returns: shape (T, 17, 4)  — (x, y, v, a) cho mỗi keypoint

    Công thức:
      v[t] = ||pos[t] - pos[t-1]||₂  (tốc độ Euclidean, frame t so với t-1)
      a[t] = |v[t] - v[t-1]|          (biến thiên tốc độ)
      v[0] = 0 và a[0] = 0 (frame đầu tiên)
    """
    T, K, _ = data.shape

    # Tính delta vị trí: (T, 17, 2), delta[0] = 0
    delta_pos = np.zeros_like(data)              # (T, 17, 2)
    delta_pos[1:] = data[1:] - data[:-1]        # frame diff

    # Vận tốc: độ lớn Euclidean, shape (T, 17)
    velocity = np.linalg.norm(delta_pos, axis=-1)  # (T, 17)

    # Gia tốc: biến thiên vận tốc, shape (T, 17)
    delta_vel = np.zeros_like(velocity)          # (T, 17)
    delta_vel[1:] = np.abs(velocity[1:] - velocity[:-1])

    # Ghép lại: (T, 17, 4) = (x, y, v, a)
    result = np.concatenate([
        data,                    # (T, 17, 2) — x, y
        velocity[..., np.newaxis],    # (T, 17, 1) — v
        delta_vel[..., np.newaxis],   # (T, 17, 1) — a
    ], axis=-1)  # (T, 17, 4)

    # Padding frames (xy==0) → set v và a về 0 để không gây noise
    is_pad = np.all(data == 0, axis=-1)  # (T, 17), True nếu kpt bị pad
    result[is_pad, 2] = 0.0
    result[is_pad, 3] = 0.0

    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4: Horizontal Flip Augmentation — Lật gương trái/phải
# ─────────────────────────────────────────────────────────────────────────────
def horizontal_flip(data: np.ndarray) -> np.ndarray:
    """
    Lật gương toàn bộ chuỗi pose theo trục x.

    data: shape (T, 17, 4)  — (x, y, v, a)
    Returns: shape (T, 17, 4) đã lật gương

    Quy tắc:
      - x_flip = 1 - x  (toạ độ đã normalize về [0,1])
      - y, v, a giữ nguyên  (tốc độ/gia tốc là scalar nên không đổi dấu)
      - Hoán đổi keypoints left ↔ right theo COCO_FLIP_PAIRS
    """
    result = data.copy()  # (T, 17, 4)

    # Lật x coordinate
    result[:, :, 0] = 1.0 - data[:, :, 0]

    # Giữ nguyên x=0 cho padding frames
    is_pad = np.all(data[:, :, :2] == 0, axis=-1)  # (T, 17)
    result[is_pad, 0] = 0.0

    # Hoán đổi left ↔ right keypoints
    for left, right in COCO_FLIP_PAIRS:
        result[:, [left, right], :] = result[:, [right, left], :].copy()

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Step 5: Sliding Window + Padding
# ─────────────────────────────────────────────────────────────────────────────
def sliding_window_with_padding(data: np.ndarray, seq_len: int,
                                 stride: int) -> List[np.ndarray]:
    """
    Cắt chuỗi thành các đoạn có chiều dài seq_len.
    - Nếu data ngắn hơn seq_len → pad zeros ở cuối (post-padding)
    - Nếu data dài hơn seq_len → sliding window với stride cho trước

    data: shape (T, 17, F) — F = 4 sau khi thêm motion features
    Returns: list of arrays, mỗi array shape (seq_len, 17, F)
    """
    T = data.shape[0]

    if T <= seq_len:
        # Post-padding — shape-agnostic
        padded = np.zeros((seq_len, *data.shape[1:]), dtype=np.float32)
        padded[:T] = data
        return [padded]

    # Sliding window
    windows = []
    start = 0
    while start + seq_len <= T:
        windows.append(data[start:start + seq_len].copy())
        start += stride

    # Đoạn cuối (nếu còn dư frames chưa được bao phủ)
    if start < T:
        windows.append(data[T - seq_len:T].copy())

    return windows


# ─────────────────────────────────────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  BƯỚC 2: KHẮC PHỤC & TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load metadata ─────────────────────────────────────────────────────
    df = build_full_metadata()
    print(f"\n📂 Tổng file .npy trong metadata: {len(df)}")

    all_X = []      # Sẽ chứa các mảng (128, 34)
    all_y = []      # Nhãn tương ứng
    all_meta = []   # Metadata

    n_skipped_zero = 0
    n_skipped_unk = 0
    n_interpolated = 0
    n_total_windows = 0
    original_samples = 0

    # ── Xử lý từng file ──────────────────────────────────────────────────
    for idx, row in df.iterrows():
        fpath = row["npy_path"]
        label_id = int(row["label_id"])
        label_name = row["label_name"]
        action_id = row["action_id"]
        source = row["source"]

        # Bỏ qua nhãn không xác định
        if label_id < 0:
            n_skipped_unk += 1
            continue

        try:
            data = np.load(fpath)  # (T, 17, 2)
        except Exception as e:
            print(f"  [SKIP] Lỗi đọc {fpath}: {e}")
            continue

        T = data.shape[0]
        if T == 0:
            continue

        # ── Check zero ratio ─────────────────────────────────────────────
        frame_is_zero = np.all(data == 0, axis=(1, 2))
        zero_ratio = frame_is_zero.sum() / T

        if zero_ratio >= ZERO_THRESHOLD:
            n_skipped_zero += 1
            continue

        original_samples += 1

        # ── Step 1: Interpolation ────────────────────────────────────────
        # Kiểm tra xem có zero keypoints không
        has_zeros = np.any(np.all(data == 0, axis=-1))
        if has_zeros:
            data = interpolate_zeros(data)
            n_interpolated += 1

        # ── Step 2: Normalization ─────────────────────────────────────────
        data = normalize_to_01(data)  # (T, 17, 2)

        # ── Step 3: Motion features → (T, 17, 4) ─────────────────────────
        data_feat = compute_motion_features(data)

        # ── Step 4: Horizontal flip augmentation ─────────────────────────
        # Sinh 2 phiên bản: gốc và lật gương
        versions = [
            (data_feat, "orig"),
            (horizontal_flip(data_feat), "flip"),
        ]

        for data_v, aug_tag in versions:
            # ── Step 5: Sliding Window + Padding ─────────────────────────
            windows = sliding_window_with_padding(data_v, SEQ_LEN, STRIDE)
            n_total_windows += len(windows)

            for w_idx, window in enumerate(windows):
                # Flatten (128, 17, 4) → (128, 68)
                flat = window.reshape(SEQ_LEN, FEATURE_DIM)
                all_X.append(flat)
                all_y.append(label_id)
                all_meta.append({
                    "source": source,
                    "action_id": action_id,
                    "label_name": label_name,
                    "label_id": label_id,
                    "window_idx": w_idx,
                    "aug": aug_tag,
                    "original_npy": os.path.basename(fpath),
                })

    # ── Stack & Save ──────────────────────────────────────────────────────
    X = np.array(all_X, dtype=np.float32)   # (N, 128, 34)
    y = np.array(all_y, dtype=np.int64)     # (N,)

    x_path = os.path.join(OUTPUT_DIR, "X_train.npy")
    y_path = os.path.join(OUTPUT_DIR, "y_train.npy")
    meta_path = os.path.join(OUTPUT_DIR, "metadata_train.csv")

    np.save(x_path, X)
    np.save(y_path, y)
    pd.DataFrame(all_meta).to_csv(meta_path, index=False)

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  BÁO CÁO TIỀN XỬ LÝ")
    print(f"{'─'*50}")
    print(f"  File gốc           : {len(df)}")
    print(f"  Bỏ qua (toàn zero) : {n_skipped_zero}")
    print(f"  Bỏ qua (nhãn lạ)   : {n_skipped_unk}")
    print(f"  Mẫu gốc hợp lệ    : {original_samples}")
    print(f"  Đã nội suy         : {n_interpolated} file")
    print(f"  Sliding window     : SEQ_LEN={SEQ_LEN}, STRIDE={STRIDE} (overlap 50%)")
    print(f"\n  📊 OUTPUT:")
    print(f"     X_train.npy shape : {X.shape}")
    print(f"     y_train.npy shape : {y.shape}")
    print(f"     Tổng mẫu training : {len(X)}")

    # Phân bố nhãn sau augmentation
    label_dist = Counter(all_y)
    print(f"\n  📊 PHÂN BỐ NHÃN (sau sliding window):")
    for lid in sorted(label_dist.keys()):
        name = LABEL_MAP.get(lid, "Unknown")
        cnt = label_dist[lid]
        pct = cnt / len(X) * 100
        print(f"     [{lid}] {name:20s}: {cnt:4d} ({pct:.1f}%)")

    # Kiểm tra final data quality
    print(f"\n  🔍 KIỂM TRA CHẤT LƯỢNG CUỐI:")
    print(f"     Min value: {X.min():.6f}")
    print(f"     Max value: {X.max():.6f}")
    zero_after = np.mean(X == 0)
    print(f"     % giá trị = 0 (bao gồm padding): {zero_after*100:.2f}%")

    # Kiểm tra phần không phải padding
    non_pad_mask = np.any(X != 0, axis=-1)  # (N, 128) — frame nào ko phải padding
    n_padded_frames = (~non_pad_mask).sum()
    n_total_frames = non_pad_mask.size
    print(f"     Padding frames: {n_padded_frames}/{n_total_frames} "
          f"({n_padded_frames/n_total_frames*100:.1f}%)")

    print(f"\n  💾 ĐÃ LƯU:")
    print(f"     {x_path}")
    print(f"     {y_path}")
    print(f"     {meta_path}")
    print(f"\n{'='*70}")
    print(f"  ✅ TIỀN XỬ LÝ HOÀN TẤT — SẴN SÀNG CHO BƯỚC 3 (TRAINING)!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
