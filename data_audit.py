"""
data_audit.py – BƯỚC 1: QUÉT & KHÁM BỆNH DỮ LIỆU (DATA AUDIT)
================================================================
Quét toàn bộ thư mục ./data/processed_pose/ và xuất báo cáo chi tiết:
  1. Tính toàn vẹn: tổng số mẫu, phân bố nhãn (0–5)
  2. Chất lượng: zero-frame files, mất keypoints >50%
  3. Thống kê độ dài: trung bình, ngắn nhất, dài nhất

Usage:
    python data_audit.py
"""

from __future__ import annotations

import glob
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
POSE_DIR = "./data/processed_pose"
METADATA_CSV = os.path.join(POSE_DIR, "metadata_pose_final.csv")

LABEL_MAP = {
    0: "Fall",
    1: "Walking",
    2: "Stairs",
    3: "Sitting_Standing",
    4: "Bending",
    5: "Jogging_Jumping",
}


def infer_label_from_filename(fname: str) -> tuple[int, str]:
    """Suy ra label từ tên file nếu không có trong metadata."""
    name = os.path.basename(fname).replace(".npy", "")
    if name.startswith("ur_fall_fall"):
        return 0, "Fall"
    elif name.startswith("ur_fall_adl"):
        return 1, "Walking"
    # Nếu không xác định được, trả về -1
    return -1, "Unknown"


def build_full_metadata() -> pd.DataFrame:
    """
    Xây dựng metadata đầy đủ cho tất cả .npy files,
    bao gồm cả UR_Fall (không có trong CSV gốc).
    """
    # Đọc metadata Multicam từ CSV
    meta_rows = []
    if os.path.exists(METADATA_CSV):
        df_csv = pd.read_csv(METADATA_CSV)
        # Chuẩn hoá đường dẫn
        path_to_row = {}
        for _, row in df_csv.iterrows():
            npy_path = os.path.normpath(row["npy_path"])
            path_to_row[npy_path] = row.to_dict()
    else:
        path_to_row = {}

    # Quét tất cả file .npy
    all_npy = sorted(glob.glob(os.path.join(POSE_DIR, "*.npy")))

    for fpath in all_npy:
        norm_path = os.path.normpath(fpath)
        if norm_path in path_to_row:
            meta_rows.append(path_to_row[norm_path])
        else:
            # File UR_Fall — suy ra label
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


def audit():
    print("=" * 70)
    print("  BƯỚC 1: QUÉT & KHÁM BỆNH DỮ LIỆU (DATA AUDIT)")
    print("=" * 70)

    # ── 1. Xây dựng metadata đầy đủ ──────────────────────────────────────
    df = build_full_metadata()
    total_samples = len(df)

    if total_samples == 0:
        print("\n[LỖI] Không tìm thấy file .npy nào trong", POSE_DIR)
        sys.exit(1)

    print(f"\n📂 Thư mục quét: {POSE_DIR}")
    print(f"📊 Tổng số file .npy: {total_samples}")

    # ── 2. Phân bố theo nguồn ────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  PHÂN BỐ THEO NGUỒN DỮ LIỆU")
    print(f"{'─'*50}")
    for src, cnt in df["source"].value_counts().items():
        print(f"  {src:20s} : {cnt:4d} mẫu")

    # ── 3. Phân bố nhãn (0–5) ────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  PHÂN BỐ THEO NHÃN (label_id 0–5)")
    print(f"{'─'*50}")
    label_counts = df["label_id"].value_counts().sort_index()
    for lid in range(6):
        cnt = label_counts.get(lid, 0)
        name = LABEL_MAP.get(lid, "Unknown")
        bar = "█" * (cnt // 1) if cnt > 0 else ""
        print(f"  [{lid}] {name:20s} : {cnt:4d}  {bar}")
    print(f"  {'':20s}   ─────")
    print(f"  {'TỔNG':20s} : {total_samples:4d}")

    # ── 4. Kiểm tra chất lượng từng file ─────────────────────────────────
    print(f"\n{'─'*50}")
    print("  KIỂM TRA CHẤT LƯỢNG DỮ LIỆU")
    print(f"{'─'*50}")

    lengths = []
    zero_frame_files = []       # file có tất cả frame = 0
    high_missing_files = []     # >50% keypoints bị mất
    corrupted_files = []        # file lỗi
    stats_by_label = defaultdict(list)
    all_zero_counts = []
    all_missing_ratios = []

    for idx, row in df.iterrows():
        fpath = row["npy_path"]
        label_id = row["label_id"]

        try:
            data = np.load(fpath)  # (num_frames, 17, 2)
        except Exception as e:
            corrupted_files.append((fpath, str(e)))
            continue

        n_frames = data.shape[0]
        lengths.append(n_frames)
        stats_by_label[label_id].append(n_frames)

        # Kiểm tra zero frames: frame nào mà TẤT CẢ 17 keypoints đều (0,0)
        frame_is_zero = np.all(data == 0, axis=(1, 2))  # (num_frames,)
        n_zero_frames = int(frame_is_zero.sum())
        zero_ratio = n_zero_frames / n_frames if n_frames > 0 else 0

        # Kiểm tra missing keypoints: keypoint nào có (0,0)
        kpt_is_zero = np.all(data == 0, axis=-1)  # (num_frames, 17)
        missing_ratio = kpt_is_zero.sum() / (n_frames * 17) if n_frames > 0 else 0

        all_zero_counts.append(n_zero_frames)
        all_missing_ratios.append(missing_ratio)

        if zero_ratio == 1.0:
            zero_frame_files.append((fpath, n_frames))
        if missing_ratio > 0.5:
            high_missing_files.append((fpath, n_frames, missing_ratio))

    # ── 4a. File bị hỏng ─────────────────────────────────────────────────
    if corrupted_files:
        print(f"\n  ❌ FILE BỊ LỖI ({len(corrupted_files)}):")
        for f, e in corrupted_files:
            print(f"     {os.path.basename(f)} — {e}")
    else:
        print(f"\n  ✅ Không có file nào bị lỗi đọc")

    # ── 4b. File toàn zero ────────────────────────────────────────────────
    if zero_frame_files:
        print(f"\n  ⚠️  FILE TOÀN SỐ 0 ({len(zero_frame_files)}):")
        for f, n in zero_frame_files:
            print(f"     {os.path.basename(f)} — {n} frames, tất cả = 0")
    else:
        print(f"  ✅ Không có file nào toàn số 0")

    # ── 4c. Missing keypoints > 50% ──────────────────────────────────────
    if high_missing_files:
        print(f"\n  ⚠️  MẤT KEYPOINTS > 50% ({len(high_missing_files)}):")
        for f, n, ratio in high_missing_files:
            print(f"     {os.path.basename(f)} — {n} frames, missing = {ratio*100:.1f}%")
    else:
        print(f"  ✅ Không có file nào mất keypoints > 50%")

    # ── 4d. Thống kê tổng quan missing ───────────────────────────────────
    avg_missing = np.mean(all_missing_ratios) if all_missing_ratios else 0
    max_missing = np.max(all_missing_ratios) if all_missing_ratios else 0
    n_has_missing = sum(1 for r in all_missing_ratios if r > 0)
    print(f"\n  📈 Missing keypoints trung bình: {avg_missing*100:.2f}%")
    print(f"  📈 Missing keypoints cao nhất : {max_missing*100:.2f}%")
    print(f"  📈 Số file có ít nhất 1 zero  : {n_has_missing}/{total_samples}")

    # ── 5. Thống kê độ dài ────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  THỐNG KÊ ĐỘ DÀI CHUỖI HÀNH ĐỘNG")
    print(f"{'─'*50}")

    if lengths:
        lengths_arr = np.array(lengths)
        print(f"  Trung bình : {lengths_arr.mean():.1f} frames")
        print(f"  Ngắn nhất  : {lengths_arr.min()} frames")
        print(f"  Dài nhất   : {lengths_arr.max()} frames")
        print(f"  Median     : {np.median(lengths_arr):.1f} frames")
        print(f"  Std        : {lengths_arr.std():.1f} frames")

        print(f"\n  Phân bố độ dài theo nhãn:")
        for lid in sorted(stats_by_label.keys()):
            lens = np.array(stats_by_label[lid])
            name = LABEL_MAP.get(lid, "Unknown")
            print(f"    [{lid}] {name:20s}: "
                  f"n={len(lens):3d}  "
                  f"min={lens.min():4d}  "
                  f"max={lens.max():4d}  "
                  f"mean={lens.mean():.1f}")

        # Histogram dạng text
        print(f"\n  Histogram độ dài:")
        bins = [0, 20, 31, 50, 100, 128, 150, 180, 200, 300, 500]
        hist, _ = np.histogram(lengths_arr, bins=bins)
        for i in range(len(hist)):
            bar = "█" * (hist[i])
            print(f"    {bins[i]:4d}–{bins[i+1]:4d} frames: {hist[i]:3d} {bar}")

    # ── 6. Tóm tắt ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TÓM TẮT KHÁM BỆNH DỮ LIỆU")
    print(f"{'='*70}")
    print(f"  ✅ Tổng mẫu          : {total_samples}")
    print(f"  ✅ File đọc OK       : {total_samples - len(corrupted_files)}")
    print(f"  ⚠️  File toàn zero    : {len(zero_frame_files)}")
    print(f"  ⚠️  Missing >50%      : {len(high_missing_files)}")
    print(f"  📊 Nhãn 0 (Fall)     : {label_counts.get(0, 0)}")
    print(f"  📊 Nhãn 1 (Walking)  : {label_counts.get(1, 0)}")
    print(f"  📏 Độ dài: {int(lengths_arr.min())}–{int(lengths_arr.max())} frames "
          f"(trung bình {lengths_arr.mean():.0f})")

    issues = len(zero_frame_files) + len(high_missing_files) + len(corrupted_files)
    if issues == 0:
        print(f"\n  🎉 DỮ LIỆU SẠCH! Sẵn sàng cho BƯỚC 2.")
    else:
        print(f"\n  ⚡ CÓ {issues} VẤN ĐỀ CẦN XỬ LÝ Ở BƯỚC 2.")
    print("=" * 70)


if __name__ == "__main__":
    audit()
