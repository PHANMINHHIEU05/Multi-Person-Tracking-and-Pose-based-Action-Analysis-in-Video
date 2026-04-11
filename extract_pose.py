"""
extract_pose.py – Extract Human Pose Keypoints (YOLOv8-Pose)
=============================================================
Trích xuất 17 COCO keypoints cho mỗi frame, chuẩn hoá toạ độ về [0, 1],
và lưu mỗi chuỗi hành động thành file .npy với shape (num_frames, 17, 2).

Hỗ trợ 2 nguồn dữ liệu:
  • UR_Fall  – chuỗi ảnh PNG, label dựa theo prefix (fall / adl)
  • Multicam – video .avi, label dựa theo data_tuple3.csv, chỉ lấy cam1

Usage:
    python extract_pose.py                       # mặc định cả 2 nguồn
    python extract_pose.py --source ur_fall      # chỉ UR_Fall
    python extract_pose.py --source multicam     # chỉ Multicam
    python extract_pose.py --device cpu          # chạy trên CPU
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH      = "yolov8n-pose.pt"
DEVICE          = "cuda"       # overridable via --device
CONF_THRESHOLD  = 0.5
OUTPUT_DIR      = "./data/processed_pose"
METADATA_FILE   = "metadata_pose_final.csv"

UR_FALL_PATH    = "./data/UR_Fall"
MULTICAM_PATH   = "./data/Multicam"

LABEL_MAP = {
    0: "Fall",
    1: "Walking",
    2: "Stairs",
    3: "Sitting_Standing",
    4: "Bending",
    5: "Jogging_Jumping",
}

# Multicam CSV:  label 1 → Fall (project 0)  |  label 0 → non-Fall (project 1)
MULTICAM_LABEL_REMAP = {1: 0, 0: 1}


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract 17 COCO pose keypoints from UR_Fall & Multicam datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",   type=str, default=MODEL_PATH,     help="YOLOv8-Pose weights path")
    p.add_argument("--device",  type=str, default=DEVICE,         help="cuda | cpu | 0")
    p.add_argument("--conf",    type=float, default=CONF_THRESHOLD, help="Detection confidence threshold")
    p.add_argument("--outdir",  type=str, default=OUTPUT_DIR,     help="Output directory for .npy + metadata")
    p.add_argument("--source",  type=str, default="all",
                   choices=["all", "ur_fall", "multicam"],        help="Which dataset(s) to process")
    p.add_argument(
        "--multicam_cams",
        type=str,
        default="1",
        help="Comma-separated cam ids for Multicam (e.g. 1 or 2,3,4 or all)",
    )
    p.add_argument(
        "--multicam_frame_step",
        type=int,
        default=1,
        help="Frame step when extracting each annotated Multicam segment",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str, device: str) -> YOLO:
    """Load YOLOv8 pose model onto the requested device."""
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("[INFO] CUDA not available, falling back to CPU.")
        device = "cpu"
    model = YOLO(model_path)
    model.to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Core helpers
# ─────────────────────────────────────────────────────────────────────────────
def select_best_person(result) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Từ kết quả YOLOv8-Pose, chọn 1 người duy nhất (confidence cao nhất).
    Trả về (xy, kpt_conf) với xy shape (17, 2) hoặc None nếu không phát hiện.
    """
    if result.keypoints is None:
        return None

    kpts = result.keypoints
    # Kiểm tra có dữ liệu xy hay không
    if kpts.xy is None or kpts.xy.shape[0] == 0:
        return None

    # Chọn person có box confidence cao nhất
    if result.boxes is not None and len(result.boxes) > 0:
        confs = result.boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
    else:
        best_idx = 0

    xy = kpts.xy[best_idx].cpu().numpy().astype(np.float32)  # (17, 2)

    if kpts.conf is not None and kpts.conf.shape[0] > best_idx:
        kpt_conf = kpts.conf[best_idx].cpu().numpy()
    else:
        kpt_conf = np.ones(17, dtype=np.float32)

    return xy, kpt_conf


def normalize_keypoints(xy: np.ndarray, w: int, h: int) -> np.ndarray:
    """Chuẩn hoá (17, 2) keypoints về [0, 1] theo kích thước ảnh."""
    norm = xy.copy()
    if w > 0:
        norm[:, 0] /= w
    if h > 0:
        norm[:, 1] /= h
    return np.clip(norm, 0.0, 1.0)


def _process_frame(model, frame: np.ndarray, conf: float,
                   prev_kpts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chạy inference trên 1 frame, trả về (kpts_normed, updated_prev).
    Nếu không detect được → dùng prev_kpts (interpolation forward-fill).
    """
    h, w = frame.shape[:2]
    results = model(frame, conf=conf, verbose=False)
    person = select_best_person(results[0])

    if person is not None:
        xy, _ = person
        if np.any(xy > 0):
            kpts = normalize_keypoints(xy, w, h)
            return kpts, kpts.copy()

    # Không detect → dùng frame trước
    return prev_kpts.copy(), prev_kpts


# ─────────────────────────────────────────────────────────────────────────────
#  Image-sequence extraction  (UR_Fall)
# ─────────────────────────────────────────────────────────────────────────────
def extract_from_images(model, image_paths: List[str],
                        conf: float) -> np.ndarray:
    """
    Trích xuất keypoints từ danh sách ảnh.
    Returns shape (num_frames, 17, 2).
    """
    all_kpts: List[np.ndarray] = []
    prev = np.zeros((17, 2), dtype=np.float32)

    for img_path in tqdm(image_paths, desc="    frames", leave=False, unit="f"):
        img = cv2.imread(img_path)
        if img is None:
            all_kpts.append(prev.copy())
            continue
        kpts, prev = _process_frame(model, img, conf, prev)
        all_kpts.append(kpts)

    return np.array(all_kpts, dtype=np.float32) if all_kpts else np.zeros((0, 17, 2), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Video extraction  (Multicam)
# ─────────────────────────────────────────────────────────────────────────────
def extract_from_video(model, video_path: str, conf: float,
                       start_frame: Optional[int] = None,
                       end_frame: Optional[int] = None,
                       frame_step: int = 1) -> np.ndarray:
    """
    Trích xuất keypoints từ video (tuỳ chọn khoảng frame).
    Returns shape (num_frames, 17, 2).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [WARN] Không mở được video: {video_path}")
        return np.zeros((0, 17, 2), dtype=np.float32)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame is not None and start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        start_frame = 0

    if end_frame is not None:
        n_frames_raw = max(1, end_frame - start_frame + 1)
    else:
        n_frames_raw = max(0, total - start_frame)

    all_kpts: List[np.ndarray] = []
    prev = np.zeros((17, 2), dtype=np.float32)
    step = max(1, int(frame_step))

    for local_idx in tqdm(range(n_frames_raw), desc="    frames", leave=False, unit="f"):
        ret, frame = cap.read()
        if not ret:
            break
        if local_idx % step != 0:
            continue
        kpts, prev = _process_frame(model, frame, conf, prev)
        all_kpts.append(kpts)

    cap.release()
    return np.array(all_kpts, dtype=np.float32) if all_kpts else np.zeros((0, 17, 2), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset: UR_Fall
# ─────────────────────────────────────────────────────────────────────────────
def process_ur_fall(model, conf: float, output_dir: str) -> List[dict]:
    """
    Xử lý UR_Fall (image sequences).
    Label: fall-* → 0 (Fall)  |  adl-* → 1 (Walking / ADL chung)
    """
    metadata: List[dict] = []
    ur_path = Path(UR_FALL_PATH)
    if not ur_path.exists():
        print(f"[WARN] Không tìm thấy UR_Fall tại: {ur_path}")
        return metadata

    folders = sorted([f for f in ur_path.iterdir() if f.is_dir()])
    print(f"\n{'='*65}")
    print(f"  UR_FALL DATASET — {len(folders)} sequences")
    print(f"{'='*65}")

    for folder in tqdm(folders, desc="UR_Fall", unit="seq"):
        name = folder.name  # e.g. fall-01-cam0-rgb

        # Xác định label
        if name.startswith("fall"):
            label_id = 0
        elif name.startswith("adl"):
            label_id = 1
        else:
            continue
        label_name = LABEL_MAP[label_id]

        # Thu thập ảnh đã sắp xếp
        imgs = sorted(glob.glob(str(folder / "*.png")))
        if not imgs:
            imgs = sorted(glob.glob(str(folder / "*.jpg")))
        if not imgs:
            tqdm.write(f"  [SKIP] Không có ảnh trong {name}")
            continue

        tqdm.write(f"  ► {name}  |  {len(imgs)} frames  |  {label_name}")

        kpts = extract_from_images(model, imgs, conf)
        if kpts.shape[0] == 0:
            tqdm.write(f"  [SKIP] Không trích xuất được keypoints: {name}")
            continue

        npy_name = f"ur_fall_{name}.npy"
        npy_path = os.path.join(output_dir, npy_name)
        np.save(npy_path, kpts)

        metadata.append(dict(
            source="UR_Fall",
            action_id=name,
            label_name=label_name,
            label_id=label_id,
            npy_path=npy_path,
        ))
        tqdm.write(f"    ✓ {npy_path}  shape={kpts.shape}")

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset: Multicam
# ─────────────────────────────────────────────────────────────────────────────
def _find_multicam_video_root() -> Optional[Path]:
    """Tìm đường dẫn gốc chứa các thư mục chute* trong Multicam."""
    base = Path(MULTICAM_PATH)
    # Cấu trúc thực tế: data/Multicam/dataset/dataset/chute01/
    candidates = [
        base / "dataset" / "dataset",
        base / "dataset",
        base,
    ]
    for c in candidates:
        if c.exists() and any(c.glob("chute*")):
            return c
    return None


def _parse_multicam_cams(cams_arg: str, all_cam_values: np.ndarray) -> list[int]:
    allowed = sorted(int(v) for v in all_cam_values if 1 <= int(v) <= 8)
    if not allowed:
        return []
    raw = str(cams_arg).strip().lower()
    if raw == "all":
        return allowed

    selected: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            cam = int(token)
        except ValueError:
            continue
        if cam in allowed:
            selected.append(cam)
    return sorted(set(selected))


def process_multicam(model, conf: float, output_dir: str, cams_arg: str, frame_step: int) -> List[dict]:
    """
    Xử lý Multicam Fall Dataset.
    Dùng data_tuple3.csv để xác định frame range + label cho từng segment.
    Chỉ lấy cam1 để tránh duplicate viewpoints.
    """
    metadata: List[dict] = []

    csv_path = Path(MULTICAM_PATH) / "data_tuple3.csv"
    if not csv_path.exists():
        print(f"[WARN] Không tìm thấy CSV: {csv_path}")
        return metadata

    video_root = _find_multicam_video_root()
    if video_root is None:
        print("[WARN] Không tìm thấy thư mục chute* trong Multicam")
        return metadata

    # Đọc CSV annotation
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    cams = _parse_multicam_cams(cams_arg, df["cam"].dropna().unique())
    if not cams:
        print(f"[WARN] Không có cam hợp lệ từ --multicam_cams={cams_arg}")
        return metadata

    # Lọc theo cam được chọn.
    df_cam1 = df[df["cam"].isin([float(v) for v in cams])].copy()
    for col in ("chute", "start", "end", "label"):
        df_cam1[col] = df_cam1[col].astype(int)
    df_cam1["cam"] = df_cam1["cam"].astype(int)

    print(f"\n{'='*65}")
    print(
        f"  MULTICAM DATASET — {len(df_cam1)} annotated segments "
        f"(cams={','.join(str(v) for v in cams)}, frame_step={max(1, int(frame_step))})"
    )
    print(f"{'='*65}")

    for _, row in tqdm(df_cam1.iterrows(), total=len(df_cam1),
                       desc="Multicam", unit="seg"):
        chute   = int(row["chute"])
        cam     = int(row["cam"])
        start   = int(row["start"])
        end     = int(row["end"])
        mc_lbl  = int(row["label"])

        label_id   = MULTICAM_LABEL_REMAP.get(mc_lbl, 1)
        label_name = LABEL_MAP[label_id]

        # Đường dẫn video
        video_file = video_root / f"chute{chute:02d}" / f"cam{cam}.avi"
        if not video_file.exists():
            tqdm.write(f"  [SKIP] Video không tồn tại: {video_file}")
            continue

        action_id = f"chute{chute:02d}_cam{cam}_f{start}-{end}"
        tqdm.write(f"  ► {action_id}  |  {end - start + 1} frames  |  {label_name}")

        kpts = extract_from_video(model, str(video_file), conf,
                                  start_frame=start, end_frame=end, frame_step=frame_step)
        if kpts.shape[0] == 0:
            tqdm.write(f"  [SKIP] Không trích xuất được keypoints: {action_id}")
            continue

        npy_name = f"multicam_{action_id}.npy"
        npy_path = os.path.join(output_dir, npy_name)
        np.save(npy_path, kpts)

        metadata.append(dict(
            source="Multicam",
            action_id=action_id,
            label_name=label_name,
            label_id=label_id,
            npy_path=npy_path,
            cam=cam,
            start_frame=start,
            end_frame=end,
            label_raw=mc_lbl,
            video_path=str(video_file),
            frame_step=max(1, int(frame_step)),
        ))
        tqdm.write(f"    ✓ {npy_path}  shape={kpts.shape}")

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()

    print("=" * 65)
    print("  POSE KEYPOINT EXTRACTION PIPELINE")
    print(f"  Model : {args.model}")
    print(f"  Device: {args.device}  |  Conf: {args.conf}")
    print(f"  Output: {args.outdir}")
    print("=" * 65)

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[1/3] Loading YOLOv8-Pose model ...")
    model = load_model(args.model, args.device)
    print(f"  ✓ Model loaded in {time.time() - t0:.1f}s\n")

    # ── Process datasets ──────────────────────────────────────────────────
    all_meta: List[dict] = []

    print("[2/3] Extracting keypoints ...")

    if args.source in ("all", "ur_fall"):
        meta_ur = process_ur_fall(model, args.conf, args.outdir)
        all_meta.extend(meta_ur)
        print(f"\n  → UR_Fall done: {len(meta_ur)} sequences saved")

    if args.source in ("all", "multicam"):
        meta_mc = process_multicam(
            model,
            args.conf,
            args.outdir,
            cams_arg=args.multicam_cams,
            frame_step=args.multicam_frame_step,
        )
        all_meta.extend(meta_mc)
        print(f"\n  → Multicam done: {len(meta_mc)} segments saved")

    # ── Save metadata CSV ─────────────────────────────────────────────────
    print("\n[3/3] Saving metadata CSV ...")
    if all_meta:
        df_out = pd.DataFrame(all_meta,
                              columns=[
                                  "source",
                                  "action_id",
                                  "label_name",
                                  "label_id",
                                  "npy_path",
                                  "cam",
                                  "start_frame",
                                  "end_frame",
                                  "label_raw",
                                  "video_path",
                                  "frame_step",
                              ])
        meta_path = os.path.join(args.outdir, METADATA_FILE)
        df_out.to_csv(meta_path, index=False)

        print(f"  ✓ {meta_path}  ({len(df_out)} rows)")
        print(f"\n  Label distribution:")
        for lbl, cnt in df_out["label_name"].value_counts().items():
            print(f"    {lbl:20s} : {cnt}")
    else:
        print("  [WARN] Không có dữ liệu nào được xử lý!")

    print(f"\n{'='*65}")
    print("  PIPELINE HOÀN TẤT!")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
