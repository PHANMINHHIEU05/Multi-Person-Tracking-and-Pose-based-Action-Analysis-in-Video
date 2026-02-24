"""
Module A – Human Detection on Video using YOLOv8 (Ultralytics)
================================================================
Detects persons in a video file, overlays bounding boxes + confidence,
and saves per-frame detections (CSV or JSONL) + a meta.json summary.

Usage examples:
    python src/module_a_detect.py --video data/video/input.mp4 \
        --out runs/detect/run1 --model yolov8n.pt --device 0

    python src/module_a_detect.py --video data/video/input.mp4 \
        --out runs/detect/run2 --resize 1280x720 --stride 2 --preview
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Module A – Person Detection with YOLOv8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",      type=str,   required=True,          help="Path to input video file")
    p.add_argument("--out",        type=str,   required=True,          help="Output directory for all results")
    p.add_argument("--model",      type=str,   default="yolov8n.pt",   help="YOLOv8 weights file or model name")
    p.add_argument("--device",     type=str,   default="auto",         help="Device: auto | cpu | 0 (GPU index)")
    p.add_argument("--imgsz",      type=int,   default=640,            help="Inference image size (square)")
    p.add_argument("--conf",       type=float, default=0.35,           help="Confidence threshold")
    p.add_argument("--iou",        type=float, default=0.5,            help="IoU threshold for NMS")
    p.add_argument("--stride",     type=int,   default=1,              help="Process every Nth frame (1 = all frames)")
    p.add_argument("--resize",     type=str,   default="",             help="Resize frames before inference, e.g. 1280x720. Empty = original size")
    p.add_argument("--preview",    action="store_true",                help="Show preview window while processing (requires display)")
    p.add_argument("--save_csv",   action="store_true", default=True,  help="Save per-frame detections as CSV")
    p.add_argument("--save_jsonl", action="store_true",                help="Save per-frame detections as JSONL (overrides CSV)")
    return p


# --------------------------------------------------------------------------- #
#  Device resolution
# --------------------------------------------------------------------------- #
def resolve_device(device_arg: str) -> str:
    """Return a device string suitable for Ultralytics / torch."""
    if device_arg == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device_arg


# --------------------------------------------------------------------------- #
#  Video I/O helpers
# --------------------------------------------------------------------------- #
def open_video(path: str) -> cv2.VideoCapture:
    """Open a video file and validate it is readable."""
    if not Path(path).exists():
        print(f"[ERROR] Video file not found: {path}", file=sys.stderr)
        sys.exit(1)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {path}", file=sys.stderr)
        sys.exit(1)
    return cap


def get_video_meta(cap: cv2.VideoCapture) -> Tuple[int, int, float, int]:
    """Return (width, height, fps, total_frame_count) for an open VideoCapture."""
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0   # fall back to 30 if unreadable
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fps, n


def parse_resize(resize_str: str) -> Optional[Tuple[int, int]]:
    """Parse '1280x720' → (1280, 720). Returns None if empty."""
    if not resize_str:
        return None
    parts = resize_str.lower().split("x")
    if len(parts) != 2:
        print(f"[WARN] Invalid --resize value '{resize_str}'. Ignoring.", file=sys.stderr)
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        print(f"[WARN] Could not parse --resize '{resize_str}'. Ignoring.", file=sys.stderr)
        return None


def make_video_writer(
    out_path: str,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    """Create a VideoWriter, trying H.264 (avc1) first, then mp4v."""
    fourcc_avc1 = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc_avc1, fps, (width, height))
    if writer.isOpened():
        print(f"[INFO] VideoWriter codec: H.264 (avc1)")
        return writer
    writer.release()
    # Fallback to mp4v
    fourcc_mp4v = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc_mp4v, fps, (width, height))
    if writer.isOpened():
        print(f"[INFO] VideoWriter codec: mp4v (H.264 not available)")
        return writer
    writer.release()
    print("[ERROR] Cannot open VideoWriter. Check OpenCV codec support.", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
#  Detection
# --------------------------------------------------------------------------- #
def detect_frame(
    model: YOLO,
    frame: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> List[dict]:
    """
    Run YOLOv8 inference on a single frame.

    Returns a list of detection dicts:
        {conf, x1, y1, x2, y2}
    Coordinates are in pixel space of the supplied frame.
    """
    results = model.predict(
        source=frame,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=[PERSON_CLASS_ID],
        device=device,
        verbose=False,
    )

    detections: List[dict] = []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy  = boxes.xyxy.cpu().numpy()   # (N, 4)
    confs = boxes.conf.cpu().numpy()   # (N,)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        detections.append(
            {
                "conf": float(confs[i]),
                "x1":   int(x1),
                "y1":   int(y1),
                "x2":   int(x2),
                "y2":   int(y2),
            }
        )
    return detections


# --------------------------------------------------------------------------- #
#  Drawing
# --------------------------------------------------------------------------- #
def draw_boxes(
    frame: np.ndarray,
    detections: List[dict],
    frame_idx: int,
    fps_estimate: float,
) -> np.ndarray:
    """
    Draw bounding boxes with confidence labels and overlay frame info.
    Thickness and font size are scaled relative to frame width.
    """
    h, w = frame.shape[:2]
    thickness  = max(1, w // 600)
    font_scale = max(0.4, w / 1280 * 0.7)
    font       = cv2.FONT_HERSHEY_SIMPLEX
    color_box  = (0, 255, 0)          # green boxes
    color_text = (255, 255, 255)      # white text
    color_bg   = (0, 180, 0)          # dark-green label bg

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf_val = det["conf"]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, thickness)

        # Label background + text
        label = f"{PERSON_CLASS_NAME} {conf_val:.2f}"
        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y1 - 4, lh + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - lh - baseline),
            (x1 + lw, label_y + baseline),
            color_bg,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label, (x1, label_y),
            font, font_scale, color_text, thickness, cv2.LINE_AA,
        )

    # Frame index + FPS overlay (top-left corner)
    info_text  = f"Frame: {frame_idx}  FPS: {fps_estimate:.1f}"
    (iw, ih), _ = cv2.getTextSize(info_text, font, font_scale * 0.9, 1)
    cv2.rectangle(frame, (4, 4), (iw + 10, ih + 12), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        frame, info_text, (8, ih + 6),
        font, font_scale * 0.9, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return frame


# --------------------------------------------------------------------------- #
#  Data writers
# --------------------------------------------------------------------------- #
class DetectionWriter:
    """Streams detection rows to CSV or JSONL without loading all into RAM."""

    def __init__(self, path: str, use_jsonl: bool = False):
        self.use_jsonl = use_jsonl
        self.path = path
        self._file = open(path, "w", newline="", encoding="utf-8")
        if not use_jsonl:
            self._csv_writer = csv.DictWriter(
                self._file,
                fieldnames=[
                    "frame_idx", "timestamp_sec",
                    "det_idx",
                    "class_id", "class_name", "conf",
                    "x1", "y1", "x2", "y2",
                ],
            )
            self._csv_writer.writeheader()

    def write(
        self,
        frame_idx: int,
        timestamp_sec: float,
        detections: List[dict],
    ):
        for det_idx, det in enumerate(detections):
            row = {
                "frame_idx":     frame_idx,
                "timestamp_sec": round(timestamp_sec, 4),
                "det_idx":       det_idx,
                "class_id":      PERSON_CLASS_ID,
                "class_name":    PERSON_CLASS_NAME,
                "conf":          round(det["conf"], 4),
                "x1":            det["x1"],
                "y1":            det["y1"],
                "x2":            det["x2"],
                "y2":            det["y2"],
            }
            if self.use_jsonl:
                self._file.write(json.dumps(row) + "\n")
            else:
                self._csv_writer.writerow(row)

    def close(self):
        self._file.close()


# --------------------------------------------------------------------------- #
#  Meta persistence
# --------------------------------------------------------------------------- #
def save_meta(
    path: str,
    args: argparse.Namespace,
    device: str,
    input_fps: float,
    output_fps: float,
    frame_size: Tuple[int, int],      # (width, height) after optional resize
    processed_frames: int,
    total_frames: int,
    avg_fps_proc: float,
    avg_dets: float,
    resize_applied: Optional[Tuple[int, int]],
):
    meta = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "model":            args.model,
        "device":           device,
        "imgsz":            args.imgsz,
        "conf":             args.conf,
        "iou":              args.iou,
        "stride":           args.stride,
        "resize":           f"{resize_applied[0]}x{resize_applied[1]}" if resize_applied else "original",
        "input_fps":        round(input_fps, 4),
        "output_fps":       round(output_fps, 4),
        "frame_size":       {"width": frame_size[0], "height": frame_size[1]},
        "total_frames":     total_frames,
        "processed_frames": processed_frames,
        "avg_processing_fps": round(avg_fps_proc, 2),
        "avg_dets_per_frame": round(avg_dets, 3),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# --------------------------------------------------------------------------- #
#  Main pipeline
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace):
    # ── Device & Model ────────────────────────────────────────────────────── #
    device = resolve_device(args.device)
    print(f"[INFO] Device  : {device} ({'GPU – ' + torch.cuda.get_device_name(int(device)) if device != 'cpu' and torch.cuda.is_available() else 'CPU'})")
    print(f"[INFO] Model   : {args.model}")

    model = YOLO(args.model)

    # ── Input video ───────────────────────────────────────────────────────── #
    cap = open_video(args.video)
    orig_w, orig_h, input_fps, total_frames = get_video_meta(cap)
    print(f"[INFO] Video   : {args.video}  [{orig_w}x{orig_h} @ {input_fps:.2f} FPS, ~{total_frames} frames]")

    resize_wh = parse_resize(args.resize)
    frame_w   = resize_wh[0] if resize_wh else orig_w
    frame_h   = resize_wh[1] if resize_wh else orig_h

    output_fps = input_fps / max(1, args.stride)

    # ── Output directory ──────────────────────────────────────────────────── #
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_out_path = str(out_dir / "video_detect.mp4")
    ext = "jsonl" if args.save_jsonl else "csv"
    data_out_path  = str(out_dir / f"detections.{ext}")
    meta_out_path  = str(out_dir / "meta.json")

    # ── Writers ───────────────────────────────────────────────────────────── #
    video_writer   = make_video_writer(video_out_path, frame_w, frame_h, output_fps)
    det_writer     = DetectionWriter(data_out_path, use_jsonl=args.save_jsonl)

    # ── Processing loop ───────────────────────────────────────────────────── #
    processed_frames = 0
    total_dets       = 0
    loop_start       = time.perf_counter()
    fps_window       = []           # rolling FPS estimate (last N frames)
    FPS_WINDOW_SIZE  = 30

    pbar = tqdm(total=total_frames if total_frames > 0 else None,
                unit="fr", desc="Detecting", dynamic_ncols=True)

    frame_idx = -1
    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break                           # end of video or read error

            frame_idx += 1
            pbar.update(1)

            # ── Stride: skip frames not in schedule ─────────────────────── #
            if frame_idx % args.stride != 0:
                continue

            # ── Resize if requested ─────────────────────────────────────── #
            frame = (
                cv2.resize(raw_frame, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                if resize_wh else raw_frame
            )

            # ── Inference ───────────────────────────────────────────────── #
            t0          = time.perf_counter()
            detections  = detect_frame(model, frame, args.imgsz, args.conf, args.iou, device)
            elapsed     = time.perf_counter() - t0

            fps_window.append(1.0 / max(elapsed, 1e-6))
            if len(fps_window) > FPS_WINDOW_SIZE:
                fps_window.pop(0)
            fps_estimate = sum(fps_window) / len(fps_window)

            timestamp_sec = frame_idx / input_fps

            # ── Save detections ─────────────────────────────────────────── #
            det_writer.write(frame_idx, timestamp_sec, detections)
            total_dets     += len(detections)
            processed_frames += 1

            # ── Draw + write output frame ───────────────────────────────── #
            annotated = draw_boxes(frame.copy(), detections, frame_idx, fps_estimate)
            video_writer.write(annotated)

            # ── Optional preview ────────────────────────────────────────── #
            if args.preview:
                cv2.imshow("Module A – Detection Preview", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Preview closed by user.")
                    break

            # ── Progress bar postfix ─────────────────────────────────────── #
            pbar.set_postfix(fps=f"{fps_estimate:.1f}", dets=len(detections))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        pbar.close()
        cap.release()
        video_writer.release()
        det_writer.close()
        if args.preview:
            cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────── #
    total_elapsed  = time.perf_counter() - loop_start
    avg_fps_proc   = processed_frames / max(total_elapsed, 1e-6)
    avg_dets       = total_dets / max(processed_frames, 1)

    print("\n" + "=" * 60)
    print("  DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Total frames in video : {frame_idx + 1}")
    print(f"  Processed frames      : {processed_frames}")
    print(f"  Avg processing FPS    : {avg_fps_proc:.2f}")
    print(f"  Avg detections/frame  : {avg_dets:.2f}")
    print(f"  Output video          : {video_out_path}")
    print(f"  Detections file       : {data_out_path}")
    print(f"  Meta file             : {meta_out_path}")
    print("=" * 60 + "\n")

    # ── Save meta ─────────────────────────────────────────────────────────── #
    save_meta(
        path=meta_out_path,
        args=args,
        device=device,
        input_fps=input_fps,
        output_fps=output_fps,
        frame_size=(frame_w, frame_h),
        processed_frames=processed_frames,
        total_frames=frame_idx + 1,
        avg_fps_proc=avg_fps_proc,
        avg_dets=avg_dets,
        resize_applied=resize_wh,
    )


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    run(args)
