"""
Module B – Multi-Person Tracking on Video using YOLOv8 Track
=============================================================
Assigns stable track IDs to persons across video frames using
Ultralytics YOLO built-in tracker (ByteTrack or BoT-SORT).

Outputs
-------
  runs/track/<run_name>/
    video_track.mp4      – annotated video (bbox + ID + conf)
    tracks.csv           – per-frame tracklet rows
    tracks.jsonl         – per-frame tracklet rows (optional, alternative)
    meta.json            – run settings (model, tracker, thresholds, video info)
    summary.json         – diagnostics (unique IDs, lost tracks, avg active, …)

Usage examples
--------------
    python src/module_b_track.py --video data/video/input.mp4 \\
        --out runs/track/run1 --model yolov8n.pt --device 0 --tracker botsort

    python src/module_b_track.py --video data/video/input.mp4 \\
        --out runs/track/run2 --tracker bytetrack --stride 2 --resize 1280x720
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
PERSON_CLASS_ID   = 0
PERSON_CLASS_NAME = "person"

# Tracker shorthand → (Ultralytics built-in name, local config fallback)
_TRACKER_ALIASES: Dict[str, str] = {
    "bytetrack": "bytetrack.yaml",
    "botsort":   "botsort.yaml",
}

# Gap (frames) before a reappearing track is counted as a "lost track event"
LOST_GAP_THRESHOLD = 10

# IoU overlap threshold used in the id-switch proxy heuristic
ID_SWITCH_IOU_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Module B – Multi-Person Tracking with YOLOv8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",      type=str,   required=True,
                   help="Path to input video file")
    p.add_argument("--out",        type=str,   required=True,
                   help="Output directory for all results")
    p.add_argument("--model",      type=str,   default="yolov8n.pt",
                   help="YOLOv8 weights file or model name")
    p.add_argument("--device",     type=str,   default="auto",
                   help="Device: auto | cpu | 0 (GPU index)")
    p.add_argument("--tracker",    type=str,   default="botsort",
                   help="Tracker: botsort | bytetrack | path/to/tracker.yaml")
    p.add_argument("--imgsz",      type=int,   default=640,
                   help="Inference image size (square pixels)")
    p.add_argument("--conf",       type=float, default=0.35,
                   help="Confidence threshold")
    p.add_argument("--iou",        type=float, default=0.5,
                   help="IoU threshold for NMS")
    p.add_argument("--stride",     type=int,   default=1,
                   help="Process every Nth frame (1 = all frames)")
    p.add_argument("--resize",     type=str,   default="",
                   help="Resize frames before inference, e.g. 1280x720")
    p.add_argument("--max_det",    type=int,   default=100,
                   help="Maximum detections per frame")
    p.add_argument("--preview",    action="store_true",
                   help="Show preview window while processing (requires display)")
    p.add_argument("--save_csv",   action="store_true", default=True,
                   help="Save per-frame tracks as CSV (default)")
    p.add_argument("--save_jsonl", action="store_true",
                   help="Save per-frame tracks as JSONL (overrides CSV)")
    return p


# --------------------------------------------------------------------------- #
#  Device resolution
# --------------------------------------------------------------------------- #
def resolve_device(device_arg: str) -> str:
    """Return device string suitable for Ultralytics / torch."""
    if device_arg == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device_arg


# --------------------------------------------------------------------------- #
#  Tracker resolution
# --------------------------------------------------------------------------- #
def resolve_tracker(tracker_arg: str) -> str:
    """
    Resolve --tracker value to a yaml path string understood by Ultralytics.

    Priority:
      1. Explicit file path that exists → use as-is
      2. Known alias (botsort / bytetrack) → check local config/ first,
         then fall back to Ultralytics built-in (bare yaml name)
    """
    # Explicit path given
    if os.path.isfile(tracker_arg):
        print(f"[INFO] Tracker cfg : {tracker_arg} (explicit file)")
        return tracker_arg

    alias_lower = tracker_arg.lower()
    yaml_name   = _TRACKER_ALIASES.get(alias_lower)
    if yaml_name is None:
        print(f"[WARN] Unknown tracker '{tracker_arg}'. Passing to Ultralytics as-is.")
        return tracker_arg

    # Check local config directory relative to CWD or script location
    for base in [Path.cwd(), Path(__file__).parent.parent]:
        local_cfg = base / "config" / yaml_name
        if local_cfg.exists():
            print(f"[INFO] Tracker cfg : {local_cfg} (local config)")
            return str(local_cfg)

    # Fall back to Ultralytics built-in
    print(f"[INFO] Tracker cfg : {yaml_name} (Ultralytics built-in)")
    return yaml_name


# --------------------------------------------------------------------------- #
#  Video I/O helpers
# --------------------------------------------------------------------------- #
def open_video(path: str) -> cv2.VideoCapture:
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fps, n


def parse_resize(resize_str: str) -> Optional[Tuple[int, int]]:
    """Parse '1280x720' → (1280, 720). Returns None if empty."""
    if not resize_str:
        return None
    parts = resize_str.lower().split("x")
    if len(parts) != 2:
        print(f"[WARN] Invalid --resize '{resize_str}'. Ignoring.", file=sys.stderr)
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        print(f"[WARN] Cannot parse --resize '{resize_str}'. Ignoring.", file=sys.stderr)
        return None


def make_video_writer(out_path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """Create a VideoWriter; try H.264 first then fall back to mp4v."""
    for fourcc_str, label in [("avc1", "H.264 (avc1)"), ("mp4v", "mp4v")]:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"[INFO] VideoWriter codec : {label}")
            return writer
        writer.release()
    print("[ERROR] Cannot open VideoWriter. Check OpenCV codec support.", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
#  Drawing utilities
# --------------------------------------------------------------------------- #
def _track_color(track_id: int) -> Tuple[int, int, int]:
    """Deterministic BGR colour derived from track_id (hash-based)."""
    rng = np.random.default_rng(int(track_id) * 2654435761 & 0xFFFFFFFF)
    h   = int(rng.integers(0, 180))          # hue
    hsv = np.array([[[h, 220, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(
    frame: np.ndarray,
    tracks: List[dict],
    frame_idx: int,
    fps_estimate: float,
    active_count: int,
) -> np.ndarray:
    """Draw bounding boxes, track IDs, conf scores, and HUD overlay."""
    h, w    = frame.shape[:2]
    thick   = max(1, w // 600)
    fs      = max(0.4, w / 1280 * 0.65)
    font    = cv2.FONT_HERSHEY_SIMPLEX

    for trk in tracks:
        tid  = int(trk["track_id"])
        x1, y1, x2, y2 = trk["x1"], trk["y1"], trk["x2"], trk["y2"]
        conf = trk["conf"]
        color = _track_color(tid)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # Label: ID + conf
        label = f"ID:{tid}  {conf:.2f}"
        (lw, lh), bl = cv2.getTextSize(label, font, fs, thick)
        ly = max(y1 - 4, lh + 4)
        cv2.rectangle(frame, (x1, ly - lh - bl), (x1 + lw, ly + bl), color, cv2.FILLED)
        cv2.putText(frame, label, (x1, ly), font, fs, (255, 255, 255), thick, cv2.LINE_AA)

    # HUD: frame / FPS / active tracks (top-left)
    hud = f"Frame:{frame_idx}  FPS:{fps_estimate:.1f}  Active:{active_count}"
    (hw, hh), _ = cv2.getTextSize(hud, font, fs * 0.85, 1)
    cv2.rectangle(frame, (4, 4), (hw + 12, hh + 14), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, hud, (8, hh + 8), font, fs * 0.85, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


# --------------------------------------------------------------------------- #
#  Track extraction from Ultralytics results
# --------------------------------------------------------------------------- #
def extract_tracks(results) -> List[dict]:
    """
    Parse Ultralytics track result for frame.

    Returns a list of dicts:
        track_id, class_id, class_name, conf,
        x1, y1, x2, y2, cx, cy, w, h

    If boxes.id is None (tracker not yet initialised or no active tracks)
    the function returns an empty list – this is intentional; partial
    frames without IDs are not written to the track file.
    """
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    ids   = boxes.id                  # may be None before tracker is warmed up
    if ids is None:
        return []

    xyxy  = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss  = boxes.cls.cpu().numpy().astype(int)
    ids   = ids.cpu().numpy().astype(int)

    out = []
    for i in range(len(ids)):
        if clss[i] != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = int(xyxy[i, 0]), int(xyxy[i, 1]), int(xyxy[i, 2]), int(xyxy[i, 3])
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw // 2
        cy = y1 + bh // 2
        out.append(
            {
                "track_id":   int(ids[i]),
                "class_id":   PERSON_CLASS_ID,
                "class_name": PERSON_CLASS_NAME,
                "conf":       float(confs[i]),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx,  "cy": cy,
                "w":  bw,  "h":  bh,
            }
        )
    return out


# --------------------------------------------------------------------------- #
#  Track data writer
# --------------------------------------------------------------------------- #
_TRACK_FIELDS = [
    "frame_idx", "timestamp_sec",
    "track_id", "class_id", "class_name", "conf",
    "x1", "y1", "x2", "y2", "cx", "cy", "w", "h",
    "is_new_track", "age_frames", "source",
]


class TrackWriter:
    """Streams track rows to CSV or JSONL without loading all data into RAM."""

    def __init__(self, path: str, use_jsonl: bool = False):
        self.use_jsonl = use_jsonl
        self.path      = path
        self._file     = open(path, "w", newline="", encoding="utf-8")
        if not use_jsonl:
            self._csv = csv.DictWriter(self._file, fieldnames=_TRACK_FIELDS)
            self._csv.writeheader()

    def write_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        tracks: List[dict],
        known_ids: set,
        track_ages: Dict[int, int],
        source_label: str,
    ):
        for trk in tracks:
            tid   = trk["track_id"]
            is_new = tid not in known_ids
            age    = track_ages.get(tid, 0)
            row = {
                "frame_idx":     frame_idx,
                "timestamp_sec": round(timestamp_sec, 4),
                "track_id":      tid,
                "class_id":      trk["class_id"],
                "class_name":    trk["class_name"],
                "conf":          round(trk["conf"], 4),
                "x1":  trk["x1"], "y1": trk["y1"],
                "x2":  trk["x2"], "y2": trk["y2"],
                "cx":  trk["cx"], "cy": trk["cy"],
                "w":   trk["w"],  "h":  trk["h"],
                "is_new_track":  int(is_new),
                "age_frames":    age,
                "source":        source_label,
            }
            if self.use_jsonl:
                self._file.write(json.dumps(row) + "\n")
            else:
                self._csv.writerow(row)

    def close(self):
        self._file.close()


# --------------------------------------------------------------------------- #
#  Diagnostics helpers
# --------------------------------------------------------------------------- #
def _bbox_iou(a: dict, b: dict) -> float:
    """Compute IoU between two bbox dicts with keys x1,y1,x2,y2."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw  = max(0, ix2 - ix1)
    ih  = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter + 1e-9)


class TrackDiagnostics:
    """
    Lightweight in-memory diagnostics accumulated over the tracking run.

    Tracked quantities
    ------------------
    * track_id → list of frame indices where track was seen (for dwell time)
    * lost_track_events: proxy count – track reappears after >LOST_GAP_THRESHOLD
      frames of absence
    * id_switch_proxy: heuristic count – two different IDs overlap heavily in
      adjacent frames (HIGH_IOU_THRESHOLD).  Documented as approximate.
    * active_counts: list of active track counts per processed frame
    """

    def __init__(self):
        self.track_frames:       Dict[int, List[int]]   = collections.defaultdict(list)
        self.last_seen:          Dict[int, int]         = {}   # id → last frame_idx
        self.known_ids:          set                    = set()
        self.track_ages:         Dict[int, int]         = {}   # id → age (frames seen)
        self.active_counts:      List[int]              = []
        self.lost_track_events:  int                    = 0
        self.id_switch_proxy:    int                    = 0
        self._prev_tracks:       List[dict]             = []   # tracks from last frame

    def update(self, frame_idx: int, tracks: List[dict]):
        current_ids = {t["track_id"] for t in tracks}

        # Age & lost-track events
        for trk in tracks:
            tid = trk["track_id"]
            if tid in self.last_seen:
                gap = frame_idx - self.last_seen[tid]
                if gap > LOST_GAP_THRESHOLD and tid in self.known_ids:
                    self.lost_track_events += 1
            self.last_seen[tid]  = frame_idx
            self.known_ids.add(tid)
            self.track_frames[tid].append(frame_idx)
            self.track_ages[tid] = self.track_ages.get(tid, 0) + 1

        # ID-switch proxy heuristic (adjacent frame, high IoU, different ID)
        if self._prev_tracks:
            for prev in self._prev_tracks:
                for curr in tracks:
                    if prev["track_id"] != curr["track_id"]:
                        iou = _bbox_iou(prev, curr)
                        if iou >= ID_SWITCH_IOU_THRESHOLD:
                            self.id_switch_proxy += 1

        self.active_counts.append(len(current_ids))
        self._prev_tracks = tracks

    def compute_summary(
        self,
        total_frames: int,
        processed_frames: int,
        avg_proc_fps: float,
        tracker_label: str,
        model_label: str,
    ) -> dict:
        dwell_lengths = [len(v) for v in self.track_frames.values()]

        # Track-length histogram (bins: 1-5, 6-15, 16-30, 31-60, 61-120, 120+)
        bins   = [(1, 5), (6, 15), (16, 30), (31, 60), (61, 120), (121, 10**9)]
        labels = ["1-5", "6-15", "16-30", "31-60", "61-120", "121+"]
        hist   = {lbl: 0 for lbl in labels}
        for dlen in dwell_lengths:
            for (lo, hi), lbl in zip(bins, labels):
                if lo <= dlen <= hi:
                    hist[lbl] += 1
                    break

        avg_active = (
            sum(self.active_counts) / len(self.active_counts)
            if self.active_counts else 0.0
        )

        return {
            "timestamp":              datetime.now(timezone.utc).isoformat(),
            "model":                  model_label,
            "tracker":                tracker_label,
            "total_frames":           total_frames,
            "processed_frames":       processed_frames,
            "avg_processing_fps":     round(avg_proc_fps, 2),
            "unique_track_ids":       len(self.known_ids),
            "avg_active_tracks_per_frame": round(avg_active, 2),
            "track_length_histogram": hist,
            "lost_track_events":      self.lost_track_events,
            "id_switch_proxy": {
                "count":  self.id_switch_proxy,
                "note":   (
                    "Heuristic proxy only – counts adjacent-frame pairs where "
                    f"two different track IDs have IoU >= {ID_SWITCH_IOU_THRESHOLD}. "
                    "Not a true MOTA ID-switch metric."
                ),
            },
        }


# --------------------------------------------------------------------------- #
#  Meta persistence
# --------------------------------------------------------------------------- #
def save_meta(
    path: str,
    args: argparse.Namespace,
    device: str,
    tracker_resolved: str,
    input_fps: float,
    output_fps: float,
    orig_wh: Tuple[int, int],
    frame_wh: Tuple[int, int],
    total_frames: int,
    processed_frames: int,
):
    meta = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "model":            args.model,
        "tracker":          args.tracker,
        "tracker_cfg":      tracker_resolved,
        "device":           device,
        "imgsz":            args.imgsz,
        "conf":             args.conf,
        "iou":              args.iou,
        "max_det":          args.max_det,
        "stride":           args.stride,
        "resize":           (f"{frame_wh[0]}x{frame_wh[1]}"
                             if frame_wh != orig_wh else "original"),
        "input_fps":        round(input_fps, 4),
        "output_fps":       round(output_fps, 4),
        "original_resolution": {"width": orig_wh[0],  "height": orig_wh[1]},
        "processed_resolution": {"width": frame_wh[0], "height": frame_wh[1]},
        "total_frames":     total_frames,
        "processed_frames": processed_frames,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# --------------------------------------------------------------------------- #
#  Main pipeline
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace):
    # ── Device & model ────────────────────────────────────────────────────── #
    device          = resolve_device(args.device)
    tracker_cfg     = resolve_tracker(args.tracker)
    source_label    = f"{args.model}/{args.tracker}"

    gpu_name = ""
    if device != "cpu" and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(int(device))
        except Exception:
            gpu_name = "GPU"
    device_disp = f"GPU – {gpu_name}" if gpu_name else "CPU"
    print(f"[INFO] Device  : {device} ({device_disp})")
    print(f"[INFO] Model   : {args.model}")
    print(f"[INFO] Tracker : {tracker_cfg}")

    model = YOLO(args.model)

    # ── Input video ───────────────────────────────────────────────────────── #
    cap = open_video(args.video)
    orig_w, orig_h, input_fps, total_frames = get_video_meta(cap)
    print(
        f"[INFO] Video   : {args.video}  "
        f"[{orig_w}x{orig_h} @ {input_fps:.2f} FPS, ~{total_frames} frames]"
    )

    resize_wh = parse_resize(args.resize)
    frame_w   = resize_wh[0] if resize_wh else orig_w
    frame_h   = resize_wh[1] if resize_wh else orig_h
    output_fps = input_fps / max(1, args.stride)

    # ── Output directory ──────────────────────────────────────────────────── #
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext             = "jsonl" if args.save_jsonl else "csv"
    video_out_path  = str(out_dir / "video_track.mp4")
    tracks_path     = str(out_dir / f"tracks.{ext}")
    meta_path       = str(out_dir / "meta.json")
    summary_path    = str(out_dir / "summary.json")

    # ── Writers & diagnostics ─────────────────────────────────────────────── #
    video_writer  = make_video_writer(video_out_path, frame_w, frame_h, output_fps)
    track_writer  = TrackWriter(tracks_path, use_jsonl=args.save_jsonl)
    diag          = TrackDiagnostics()

    # ── Processing loop ───────────────────────────────────────────────────── #
    processed_frames = 0
    loop_start       = time.perf_counter()
    fps_window: List[float] = []
    FPS_WINDOW  = 30

    pbar = tqdm(
        total=total_frames if total_frames > 0 else None,
        unit="fr", desc="Tracking", dynamic_ncols=True,
    )

    frame_idx = -1
    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            pbar.update(1)

            # ── Stride ──────────────────────────────────────────────────── #
            if frame_idx % args.stride != 0:
                continue

            # ── Optional resize ──────────────────────────────────────────── #
            frame = (
                cv2.resize(raw_frame, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                if resize_wh else raw_frame
            )

            # ── Track inference ──────────────────────────────────────────── #
            t0 = time.perf_counter()
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_cfg,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                classes=[PERSON_CLASS_ID],
                max_det=args.max_det,
                device=device,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0

            fps_window.append(1.0 / max(elapsed, 1e-6))
            if len(fps_window) > FPS_WINDOW:
                fps_window.pop(0)
            fps_estimate = sum(fps_window) / len(fps_window)

            timestamp_sec = frame_idx / input_fps

            # ── Extract tracks ───────────────────────────────────────────── #
            tracks = extract_tracks(results)

            # ── Diagnostics update ───────────────────────────────────────── #
            diag.update(frame_idx, tracks)

            # ── Write track rows ──────────────────────────────────────────── #
            track_writer.write_frame(
                frame_idx     = frame_idx,
                timestamp_sec = timestamp_sec,
                tracks        = tracks,
                known_ids     = diag.known_ids,
                track_ages    = diag.track_ages,
                source_label  = source_label,
            )
            # Register new IDs into known set AFTER writing (is_new_track flag)
            for t in tracks:
                diag.known_ids.add(t["track_id"])

            processed_frames += 1

            # ── Draw & write output frame ─────────────────────────────────── #
            annotated = draw_tracks(
                frame.copy(), tracks, frame_idx, fps_estimate, len(tracks)
            )
            video_writer.write(annotated)

            # ── Optional preview ──────────────────────────────────────────── #
            if args.preview:
                cv2.imshow("Module B – Tracking Preview", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Preview closed by user.")
                    break

            pbar.set_postfix(fps=f"{fps_estimate:.1f}", trk=len(tracks))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        pbar.close()
        cap.release()
        video_writer.release()
        track_writer.close()
        if args.preview:
            cv2.destroyAllWindows()

    # ── Post-run metrics ──────────────────────────────────────────────────── #
    total_elapsed = time.perf_counter() - loop_start
    avg_proc_fps  = processed_frames / max(total_elapsed, 1e-6)

    # ── Save meta & summary ───────────────────────────────────────────────── #
    save_meta(
        path             = meta_path,
        args             = args,
        device           = device,
        tracker_resolved = tracker_cfg,
        input_fps        = input_fps,
        output_fps       = output_fps,
        orig_wh          = (orig_w, orig_h),
        frame_wh         = (frame_w, frame_h),
        total_frames     = frame_idx + 1,
        processed_frames = processed_frames,
    )

    summary = diag.compute_summary(
        total_frames     = frame_idx + 1,
        processed_frames = processed_frames,
        avg_proc_fps     = avg_proc_fps,
        tracker_label    = args.tracker,
        model_label      = args.model,
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Console summary ───────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("  TRACKING SUMMARY")
    print("=" * 60)
    print(f"  Total frames in video     : {frame_idx + 1}")
    print(f"  Processed frames          : {processed_frames}")
    print(f"  Avg processing FPS        : {avg_proc_fps:.2f}")
    print(f"  Unique track IDs          : {summary['unique_track_ids']}")
    print(f"  Avg active tracks/frame   : {summary['avg_active_tracks_per_frame']:.2f}")
    print(f"  Lost-track events (proxy) : {summary['lost_track_events']}")
    print(f"  ID-switch proxy count     : {summary['id_switch_proxy']['count']}")
    print(f"  Track-length histogram    : {summary['track_length_histogram']}")
    print("-" * 60)
    print(f"  Output video   : {video_out_path}")
    print(f"  Tracks file    : {tracks_path}")
    print(f"  Meta file      : {meta_path}")
    print(f"  Summary file   : {summary_path}")
    print("=" * 60 + "\n")


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    run(args)
