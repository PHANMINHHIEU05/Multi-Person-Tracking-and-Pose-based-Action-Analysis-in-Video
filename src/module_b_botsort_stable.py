"""
Module B++ (Stable) – BoT-SORT + Feature Memory Bank + Trajectory Merge
========================================================================
Full pipeline:
  1. YOLOv8 + BoT-SORT online tracking
  2. Tentative→Confirmed ID policy (avoid trusting face-only first frames)
  3. Online re-association via Feature Memory Bank (upper-body HSV + motion)
  4. Optional offline trajectory merge (global optimization lite on tracks.csv)

Outputs  (runs/track/<run_name>/)
---------------------------------
  video_track.mp4       annotated video
  tracks.csv            per-frame rows (raw + stable IDs, status, is_stitched)
  tracks_merged.csv     after offline merge (optional)
  meta.json             run settings
  summary.json          diagnostics

Usage examples
--------------
  python src/module_b_botsort_stable.py \\
      --video data/video/video1.mp4 \\
      --out runs/track/run_stable \\
      --model yolov8n.pt --device 0 \\
      --imgsz 640 --conf 0.35 --resize 1280x720

  python src/module_b_botsort_stable.py \\
      --video data/video/video1.mp4 \\
      --out runs/track/run_stable2 \\
      --disable_offline_merge --ttl_seconds 3.0 --final_score_thresh 0.75
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
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Add project root to path so utils package is importable when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.appearance import extract_feature
from src.utils.stitching import (
    IDRemapper,
    MemoryBank,
    OnlineReassociator,
    TrackState,
)
from src.utils.trajectory_merge import run_offline_merge

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
PERSON_CLASS_ID = 0
SOURCE_LABEL    = "botsort"

_CSV_FIELDS = [
    "frame_idx",
    "timestamp_sec",
    "raw_track_id",
    "stable_track_id",
    "conf",
    "x1", "y1", "x2", "y2",
    "cx", "cy", "w", "h",
    "status",
    "is_stitched_online",
    "source_tracker",
]


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Module B++ – BoT-SORT + Memory Bank + Offline Merge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    p.add_argument("--video",   required=True,  help="Input video path")
    p.add_argument("--out",     required=True,  help="Output directory")
    # Model
    p.add_argument("--model",   default="yolov8n.pt")
    p.add_argument("--device",  default="auto",
                   help="auto | cpu | 0 (GPU index)")
    p.add_argument("--tracker", default="config/botsort_custom.yaml")
    # Inference
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--conf",    type=float, default=0.35)
    p.add_argument("--iou",     type=float, default=0.5)
    p.add_argument("--max_det", type=int,   default=200)
    # Pre-processing
    p.add_argument("--stride",  type=int,   default=1,
                   help="Process every Nth frame")
    p.add_argument("--resize",  type=str,   default="1280x720",
                   help="Resize frames, e.g. 1280x720")
    # Display
    p.add_argument("--enable_preview", action="store_true")
    # Tentative / confirmed
    p.add_argument("--confirm_min_frames",   type=int,   default=6)
    p.add_argument("--confirm_min_bbox_h",   type=int,   default=60)
    p.add_argument("--confirm_min_avg_conf", type=float, default=0.35)
    # Suppress: minimum frames before showing ID on screen at all
    p.add_argument("--display_min_frames",   type=int,   default=4,
                   help="A track must exist >= this many frames to appear on video")
    # Memory bank
    p.add_argument("--ttl_seconds",   type=float, default=15.0)
    # Online re-association
    p.add_argument("--max_gap_frames",    type=int,   default=200)
    p.add_argument("--max_center_dist",   type=int,   default=400)
    p.add_argument("--alpha_pos",         type=float, default=0.45)
    p.add_argument("--min_app_sim",       type=float, default=0.30)
    p.add_argument("--final_score_thresh",type=float, default=0.35)
    # Feature extractor
    p.add_argument("--hist_h_bins", type=int, default=30)
    p.add_argument("--hist_s_bins", type=int, default=32)
    # Offline merge
    p.add_argument("--disable_online_stitch",  action="store_true")
    p.add_argument("--disable_offline_merge",  action="store_true")
    p.add_argument("--offline_max_gap",        type=int,   default=200)
    p.add_argument("--offline_max_join_dist",  type=float, default=350.0)
    p.add_argument("--offline_max_dir_change", type=float, default=90.0)
    p.add_argument("--offline_min_seg_len",    type=int,   default=5)
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Device & tracker resolution
# --------------------------------------------------------------------------- #
def _resolve_device(arg: str) -> str:
    if arg == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return arg


def _resolve_tracker(arg: str) -> str:
    if os.path.isfile(arg):
        return arg
    for base in (Path.cwd(), Path(__file__).parent.parent):
        p = base / arg
        if p.exists():
            return str(p)
    print(f"[WARN] Tracker '{arg}' not found on disk – passing as-is.")
    return arg


def _parse_resize(s: str) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    parts = s.lower().split("x")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------- #
#  Video I/O
# --------------------------------------------------------------------------- #
def open_video(path: str) -> cv2.VideoCapture:
    if not Path(path).exists():
        print(f"[ERROR] Video not found: {path}", file=sys.stderr)
        sys.exit(1)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {path}", file=sys.stderr)
        sys.exit(1)
    return cap


def _make_writer(path: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    for code, label in (("avc1", "H.264"), ("mp4v", "MPEG4")):
        vw = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*code), fps, (w, h)
        )
        if vw.isOpened():
            print(f"[INFO] VideoWriter : {label}")
            return vw
        vw.release()
    print("[ERROR] No VideoWriter codec found.", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
#  Drawing
# --------------------------------------------------------------------------- #
def _track_color(tid: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(int(tid) * 2654435761 & 0xFFFF_FFFF)
    h   = int(rng.integers(0, 180))
    hsv = np.array([[[h, 210, 230]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_frame(
    frame:     np.ndarray,
    states:    List[dict],
    frame_idx: int,
    fps_est:   float,
) -> np.ndarray:
    """
    Draw bboxes + labels on frame.

    Label format:  ID:<stable_id>[raw:<raw_id>]  <conf:.2f>  [T] if tentative
    """
    fh, fw = frame.shape[:2]
    thick  = max(1, fw // 600)
    fsc    = max(0.38, fw / 1280 * 0.60)
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for s in states:
        sid   = int(s["stable_track_id"])
        rid   = int(s["raw_track_id"])
        x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
        conf  = float(s["conf"])
        stat  = s["status"]
        stitched = bool(s["is_stitched_online"])
        color = _track_color(sid)

        # Dashed outline for tentative tracks
        if stat == "tentative":
            # Draw corner markers instead of full rect
            cl = max(10, min(30, (x2 - x1) // 4))
            for px, py, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(frame, (px, py), (px + dx * cl, py), color, thick)
                cv2.line(frame, (px, py), (px, py + dy * cl), color, thick)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # Label
        raw_tag = f"[r{rid}]" if sid != rid else ""
        stitch_tag = "*" if stitched else ""
        tent_tag   = "T " if stat == "tentative" else ""
        label = f"{tent_tag}ID:{sid}{stitch_tag}{raw_tag}  {conf:.2f}"
        (lw, lh), bl = cv2.getTextSize(label, font, fsc, thick)
        ly = max(y1 - 4, lh + 4)
        cv2.rectangle(frame, (x1, ly - lh - bl),
                      (x1 + lw, ly + bl), color, cv2.FILLED)
        cv2.putText(frame, label, (x1, ly), font, fsc,
                    (255, 255, 255), thick, cv2.LINE_AA)

    # HUD
    n_conf = sum(1 for s in states if s["status"] == "confirmed")
    n_tent = len(states) - n_conf
    hud = (
        f"Frame:{frame_idx}  FPS:{fps_est:.1f}  "
        f"Confirmed:{n_conf}  Tentative:{n_tent}"
    )
    (hw, hh), _ = cv2.getTextSize(hud, font, fsc * 0.82, 1)
    cv2.rectangle(frame, (4, 4), (hw + 12, hh + 14), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, hud, (8, hh + 8), font,
                fsc * 0.82, (180, 220, 180), 1, cv2.LINE_AA)

    return frame


# --------------------------------------------------------------------------- #
#  Raw track extraction from Ultralytics results
# --------------------------------------------------------------------------- #
def _extract_detections(results) -> List[dict]:
    """
    Parse one Ultralytics track result.
    Returns list of {raw_id, conf, x1, y1, x2, y2, cx, cy, w, h}.
    """
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return []

    xyxy  = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss  = boxes.cls.cpu().numpy().astype(int)
    ids   = boxes.id.cpu().numpy().astype(int)

    out = []
    for i in range(len(ids)):
        if clss[i] != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = (
            int(xyxy[i, 0]), int(xyxy[i, 1]),
            int(xyxy[i, 2]), int(xyxy[i, 3]),
        )
        bw, bh = x2 - x1, y2 - y1
        out.append({
            "raw_id": int(ids[i]),
            "conf":   float(confs[i]),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": x1 + bw // 2,
            "cy": y1 + bh // 2,
            "w":  bw,
            "h":  bh,
        })
    return out


# --------------------------------------------------------------------------- #
#  CSV writer
# --------------------------------------------------------------------------- #
class StableCSVWriter:
    def __init__(self, path: str):
        self._f   = open(path, "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(
            self._f, fieldnames=_CSV_FIELDS, extrasaction="ignore"
        )
        self._csv.writeheader()

    def write(self, rows: List[dict]):
        for r in rows:
            self._csv.writerow(r)

    def close(self):
        self._f.flush()
        self._f.close()


# --------------------------------------------------------------------------- #
#  Main pipeline
# --------------------------------------------------------------------------- #
def run_tracking(args: argparse.Namespace) -> dict:
    device  = _resolve_device(args.device)
    tracker = _resolve_tracker(args.tracker)
    resize  = _parse_resize(args.resize)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Tracker     : {tracker}")
    print(f"[INFO] Output dir  : {out_dir}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = YOLO(args.model)

    # ── Video ────────────────────────────────────────────────────────────────
    cap          = open_video(args.video)
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = resize[0] if resize else vid_w
    out_h = resize[1] if resize else vid_h

    ttl_frames = max(1, int(args.ttl_seconds * vid_fps))

    # ── Output files ─────────────────────────────────────────────────────────
    vout_path    = str(out_dir / "video_track.mp4")
    csv_path     = str(out_dir / "tracks.csv")
    summary_path = str(out_dir / "summary.json")
    meta_path    = str(out_dir / "meta.json")
    merged_path  = str(out_dir / "tracks_merged.csv")

    video_writer = _make_writer(vout_path, out_w, out_h, vid_fps)
    csv_writer   = StableCSVWriter(csv_path)

    # ── Online components ────────────────────────────────────────────────────
    bank         = MemoryBank(ttl_frames=ttl_frames)
    reassociator = OnlineReassociator(
        max_gap_frames     = args.max_gap_frames,
        max_center_dist    = args.max_center_dist,
        alpha_pos          = args.alpha_pos,
        min_app_sim        = args.min_app_sim,
        final_score_thresh = args.final_score_thresh,
        frame_w            = out_w,
        frame_h            = out_h,
    )
    remapper = IDRemapper()

    # raw_id -> TrackState for currently active tracks
    active: Dict[int, TrackState] = {}
    # raw_ids seen in previous frame (to detect disappearances)
    prev_raw_ids: set = set()

    # ── Stats ─────────────────────────────────────────────────────────────────
    processed_frames   = 0
    all_raw_ids:  set  = set()
    all_stable_ids: set = set()
    active_counts: List[int] = []
    stitch_online  = 0
    lost_proxy     = 0
    last_seen_stable: Dict[int, int] = {}
    frame_times: List[float] = []

    # Already-matched stable IDs this frame (avoid double-assign)
    matched_this_frame: set = set()

    # ── Main loop ────────────────────────────────────────────────────────────
    pbar = tqdm(total=total_frames, unit="fr", desc="Tracking (Stable)")

    frame_idx = 0
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)

        if args.stride > 1 and (frame_idx - 1) % args.stride != 0:
            continue

        t0 = time.perf_counter()

        frame = (
            cv2.resize(raw_frame, resize, interpolation=cv2.INTER_LINEAR)
            if resize else raw_frame
        )

        # ── YOLO tracking ────────────────────────────────────────────────────
        results = model.track(
            frame,
            persist = True,
            tracker = tracker,
            imgsz   = args.imgsz,
            conf    = args.conf,
            iou     = args.iou,
            max_det = args.max_det,
            classes = [PERSON_CLASS_ID],
            device  = device,
            verbose = False,
        )
        detections = _extract_detections(results)
        current_raw_ids = {d["raw_id"] for d in detections}

        # ── Register disappeared tracks into MemoryBank ───────────────────
        disappeared = prev_raw_ids - current_raw_ids
        for rid in disappeared:
            if rid in active:
                st = active.pop(rid)
                bank.add(st.as_bank_record())
        bank.expire(frame_idx)

        # ── Process detections ────────────────────────────────────────────
        matched_this_frame = set()
        frame_rows: List[dict] = []

        for det in detections:
            rid  = det["raw_id"]
            conf = det["conf"]
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            cx, cy = float(det["cx"]), float(det["cy"])

            # Appearance feature (upper-body)
            feat = extract_feature(
                frame, x1, y1, x2, y2,
                h_bins=args.hist_h_bins,
                s_bins=args.hist_s_bins,
            )

            is_new_raw = rid not in active
            is_stitched = False

            # ── Online stitching: brand-new raw_id ───────────────────────
            if is_new_raw and not args.disable_online_stitch:
                best_sid, score = reassociator.search(
                    frame_idx, cx, cy, x1, y1, x2, y2, feat,
                    bank, exclude_stable_ids=matched_this_frame,
                )
                if best_sid is not None:
                    remapper.merge(rid, best_sid)
                    bank.remove(best_sid)
                    matched_this_frame.add(best_sid)
                    is_stitched = True

            # ── Resolve or create stable_id ────────────────────────────
            stable_id = remapper.resolve(rid)

            # ── Create or update TrackState ─────────────────────────────
            if is_new_raw:
                st = TrackState(
                    raw_id           = rid,
                    stable_id        = stable_id,
                    first_seen_frame = frame_idx,
                    last_seen_frame  = frame_idx,
                    cx = cx, cy = cy,
                    x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                    w  = x2 - x1, h = y2 - y1,
                    conf = conf,
                    feat = feat,
                    frame_w = out_w,
                    frame_h = out_h,
                )
                st.seen_frames = 1
                if feat is not None:
                    st.feat_count = 1
                active[rid] = st
            else:
                st = active[rid]
                st.stable_id = stable_id
                st.update(frame_idx, x1, y1, x2, y2, conf, feat)

            st.promote_if_ready(
                args.confirm_min_frames,
                args.confirm_min_bbox_h,
                args.confirm_min_avg_conf,
            )

            # ── Suppress short-lived detections ─────────────────────────
            # If the track has existed for fewer than display_min_frames,
            # do NOT draw it or write it to CSV.  This prevents brief
            # face-only or partial detections from getting a visible ID.
            if st.seen_frames < args.display_min_frames:
                continue

            # ── Stats ────────────────────────────────────────────────────
            all_raw_ids.add(rid)
            all_stable_ids.add(stable_id)
            if stable_id in last_seen_stable:
                if frame_idx - last_seen_stable[stable_id] > 10:
                    lost_proxy += 1
            last_seen_stable[stable_id] = frame_idx

            if is_stitched:
                stitch_online += 1

            # ── CSV row ──────────────────────────────────────────────────
            frame_rows.append({
                "frame_idx":        frame_idx,
                "timestamp_sec":    round((frame_idx - 1) / vid_fps, 4),
                "raw_track_id":     rid,
                "stable_track_id":  stable_id,
                "conf":             round(conf, 4),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": int(cx), "cy": int(cy),
                "w":  x2 - x1, "h": y2 - y1,
                "status":           st.status,
                "is_stitched_online": int(is_stitched),
                "source_tracker":   SOURCE_LABEL,
            })

        prev_raw_ids = current_raw_ids
        active_counts.append(len(detections))

        # ── Write CSV ─────────────────────────────────────────────────────────
        csv_writer.write(frame_rows)

        # ── Draw ─────────────────────────────────────────────────────────────
        fps_disp = 1.0 / (time.perf_counter() - t0 + 1e-9)
        frame = draw_frame(frame, frame_rows, frame_idx, fps_disp)
        video_writer.write(frame)

        if args.enable_preview:
            cv2.imshow("Stable Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        processed_frames += 1
        frame_times.append(time.perf_counter() - t0)

    pbar.close()
    cap.release()
    video_writer.release()
    csv_writer.close()
    if args.enable_preview:
        cv2.destroyAllWindows()

    avg_fps    = 1.0 / (np.mean(frame_times) + 1e-9) if frame_times else 0.0
    avg_active = float(np.mean(active_counts)) if active_counts else 0.0

    # ── Offline trajectory merge ──────────────────────────────────────────────
    merges_offline = 0
    if not args.disable_offline_merge:
        print("[INFO] Running offline trajectory merge …")
        merges_offline = run_offline_merge(
            tracks_csv       = csv_path,
            output_csv       = merged_path,
            max_gap_frames   = args.offline_max_gap,
            max_join_dist_px = args.offline_max_join_dist,
            max_dir_change   = args.offline_max_dir_change,
            min_seg_length   = args.offline_min_seg_len,
        )
        print(f"[INFO] Offline merges : {merges_offline}")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "processed_frames":           processed_frames,
        "unique_raw_ids":              len(all_raw_ids),
        "unique_stable_ids":           len(all_stable_ids),
        "avg_active_tracks_per_frame": round(avg_active, 2),
        "stitch_count_online":         stitch_online,
        "merges_offline_count":        merges_offline,
        "lost_track_events_proxy":     lost_proxy,
        "avg_processing_fps":          round(avg_fps, 2),
        "notes_about_thresholds": (
            f"confirm_min_frames={args.confirm_min_frames}, "
            f"confirm_min_bbox_h={args.confirm_min_bbox_h}, "
            f"display_min_frames={args.display_min_frames}, "
            f"ttl={args.ttl_seconds}s, "
            f"max_gap={args.max_gap_frames}fr, "
            f"min_app_sim={args.min_app_sim}, "
            f"final_score_thresh={args.final_score_thresh}"
        ),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Meta JSON ─────────────────────────────────────────────────────────────
    meta = {
        "model":          args.model,
        "device":         device,
        "tracker_config": tracker,
        "imgsz":          args.imgsz,
        "conf":           args.conf,
        "iou":            args.iou,
        "max_det":        args.max_det,
        "resize":         args.resize,
        "stride":         args.stride,
        "input_fps":      round(vid_fps, 3),
        "output_fps":     round(vid_fps, 3),
        "frame_width":    out_w,
        "frame_height":   out_h,
        "confirm_min_frames":    args.confirm_min_frames,
        "confirm_min_bbox_h":    args.confirm_min_bbox_h,
        "confirm_min_avg_conf":  args.confirm_min_avg_conf,
        "display_min_frames":    args.display_min_frames,
        "ttl_seconds":           args.ttl_seconds,
        "max_gap_frames":        args.max_gap_frames,
        "max_center_dist":       args.max_center_dist,
        "alpha_pos":             args.alpha_pos,
        "min_app_sim":           args.min_app_sim,
        "final_score_thresh":    args.final_score_thresh,
        "offline_merge_enabled": not args.disable_offline_merge,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return summary


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    print("=" * 60)
    print("[INFO] Module B++ – BoT-SORT Stable Tracking")
    print(f"       video  : {args.video}")
    print(f"       output : {args.out}")
    print(f"       confirm: min_frames={args.confirm_min_frames}, "
          f"min_h={args.confirm_min_bbox_h}px, "
          f"display_min={args.display_min_frames}fr")
    print(f"       stitch : online={not args.disable_online_stitch}, "
          f"offline={not args.disable_offline_merge}")
    print(f"       bank   : ttl={args.ttl_seconds}s, "
          f"max_gap={args.max_gap_frames}fr, "
          f"min_sim={args.min_app_sim}")
    print("=" * 60)

    t_start = time.perf_counter()
    summary  = run_tracking(args)
    elapsed  = time.perf_counter() - t_start

    print()
    print("[DONE] ─────────────────────────────────────────────────")
    print(f"  Processed frames             : {summary['processed_frames']}")
    print(f"  Unique raw IDs               : {summary['unique_raw_ids']}")
    print(f"  Unique stable IDs (online)   : {summary['unique_stable_ids']}")
    print(f"  Avg active tracks / frame    : {summary['avg_active_tracks_per_frame']}")
    print(f"  Online stitches              : {summary['stitch_count_online']}")
    print(f"  Offline merges               : {summary['merges_offline_count']}")
    print(f"  Lost-track events (proxy)    : {summary['lost_track_events_proxy']}")
    print(f"  Avg processing FPS           : {summary['avg_processing_fps']}")
    print(f"  Wall-clock time              : {elapsed:.1f}s")
    print(f"  Output dir                   : {args.out}")
    print("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
