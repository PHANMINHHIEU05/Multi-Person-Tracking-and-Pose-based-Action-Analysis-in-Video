"""
Module B (Enhanced) – Multi-Person Tracking with BoT-SORT + ID Stability
=========================================================================
Track up to ~20 people in 720p video with Ultralytics YOLOv8 + BoT-SORT.
Adds two post-processing layers on top of raw tracker output:

  1. Motion Gating   – reject detections that imply impossible movement.
  2. Short-Gap Track Stitching – merge a new ID back into the old one when
     the same person reappears after a brief disappearance.

Outputs (runs/track/<run_name>/)
---------------------------------
  video_track.mp4  – annotated video
  tracks.csv       – per-frame rows with final + raw track IDs
  summary.json     – diagnostics (stitch_count, unique IDs, …)
  meta.json        – run config snapshot

Usage examples
--------------
  python src/module_b_botsort.py \\
      --video data/in.mp4 --out runs/track/run_botsort \\
      --model yolov8n.pt --device 0 \\
      --tracker config/botsort_custom.yaml \\
      --imgsz 640 --conf 0.35 --resize 1280x720

  python src/module_b_botsort.py \\
      --video data/in.mp4 --out runs/track/run_botsort2 \\
      --enable_hist_stitch --max_gap_frames 30 --max_center_dist 120
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

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
PERSON_CLASS_ID = 0

# Default EMA alpha for velocity smoothing (higher = more reactive)
VEL_EMA_ALPHA = 0.4

# CSV column order (matches output_data_schema in task spec)
_CSV_FIELDS = [
    "frame_idx",
    "timestamp_sec",
    "track_id",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "w",
    "h",
    "is_stitched",
    "raw_track_id",
]


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Module B Enhanced – BoT-SORT tracking with ID stability",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    p.add_argument("--video",   type=str, required=True,
                   help="Path to input video file")
    p.add_argument("--out",     type=str, required=True,
                   help="Output directory for results")
    # Model
    p.add_argument("--model",   type=str, default="yolov8n.pt",
                   help="YOLOv8 weights (e.g. yolov8n.pt, yolov8s.pt)")
    p.add_argument("--device",  type=str, default="auto",
                   help="Device: auto | cpu | 0 (GPU index)")
    p.add_argument("--tracker", type=str, default="config/botsort_custom.yaml",
                   help="Tracker YAML config path")
    # Inference
    p.add_argument("--imgsz",   type=int,   default=640,
                   help="YOLO inference image size")
    p.add_argument("--conf",    type=float, default=0.35,
                   help="Detection confidence threshold")
    p.add_argument("--iou",     type=float, default=0.5,
                   help="NMS IoU threshold")
    p.add_argument("--max_det", type=int,   default=200,
                   help="Max detections per frame")
    # Pre-processing
    p.add_argument("--stride",  type=int,   default=1,
                   help="Process every Nth frame (1 = all)")
    p.add_argument("--resize",  type=str,   default="1280x720",
                   help="Resize frames before YOLO, e.g. 1280x720")
    # Display
    p.add_argument("--preview", action="store_true",
                   help="Show live preview window")
    # Motion gating
    p.add_argument("--enable_motion_gating", type=_bool_flag, default=True,
                   help="Enable motion gating (reject teleporting detections)")
    p.add_argument("--max_px_per_frame", type=int, default=80,
                   help="Max centroid displacement per frame (at 720p)")
    # Short-gap stitching
    p.add_argument("--enable_stitching", type=_bool_flag, default=True,
                   help="Enable short-gap track stitching")
    p.add_argument("--max_gap_frames",   type=int,   default=60,
                   help="Max frames gap between ended and new track for stitch")
    p.add_argument("--max_center_dist",  type=int,   default=200,
                   help="Max centre-to-centre distance (px) for stitch candidate")
    p.add_argument("--min_iou_merge",    type=float, default=0.05,
                   help="Min IoU between last and current bbox for stitch")
    p.add_argument("--enable_hist_stitch", type=_bool_flag, default=True,
                   help="Add HSV histogram similarity check to stitching")
    p.add_argument("--hist_sim_thresh",  type=float, default=0.45,
                   help="Min HSV histogram correlation for stitching")
    return p.parse_args()


def _bool_flag(v: str | bool) -> bool:
    """Argparse type helper for True/False string flags."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


# --------------------------------------------------------------------------- #
#  Utilities
# --------------------------------------------------------------------------- #
def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device_arg


def _resolve_tracker(tracker_arg: str) -> str:
    """Find tracker yaml: explicit path  → local config/  → ultralytics builtin."""
    if os.path.isfile(tracker_arg):
        return tracker_arg
    # Try relative to CWD and script root
    for base in (Path.cwd(), Path(__file__).parent.parent):
        p = base / tracker_arg
        if p.exists():
            return str(p)
    print(f"[WARN] Tracker config '{tracker_arg}' not found on disk – "
          f"passing to Ultralytics as-is (may use built-in).")
    return tracker_arg


def _parse_resize(s: str) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    parts = s.lower().split("x")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _bbox_iou(a: dict, b: dict) -> float:
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


def _hsv_hist(frame: np.ndarray, bbox: dict) -> Optional[np.ndarray]:
    """Compute normalised 2-channel (H,S) histogram for a bbox crop."""
    x1 = max(0, bbox["x1"])
    y1 = max(0, bbox["y1"])
    x2 = min(frame.shape[1], bbox["x2"])
    y2 = min(frame.shape[0], bbox["y2"])
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten()


def _hist_similarity(h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> float:
    """Bhattacharyya similarity (0=no overlap, 1=identical).

    cv2.compareHist with HISTCMP_BHATTACHARYYA returns a *distance* [0,1]
    where 0 = identical.  We flip it so higher = more similar.
    Both arrays must be 1-D float32 of equal length.
    """
    if h1 is None or h2 is None:
        return 0.0
    a = h1.astype(np.float32).flatten()
    b = h2.astype(np.float32).flatten()
    if a.shape != b.shape:
        return 0.0
    dist = cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
    return float(1.0 - np.clip(dist, 0.0, 1.0))


# --------------------------------------------------------------------------- #
#  Video I/O
# --------------------------------------------------------------------------- #
def open_video(path: str) -> cv2.VideoCapture:
    if not Path(path).exists():
        print(f"[ERROR] Video not found: {path}", file=sys.stderr)
        sys.exit(1)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {path}", file=sys.stderr)
        sys.exit(1)
    return cap


def _make_writer(path: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    for code, name in (("avc1", "H.264/avc1"), ("mp4v", "MPEG4/mp4v")):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), fps, (w, h))
        if vw.isOpened():
            print(f"[INFO] VideoWriter codec : {name}")
            return vw
        vw.release()
    print("[ERROR] No usable VideoWriter codec found.", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
#  Drawing
# --------------------------------------------------------------------------- #
def _track_color(tid: int) -> Tuple[int, int, int]:
    """Deterministic BGR colour from track ID (hash-based, high saturation)."""
    rng = np.random.default_rng(int(tid) * 2654435761 & 0xFFFF_FFFF)
    h   = int(rng.integers(0, 180))
    hsv = np.array([[[h, 220, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw(
    frame: np.ndarray,
    tracks: List[dict],
    frame_idx: int,
    fps_disp: float,
) -> np.ndarray:
    """Draw bboxes, labels, and HUD on *frame* (in-place). Returns frame."""
    fh, fw = frame.shape[:2]
    thick  = max(1, fw // 600)
    fscale = max(0.4, fw / 1280 * 0.65)
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for trk in tracks:
        tid   = int(trk["track_id"])
        x1, y1, x2, y2 = trk["x1"], trk["y1"], trk["x2"], trk["y2"]
        conf  = trk["conf"]
        color = _track_color(tid)
        stitched = trk.get("is_stitched", False)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        label = f"ID:{tid}{'*' if stitched else ''}  {conf:.2f}"
        (lw, lh), bl = cv2.getTextSize(label, font, fscale, thick)
        ly = max(y1 - 4, lh + 4)
        cv2.rectangle(frame, (x1, ly - lh - bl), (x1 + lw, ly + bl),
                      color, cv2.FILLED)
        cv2.putText(frame, label, (x1, ly), font, fscale,
                    (255, 255, 255), thick, cv2.LINE_AA)

    active = len(tracks)
    hud = f"Frame:{frame_idx}  FPS:{fps_disp:.1f}  Active:{active}"
    (hw, hh), _ = cv2.getTextSize(hud, font, fscale * 0.85, 1)
    cv2.rectangle(frame, (4, 4), (hw + 12, hh + 14), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, hud, (8, hh + 8), font, fscale * 0.85,
                (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- #
#  Track extraction from Ultralytics results
# --------------------------------------------------------------------------- #
def _extract_raw_tracks(results) -> List[dict]:
    """
    Parse one Ultralytics track result into a list of track dicts.
    Returns [] if tracker hasn't assigned IDs yet (boxes.id is None).
    """
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    if boxes.id is None:
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
            "track_id":     int(ids[i]),
            "raw_track_id": int(ids[i]),   # always preserves original
            "conf":         float(confs[i]),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": x1 + bw // 2,
            "cy": y1 + bh // 2,
            "w":  bw,
            "h":  bh,
            "is_stitched": False,
        })
    return out


# --------------------------------------------------------------------------- #
#  Motion Gating
# --------------------------------------------------------------------------- #
class MotionGate:
    """
    Per-frame filter: flag detections whose centroid moved more than
    `max_px_per_frame` pixels since the last frame, based on EMA-velocity.

    Flagged detections are not removed – they keep their ID but are marked so
    that the stitcher can decide whether to merge them.
    """

    def __init__(self, max_px: int = 80):
        self.max_px = max_px
        # id -> {"cx": ..., "cy": ..., "vx": ..., "vy": ..., "frame": ...}
        self._state: Dict[int, dict] = {}

    def _scale_max(self, frame_h: int) -> float:
        """Scale threshold proportionally to frame height (calibrated at 720)."""
        return self.max_px * (frame_h / 720.0)

    def apply(
        self,
        tracks: List[dict],
        frame_idx: int,
        frame_h: int,
    ) -> List[dict]:
        """
        For each track, check displacement vs EMA-velocity prediction.
        Marks suspicious tracks with low_conf_flag; does NOT remove them.
        Updates velocity state.
        """
        threshold = self._scale_max(frame_h)
        for trk in tracks:
            tid = trk["track_id"]
            cx, cy = trk["cx"], trk["cy"]

            if tid in self._state:
                st = self._state[tid]
                gap = max(1, frame_idx - st["frame"])
                pred_cx = st["cx"] + st["vx"] * gap
                pred_cy = st["cy"] + st["vy"] * gap
                dist = np.hypot(cx - pred_cx, cy - pred_cy)
                if dist > threshold * gap:
                    trk["motion_suspicious"] = True
                # Update EMA velocity
                raw_vx = (cx - st["cx"]) / gap
                raw_vy = (cy - st["cy"]) / gap
                new_vx = VEL_EMA_ALPHA * raw_vx + (1 - VEL_EMA_ALPHA) * st["vx"]
                new_vy = VEL_EMA_ALPHA * raw_vy + (1 - VEL_EMA_ALPHA) * st["vy"]
            else:
                new_vx, new_vy = 0.0, 0.0

            self._state[tid] = {
                "cx": cx, "cy": cy,
                "vx": new_vx, "vy": new_vy,
                "frame": frame_idx,
            }
        return tracks

    def remove(self, tid: int):
        self._state.pop(tid, None)


# --------------------------------------------------------------------------- #
#  Short-Gap Track Stitcher
# --------------------------------------------------------------------------- #
class TrackStitcher:
    """
    Attempt to merge a newly-appeared track ID with a recently-ended track ID
    if they are likely the same person.

    Maintains:
      ended_tracks: {raw_id: EndedTrackInfo} – pool of recently ended tracks.
      remap: {raw_id: root_id}               – chain-resolved ID remap table.
    """

    def __init__(
        self,
        max_gap_frames:  int   = 60,
        max_center_dist: int   = 200,
        min_iou_merge:   float = 0.05,
        use_hist:        bool  = True,
        hist_sim_thresh: float = 0.45,
    ):
        self.max_gap     = max_gap_frames
        self.max_dist    = max_center_dist
        self.min_iou     = min_iou_merge
        self.use_hist    = use_hist
        self.hist_thresh = hist_sim_thresh

        self.remap: Dict[int, int]   = {}  # raw_id -> root_id
        self._ended: Dict[int, dict] = {}  # raw_id -> ended-track info
        self._known: set             = set()
        self.stitch_count: int       = 0

    # -- Public API --------------------------------------------------------- #

    def resolve(self, raw_id: int) -> int:
        """Follow remap chain to the root ID, preventing cycles."""
        visited = set()
        cur = raw_id
        while cur in self.remap and cur not in visited:
            visited.add(cur)
            cur = self.remap[cur]
        return cur

    def update(
        self,
        tracks: List[dict],
        frame_idx: int,
        frame: Optional[np.ndarray],
    ) -> List[dict]:
        """
        Called once per processed frame with the raw-tracker output.

        Steps:
          1. For each track, check if it's new.
          2. If new → search ended_tracks for a stitch candidate.
          3. Update remap; tag is_stitched if merged.
          4. Mark tracks that disappeared this frame into ended_tracks.
        """
        current_raw_ids = {t["track_id"] for t in tracks}

        for trk in tracks:
            raw_id = trk["track_id"]

            if raw_id not in self._known:
                # Brand-new ID from tracker → try to stitch
                candidate = self._find_candidate(raw_id, trk, frame_idx, frame)
                if candidate is not None:
                    root = self.resolve(candidate)
                    self.remap[raw_id] = root
                    self.stitch_count += 1
                self._known.add(raw_id)

            # Apply remap
            root_id = self.resolve(raw_id)
            trk["track_id"]   = root_id
            trk["is_stitched"] = (root_id != raw_id)

        # Expire stale ended entries
        self._expire(frame_idx)

        return tracks

    def mark_ended(
        self,
        raw_id: int,
        frame_idx: int,
        trk: dict,
        hist: Optional[np.ndarray],
        vx: float = 0.0,
        vy: float = 0.0,
    ):
        """Record a track that just disappeared (call when ID drops out)."""
        self._ended[raw_id] = {
            "last_frame": frame_idx,
            "bbox":       {k: trk[k] for k in ("x1", "y1", "x2", "y2")},
            "cx": trk["cx"],
            "cy": trk["cy"],
            "vx": vx,       # last known velocity (px/frame)
            "vy": vy,
            "hist": hist,
        }

    # -- Private helpers ---------------------------------------------------- #

    def _find_candidate(
        self,
        new_id: int,
        trk: dict,
        frame_idx: int,
        frame: Optional[np.ndarray],
    ) -> Optional[int]:
        """
        Search the ended-track pool for the best match to `trk`.

        Scoring (lower = better):
          score = spatial_penalty - iou_bonus - hist_bonus

        Spatial penalty: distance to **velocity-predicted** position
        (falls back to last known position if velocity is tiny).
        This is the key improvement over the naive last-position check:
        someone walking out of frame will re-enter roughly where the
        velocity vector points, not where they were last seen.
        """
        new_hist = (
            _hsv_hist(frame, trk) if (self.use_hist and frame is not None) else None
        )
        best_id    = None
        best_score = float("inf")

        for old_id, info in self._ended.items():
            gap = frame_idx - info["last_frame"]
            if gap < 1 or gap > self.max_gap:
                continue

            # ── Velocity-predicted centre ────────────────────────────────
            vx = info.get("vx", 0.0)
            vy = info.get("vy", 0.0)
            pred_cx = info["cx"] + vx * gap
            pred_cy = info["cy"] + vy * gap

            # Use the better of: predicted-pos dist vs last-known-pos dist
            pred_dist = float(np.hypot(trk["cx"] - pred_cx, trk["cy"] - pred_cy))
            last_dist = float(np.hypot(trk["cx"] - info["cx"], trk["cy"] - info["cy"]))
            dist = min(pred_dist, last_dist)

            iou = _bbox_iou(trk, info["bbox"])

            spatial_ok = (dist <= self.max_dist) or (iou >= self.min_iou)
            if not spatial_ok:
                continue

            # ── Appearance check (optional) ──────────────────────────────
            hist_sim = 0.0
            if self.use_hist and new_hist is not None:
                hist_sim = _hist_similarity(new_hist, info["hist"])
                if hist_sim < self.hist_thresh:
                    continue

            # ── Combined score (lower = better candidate) ────────────────
            # Normalise distance; penalise far candidates, reward IoU & hist
            norm_dist   = dist / max(self.max_dist, 1)
            score = norm_dist - iou * 0.5 - hist_sim * 0.3

            if score < best_score:
                best_score = score
                best_id    = old_id

        return best_id

    def _expire(self, frame_idx: int):
        # Keep ended entries for max_gap + grace period so stitching
        # can still succeed for tracks that reappear right at the edge.
        stale = [k for k, v in self._ended.items()
                 if frame_idx - v["last_frame"] > self.max_gap + 15]
        for k in stale:
            del self._ended[k]


# --------------------------------------------------------------------------- #
#  Tracking state tracker (for computing ended tracks)
# --------------------------------------------------------------------------- #
class ActiveTracker:
    """
    Keeps track of which raw IDs were active in the last two frames so we can
    (a) detect when a track disappears and feed it into the stitcher's ended pool,
    (b) supply a velocity estimate (px/frame) for more accurate re-matching.
    """

    def __init__(self):
        self._prev_ids:  set             = set()
        self._last_trk:  Dict[int, dict] = {}  # raw_id -> trk dict at frame N-1
        self._prev2_trk: Dict[int, dict] = {}  # raw_id -> trk dict at frame N-2

    def _estimate_velocity(self, raw_id: int) -> Tuple[float, float]:
        """Estimate (vx, vy) in px/frame using last two known positions."""
        cur  = self._last_trk.get(raw_id)
        prev = self._prev2_trk.get(raw_id)
        if cur is None or prev is None:
            return 0.0, 0.0
        return float(cur["cx"] - prev["cx"]), float(cur["cy"] - prev["cy"])

    def update(
        self,
        tracks: List[dict],
        frame_idx: int,
        stitcher: TrackStitcher,
        frame: Optional[np.ndarray],
        use_hist: bool,
    ):
        current_raw_ids = {t["raw_track_id"] for t in tracks}

        # IDs present last frame but absent now → ended
        ended_ids = self._prev_ids - current_raw_ids
        for eid in ended_ids:
            if eid in self._last_trk:
                hist = (
                    _hsv_hist(frame, self._last_trk[eid])
                    if (use_hist and frame is not None)
                    else None
                )
                vx, vy = self._estimate_velocity(eid)
                stitcher.mark_ended(
                    eid, frame_idx - 1, self._last_trk[eid], hist, vx=vx, vy=vy
                )

        # Shift history: prev2 ← prev ← current
        self._prev2_trk = {k: v for k, v in self._last_trk.items()}
        self._prev_ids  = set(current_raw_ids)
        self._last_trk  = {t["raw_track_id"]: t for t in tracks}


# --------------------------------------------------------------------------- #
#  CSV writer (streaming)
# --------------------------------------------------------------------------- #
class TrackCSVWriter:
    def __init__(self, path: str):
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._csv  = csv.DictWriter(self._file, fieldnames=_CSV_FIELDS,
                                    extrasaction="ignore")
        self._csv.writeheader()

    def write_frame(self, frame_idx: int, timestamp_sec: float, tracks: List[dict]):
        for trk in tracks:
            row = {
                "frame_idx":    frame_idx,
                "timestamp_sec": round(timestamp_sec, 4),
                "track_id":     trk["track_id"],
                "conf":         round(trk["conf"], 4),
                "x1":  trk["x1"],  "y1": trk["y1"],
                "x2":  trk["x2"],  "y2": trk["y2"],
                "cx":  trk["cx"],  "cy": trk["cy"],
                "w":   trk["w"],   "h":  trk["h"],
                "is_stitched":  int(trk.get("is_stitched", False)),
                "raw_track_id": trk.get("raw_track_id", trk["track_id"]),
            }
            self._csv.writerow(row)

    def close(self):
        self._file.flush()
        self._file.close()


# --------------------------------------------------------------------------- #
#  Main tracking loop
# --------------------------------------------------------------------------- #
def run_tracking(args: argparse.Namespace) -> dict:
    """
    Main tracking pipeline.

    Returns a summary dict.
    """
    # ── Setup ────────────────────────────────────────────────────────────────
    device   = _resolve_device(args.device)
    tracker  = _resolve_tracker(args.tracker)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    resize   = _parse_resize(args.resize)

    print(f"[INFO] Device       : {device}")
    print(f"[INFO] Tracker cfg  : {tracker}")
    print(f"[INFO] Output dir   : {out_dir}")

    # ── Load model ───────────────────────────────────────────────────────────
    # Device is applied per-call via model.track(device=...).
    # Do NOT call model.to("0") – PyTorch requires "cuda:0", not "0".
    model = YOLO(args.model)

    # ── Open video ───────────────────────────────────────────────────────────
    cap = open_video(args.video)
    vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = resize[0] if resize else vid_w
    out_h = resize[1] if resize else vid_h

    # ── Outputs ──────────────────────────────────────────────────────────────
    video_out    = str(out_dir / "video_track.mp4")
    tracks_out   = str(out_dir / "tracks.csv")
    summary_out  = str(out_dir / "summary.json")
    meta_out     = str(out_dir / "meta.json")

    writer     = _make_writer(video_out, out_w, out_h, vid_fps)
    csv_writer = TrackCSVWriter(tracks_out)

    # ── Post-processors ──────────────────────────────────────────────────────
    motion_gate   = MotionGate(max_px=args.max_px_per_frame)
    stitcher      = TrackStitcher(
        max_gap_frames  = args.max_gap_frames,
        max_center_dist = args.max_center_dist,
        min_iou_merge   = args.min_iou_merge,
        use_hist        = args.enable_hist_stitch,
        hist_sim_thresh = args.hist_sim_thresh,
    )
    active_tracker = ActiveTracker()

    # ── Stats ─────────────────────────────────────────────────────────────────
    processed_frames     = 0
    raw_ids_seen:  set   = set()
    final_ids_seen: set  = set()
    active_per_frame: List[int] = []
    lost_events       = 0
    last_seen_frame: Dict[int, int] = {}
    frame_times: List[float] = []

    # ── Loop ─────────────────────────────────────────────────────────────────
    pbar = tqdm(total=total_frames, unit="fr", desc="Tracking")

    frame_idx = 0
    while True:
        ret, frame_raw = cap.read()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)

        # Stride skip
        if args.stride > 1 and (frame_idx - 1) % args.stride != 0:
            continue

        t0 = time.perf_counter()

        # Resize for processing
        if resize:
            frame = cv2.resize(frame_raw, resize, interpolation=cv2.INTER_LINEAR)
        else:
            frame = frame_raw

        # ── YOLO Tracking ────────────────────────────────────────────────────
        results = model.track(
            frame,
            persist    = True,
            tracker    = tracker,
            imgsz      = args.imgsz,
            conf       = args.conf,
            iou        = args.iou,
            max_det    = args.max_det,
            classes    = [PERSON_CLASS_ID],
            device     = device,
            verbose    = False,
        )

        tracks = _extract_raw_tracks(results)

        # ── Motion Gating ────────────────────────────────────────────────────
        if args.enable_motion_gating:
            tracks = motion_gate.apply(tracks, frame_idx, frame.shape[0])

        # ── Short-gap stitching ──────────────────────────────────────────────
        active_tracker.update(
            tracks, frame_idx, stitcher, frame,
            use_hist=args.enable_hist_stitch,
        )
        if args.enable_stitching:
            tracks = stitcher.update(tracks, frame_idx, frame)
        else:
            for t in tracks:
                t["is_stitched"] = False

        # ── Stats ────────────────────────────────────────────────────────────
        for t in tracks:
            raw_id   = t["raw_track_id"]
            final_id = t["track_id"]
            raw_ids_seen.add(raw_id)
            final_ids_seen.add(final_id)
            # Lost-track proxy: track reappearing after > 10 frame gap
            if final_id in last_seen_frame:
                gap = frame_idx - last_seen_frame[final_id]
                if gap > 10:
                    lost_events += 1
            last_seen_frame[final_id] = frame_idx

        active_per_frame.append(len(tracks))

        # ── Draw ─────────────────────────────────────────────────────────────
        timestamp_sec = (frame_idx - 1) / vid_fps
        fps_disp = (1.0 / (time.perf_counter() - t0 + 1e-9))
        frame = draw(frame, tracks, frame_idx, fps_disp)

        writer.write(frame)

        if args.preview:
            cv2.imshow("BoT-SORT Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # ── CSV (streaming) ──────────────────────────────────────────────────
        csv_writer.write_frame(frame_idx, timestamp_sec, tracks)

        processed_frames += 1
        frame_times.append(time.perf_counter() - t0)

    pbar.close()
    cap.release()
    writer.release()
    csv_writer.close()
    if args.preview:
        cv2.destroyAllWindows()

    avg_fps = 1.0 / (np.mean(frame_times) + 1e-9) if frame_times else 0.0
    avg_active = float(np.mean(active_per_frame)) if active_per_frame else 0.0

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "processed_frames":           processed_frames,
        "unique_track_ids_raw":        len(raw_ids_seen),
        "unique_track_ids_after_remap": len(final_ids_seen),
        "avg_active_tracks_per_frame": round(avg_active, 2),
        "lost_track_events_proxy":     lost_events,
        "stitch_count":                stitcher.stitch_count,
        "processing_fps_avg":          round(avg_fps, 2),
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Meta JSON ────────────────────────────────────────────────────────────
    meta = {
        "model":       args.model,
        "device":      device,
        "tracker":     tracker,
        "imgsz":       args.imgsz,
        "conf":        args.conf,
        "iou":         args.iou,
        "resize":      args.resize,
        "stride":      args.stride,
        "input_fps":   round(vid_fps, 3),
        "output_fps":  round(vid_fps, 3),
        "frame_width":  out_w,
        "frame_height": out_h,
        "enable_motion_gating": args.enable_motion_gating,
        "max_px_per_frame":     args.max_px_per_frame,
        "enable_stitching":     args.enable_stitching,
        "max_gap_frames":       args.max_gap_frames,
        "max_center_dist":      args.max_center_dist,
        "min_iou_merge":        args.min_iou_merge,
        "enable_hist_stitch":   args.enable_hist_stitch,
        "hist_sim_thresh":      args.hist_sim_thresh,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return summary


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    print("[INFO] Starting Module B Enhanced – BoT-SORT with ID Stability")
    print(f"[INFO] Input video : {args.video}")
    print(f"[INFO] Output dir  : {args.out}")
    print(f"[INFO] Motion gate : enabled={args.enable_motion_gating}, "
          f"max_px={args.max_px_per_frame}")
    print(f"[INFO] Stitching   : enabled={args.enable_stitching}, "
          f"max_gap={args.max_gap_frames}, "
          f"max_dist={args.max_center_dist}, "
          f"hist={args.enable_hist_stitch}")

    t_start = time.perf_counter()
    summary = run_tracking(args)
    elapsed = time.perf_counter() - t_start

    print("\n[DONE] ─────────────────────────────────────────────")
    print(f"  Processed frames           : {summary['processed_frames']}")
    print(f"  Unique track IDs (raw)     : {summary['unique_track_ids_raw']}")
    print(f"  Unique track IDs (remapped): {summary['unique_track_ids_after_remap']}")
    print(f"  Avg active tracks / frame  : {summary['avg_active_tracks_per_frame']}")
    print(f"  Lost-track events (proxy)  : {summary['lost_track_events_proxy']}")
    print(f"  Stitch count               : {summary['stitch_count']}")
    print(f"  Processing FPS (avg)       : {summary['processing_fps_avg']}")
    print(f"  Wall-clock time            : {elapsed:.1f}s")
    print(f"  Output dir                 : {args.out}")
    print("────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
