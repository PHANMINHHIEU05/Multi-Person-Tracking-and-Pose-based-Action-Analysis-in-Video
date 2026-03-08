"""
Module C – Pose-based Action Recognition on tracked persons (v3)
================================================================
Pipeline:
  1. YOLOv8-Pose + BoT-SORT tracking
  2. Per track_id: rolling buffer of 128 frames raw keypoints
  3. Predict with v3 features: hip-centred norm + vel/acc + aspect ratio (69-dim)
  4. Update every PRED_STRIDE=16 new frames (sliding window)
  5. Output annotated video + CSV

Model:  runs/train_horizontal/final_safe_system.pth
Pose:   yolov8n-pose.pt

Usage:
    python src/module_c_action.py --video data/video/input.mp4
    python src/module_c_action.py --video data/video/input.mp4 \\
        --model_path runs/train_horizontal/final_safe_system.pth \\
        --pose_model yolov8n-pose.pt --out runs/action/run1 --preview
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
from ultralytics import YOLO

# Import model architecture from train_professional_v3
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_professional_v3 import ActionRecognitionModel


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {
    0: "Fall",
    1: "Walking",
    2: "Sitting_Quickly",
    3: "Bending",
    4: "Lying_Down",
}

# BGR colours per class
LABEL_COLORS = {
    0: (0,   0,   255),   # Fall            → red
    1: (0,   200, 0),     # Walking         → green
    2: (255, 0,   200),   # Sitting_Quickly → purple
    3: (255, 140, 0),     # Bending         → orange
    4: (0,   200, 255),   # Lying_Down      → cyan
}

SEQ_LEN       = 128    # phải khớp với lúc train
PRED_STRIDE   = 16     # cập nhật prediction mỗi PRED_STRIDE frame mới
CONF_THRESHOLD = 0.55  # ngưỡng tin cậy tối thiểu để coi prediction là hợp lệ
MIN_TRACK_FRAMES = 80  # cần ít nhất N frame trước khi predict
SMOOTH_WINDOW  = 3     # majority vote qua N lần predict gần nhất per track


# COCO keypoint indices
NOSE = 0
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16
L_SHOULDER, R_SHOULDER = 5, 6
MA_WINDOW = 5


# ─────────────────────────────────────────────────────────────────────────────
#  v3 Preprocessing (must match data_prepare_v3.py)
# ─────────────────────────────────────────────────────────────────────────────
def _hip_centered_normalize(seq: np.ndarray) -> np.ndarray:
    """(T,17,2) raw [0,1] → hip-centred, skeleton-height-scaled."""
    result = np.zeros_like(seq)
    for t in range(seq.shape[0]):
        kp = seq[t]
        mid_hip   = (kp[L_HIP] + kp[R_HIP]) / 2.0
        mid_ankle = (kp[L_ANKLE] + kp[R_ANKLE]) / 2.0
        skel_h    = np.linalg.norm(kp[NOSE] - mid_ankle)
        if skel_h < 0.05:
            mid_sh = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2.0
            skel_h = np.linalg.norm(mid_sh - mid_hip) * 3.0
        skel_h = max(skel_h, 0.01)
        result[t] = (kp - mid_hip) / skel_h
    return result


def _bbox_aspect_ratio(seq: np.ndarray) -> np.ndarray:
    """(T,17,2) raw → (T,1) W/H from keypoint bounding box."""
    T = seq.shape[0]
    ar = np.ones((T, 1), dtype=np.float32)
    for t in range(T):
        kp = seq[t]
        valid = np.any(kp != 0, axis=1)
        if valid.sum() < 2:
            continue
        vk = kp[valid]
        w = vk[:, 0].max() - vk[:, 0].min()
        h = vk[:, 1].max() - vk[:, 1].min()
        ar[t, 0] = w / max(h, 1e-4)
    return ar


def prepare_sequence(buffer: deque, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Convert buffer (deque of (17,2) raw normalised arrays) → (1, SEQ_LEN, 69).
    Applies: MA filter → hip-centre → vel/acc → aspect ratio.
    """
    frames = list(buffer)[-seq_len:]
    if len(frames) < seq_len:
        # Repeat first frame (not zeros) to avoid triggering "fall" on padding
        first = frames[0]
        pad = [first.copy()] * (seq_len - len(frames))
        frames = pad + frames
    seq = np.stack(frames, axis=0).astype(np.float32)   # (SEQ_LEN, 17, 2)

    # Aspect ratio (from raw coords, before centring)
    ar = _bbox_aspect_ratio(seq)                          # (SEQ_LEN, 1)

    # Moving-average temporal filter
    for k in range(17):
        for c in range(2):
            seq[:, k, c] = uniform_filter1d(seq[:, k, c], size=MA_WINDOW,
                                            mode="nearest")

    # Hip-centred normalisation
    normed = _hip_centered_normalize(seq)                 # (SEQ_LEN, 17, 2)

    # Velocity + acceleration
    T, K, _ = normed.shape
    delta = np.zeros_like(normed)
    delta[1:] = normed[1:] - normed[:-1]
    vel = np.linalg.norm(delta, axis=-1)                  # (T, 17)
    acc = np.zeros_like(vel)
    acc[1:] = np.abs(vel[1:] - vel[:-1])

    features = np.concatenate([
        normed,                                            # (T,17,2)
        vel[..., np.newaxis],                              # (T,17,1)
        acc[..., np.newaxis],                              # (T,17,1)
    ], axis=-1)                                            # (T,17,4)

    flat = features.reshape(seq_len, -1)                   # (T, 68)
    full = np.concatenate([flat, ar], axis=-1)             # (T, 69)
    return full[np.newaxis].astype(np.float32)             # (1, SEQ_LEN, 69)


# ─────────────────────────────────────────────────────────────────────────────
#  Action Recognizer
# ─────────────────────────────────────────────────────────────────────────────
class ActionRecognizer:
    """
    Wrapper inference-only cho ActionRecognitionModel.
    Duy trì rolling buffer keypoints per track_id.
    """

    def __init__(self, model_path: str, device: str = "auto",
                 num_classes: int = 5, hidden_dim: int = 128,
                 num_layers: int = 3, num_heads: int = 8):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = ActionRecognitionModel(
            input_dim=69,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=0.0,          # inference: no dropout
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        # Checkpoint có thể là dict {model_state_dict: ...} hoặc state_dict thẳng
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[Module C] Loaded action model from {model_path}  (device={self.device})")

        # Load StandardScaler statistics (nếu có trong checkpoint)
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std:  Optional[np.ndarray] = None
        if isinstance(checkpoint, dict):
            fm = checkpoint.get("feat_mean")
            fs = checkpoint.get("feat_std")
            if fm is not None and fs is not None:
                self.feat_mean = np.array(fm, dtype=np.float32)
                self.feat_std  = np.array(fs, dtype=np.float32)
                print(f"[Module C] StandardScaler loaded: mean={self.feat_mean.mean():.4f} std={self.feat_std.mean():.4f}")
            else:
                print("[Module C] No scaler in checkpoint — running without normalization")

        # Per-track state
        self._buffers: Dict[int, deque] = {}        # track_id → deque of (17,2)
        self._frame_count: Dict[int, int] = {}      # track_id → số frame đã thấy
        self._last_pred: Dict[int, Tuple[int, float]] = {}   # track_id → (label, conf)
        self._frames_since_pred: Dict[int, int] = {}         # track_id → frame kể từ predict gần nhất
        self._pred_history: Dict[int, list] = {}             # track_id → list of recent label_ids

    def update_track(self, track_id: int, kpts_norm: Optional[np.ndarray]):
        """
        Thêm 1 frame keypoints cho track_id.
        kpts_norm: (17, 2) đã normalize [0,1], hoặc None nếu không detect.
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=SEQ_LEN)
            self._frame_count[track_id] = 0
            self._last_pred[track_id] = (-1, 0.0)
            self._frames_since_pred[track_id] = 0

        if kpts_norm is None:
            # Dùng frame cuối cùng (forward-fill) hoặc zero
            if len(self._buffers[track_id]) > 0:
                self._buffers[track_id].append(self._buffers[track_id][-1].copy())
            else:
                self._buffers[track_id].append(np.zeros((17, 2), dtype=np.float32))
        else:
            self._buffers[track_id].append(kpts_norm)

        self._frame_count[track_id] += 1
        self._frames_since_pred[track_id] += 1

    @torch.no_grad()
    def predict(self, track_id: int) -> Tuple[int, float, str]:
        """
        Chạy inference nếu:
          - Đủ MIN_TRACK_FRAMES frame
          - Đã đến lượt predict (PRED_STRIDE)
        Returns: (label_id, confidence, label_name)
        """
        n_frames = self._frame_count.get(track_id, 0)
        since_pred = self._frames_since_pred.get(track_id, 0)

        if n_frames < MIN_TRACK_FRAMES or since_pred < PRED_STRIDE:
            # Trả về kết quả cũ
            lid, conf = self._last_pred.get(track_id, (-1, 0.0))
            return lid, conf, LABEL_MAP.get(lid, "?")

        # Reset counter
        self._frames_since_pred[track_id] = 0

        x = prepare_sequence(self._buffers[track_id])          # (1, 128, 69)
        # Apply StandardScaler if available
        if self.feat_mean is not None:
            x = (x - self.feat_mean) / self.feat_std
        xt = torch.FloatTensor(x).to(self.device)
        logits, _ = self.model(xt)                             # (1, num_classes)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        label_id = int(np.argmax(probs))
        confidence = float(probs[label_id])

        # Chỉ cập nhật nếu confidence đủ cao
        if confidence >= CONF_THRESHOLD:
            # Temporal smoothing: majority vote qua SMOOTH_WINDOW lần predict gần nhất
            hist = self._pred_history.setdefault(track_id, [])
            hist.append(label_id)
            if len(hist) > SMOOTH_WINDOW:
                hist.pop(0)
            # Majority vote
            from collections import Counter
            smoothed_id = Counter(hist).most_common(1)[0][0]
            self._last_pred[track_id] = (smoothed_id, confidence)

        lid, conf = self._last_pred[track_id]
        return lid, conf, LABEL_MAP.get(lid, "?")

    def remove_stale_tracks(self, active_ids: set, max_age: int = 90):
        """Giải phóng bộ nhớ cho track không còn active."""
        dead = [tid for tid in self._buffers if tid not in active_ids]
        for tid in dead:
            del self._buffers[tid]
            del self._frame_count[tid]
            del self._last_pred[tid]
            del self._frames_since_pred[tid]
            self._pred_history.pop(tid, None)


# ─────────────────────────────────────────────────────────────────────────────
#  Keypoint extraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_kpts_for_track(result, track_id: int,
                           w: int, h: int) -> Optional[np.ndarray]:
    """
    Từ kết quả YOLOv8-Pose (1 frame, multi-person), tìm person có
    track_id tương ứng và trả về keypoints normalize (17, 2).
    """
    if result.keypoints is None or result.boxes is None:
        return None

    boxes = result.boxes
    if boxes.id is None:
        return None

    track_ids = boxes.id.cpu().numpy().astype(int)
    idx_list = np.where(track_ids == track_id)[0]
    if len(idx_list) == 0:
        return None

    idx = idx_list[0]
    kpts = result.keypoints

    if idx >= kpts.xy.shape[0]:
        return None

    xy = kpts.xy[idx].cpu().numpy().astype(np.float32)  # (17, 2)

    # Normalize về [0,1]
    norm = xy.copy()
    if w > 0:
        norm[:, 0] /= w
    if h > 0:
        norm[:, 1] /= h
    norm = np.clip(norm, 0.0, 1.0)

    # Coi như missing nếu tất cả zero
    if np.all(norm == 0):
        return None

    return norm


# ─────────────────────────────────────────────────────────────────────────────
#  Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),                  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 11), (6, 12), (11, 12),                        # torso
    (11, 13), (13, 15), (12, 14), (14, 16),            # legs
]


def draw_skeleton(frame: np.ndarray, kpts_norm: np.ndarray,
                  color: Tuple[int, int, int], w: int, h: int):
    """Vẽ skeleton từ keypoints đã normalize lên frame."""
    pts = (kpts_norm * np.array([w, h])).astype(int)
    valid = ~np.all(kpts_norm == 0, axis=-1)   # (17,)

    for i, j in SKELETON:
        if valid[i] and valid[j]:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, 2, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        if valid[i]:
            cv2.circle(frame, tuple(pt), 4, color, -1, cv2.LINE_AA)


def draw_action_label(frame: np.ndarray, bbox: np.ndarray,
                      track_id: int, label: str, conf: float,
                      color: Tuple[int, int, int]):
    """Vẽ bounding box + track ID + action label."""
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"#{track_id} {label} {conf:.0%}" if conf > 0 else f"#{track_id} ..."
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(frame, text, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  Main processing loop
# ─────────────────────────────────────────────────────────────────────────────
def process_video(args):
    # ── Setup ───────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLOv8-Pose
    device_str = args.device if args.device != "auto" else ("0" if torch.cuda.is_available() else "cpu")
    pose_model = YOLO(args.pose_model)
    pose_model.to("cuda" if device_str == "0" else device_str)

    # Load Action Recognizer
    recognizer = ActionRecognizer(
        model_path=args.model_path,
        device=args.device,
        num_classes=5,
        hidden_dim=128,
        num_layers=3,
        num_heads=8,
    )

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video
    out_video_path = str(out_dir / "video_action.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (W, H))

    # CSV kết quả
    csv_path = str(out_dir / "actions.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "track_id", "action", "confidence",
                         "x1", "y1", "x2", "y2"])

    print(f"\n[Module C] Video: {args.video}  ({W}×{H} @ {fps:.1f}fps, {total} frames)")
    print(f"[Module C] Output: {out_dir}/")

    # ── Tracking config ────────────────────────────────────────────────
    tracker_cfg = args.tracker   # "botsort.yaml" or "bytetrack.yaml"

    frame_idx = 0
    pbar = tqdm(total=total, desc="Processing", unit="f")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. YOLOv8-Pose + Tracking ─────────────────────────────────
        results = pose_model.track(
            frame,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            classes=[0],          # chỉ detect người
            tracker=tracker_cfg,
            verbose=False,
        )
        result = results[0]

        active_ids = set()

        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            bboxes    = result.boxes.xyxy.cpu().numpy()

            for i, tid in enumerate(track_ids):
                active_ids.add(tid)
                bbox = bboxes[i]

                # ── 2. Lấy keypoints cho người này ────────────────────
                kpts = extract_kpts_for_track(result, tid, W, H)
                recognizer.update_track(tid, kpts)

                # ── 3. Predict action ──────────────────────────────────
                label_id, conf, label_name = recognizer.predict(tid)
                color = LABEL_COLORS.get(label_id, (200, 200, 200))

                # ── 4. Vẽ lên frame ───────────────────────────────────
                if args.draw_skeleton and kpts is not None:
                    draw_skeleton(frame, kpts, color, W, H)
                draw_action_label(frame, bbox, tid, label_name, conf, color)

                # ── 5. Ghi CSV ────────────────────────────────────────
                x1, y1, x2, y2 = bbox.astype(int)
                csv_writer.writerow([frame_idx, tid, label_name,
                                     f"{conf:.4f}", x1, y1, x2, y2])

        # Dọn track cũ
        recognizer.remove_stale_tracks(active_ids)

        writer.write(frame)

        if args.preview:
            cv2.imshow("Module C – Action Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Dừng sớm theo yêu cầu người dùng.")
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    csv_file.close()
    if args.preview:
        cv2.destroyAllWindows()

    # ── Summary ────────────────────────────────────────────────────────
    summary = {
        "video": args.video,
        "frames_processed": frame_idx,
        "fps": fps,
        "resolution": f"{W}x{H}",
        "output_video": out_video_path,
        "actions_csv": csv_path,
        "action_classes": LABEL_MAP,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Module C] Done! {frame_idx} frames processed.")
    print(f"  → Video  : {out_video_path}")
    print(f"  → CSV    : {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Module C – Pose-based Action Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",       required=True,
                   help="Đường dẫn tới video input")
    p.add_argument("--model_path",  default="runs/train_horizontal/final_safe_system.pth",
                   help="Path to trained .pth checkpoint")
    p.add_argument("--pose_model",  default="yolov8n-pose.pt",
                   help="YOLOv8-Pose weight")
    p.add_argument("--out",         default="runs/action/run1",
                   help="Thư mục output")
    p.add_argument("--tracker",     default="botsort.yaml",
                   choices=["botsort.yaml", "bytetrack.yaml"],
                   help="Tracker config")
    p.add_argument("--device",      default="auto",
                   help="auto | cuda | cpu")
    p.add_argument("--conf",        type=float, default=0.25,
                   help="YOLO confidence threshold (thấp hơn = detect nhiều người hơn)")
    p.add_argument("--iou",         type=float, default=0.45,
                   help="IoU threshold for NMS")
    p.add_argument("--imgsz",       type=int,   default=640,
                   help="Inference image size — dùng 1280 nếu người nhỏ bị miss")
    p.add_argument("--max_det",     type=int,   default=50,
                   help="Số người tối đa detect mỗi frame")
    p.add_argument("--draw_skeleton", action="store_true", default=True,
                   help="Vẽ skeleton lên video output")
    p.add_argument("--preview",     action="store_true",
                   help="Hiển thị real-time preview window")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    process_video(args)
