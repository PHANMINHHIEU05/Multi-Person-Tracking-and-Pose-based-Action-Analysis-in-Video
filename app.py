from __future__ import annotations

import tempfile
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "runs" / "streamlit_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    0: "Fall",
    1: "Walking",
    2: "Sitting_Quickly",
    3: "Bending",
    4: "Lying_Down",
}

LABEL_COLORS = {
    0: (0, 0, 255),
    1: (0, 200, 0),
    2: (255, 0, 200),
    3: (255, 140, 0),
    4: (0, 200, 255),
}

SEQ_LEN = 128
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

NOSE = 0
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16
L_SHOULDER, R_SHOULDER = 5, 6


@st.cache_resource
def load_pose_model(weights_path: str):
    model = YOLO(weights_path)
    if torch.cuda.is_available():
        model.to("cuda")
    warmup = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(warmup, verbose=False)
    return model


@st.cache_resource
def load_action_model_class():
    from train_professional_v3 import ActionRecognitionModel

    return ActionRecognitionModel


@st.cache_data(show_spinner=False)
def resolve_default_action_model_path() -> str:
    joblib_candidates = [p for p in ROOT.glob("runs/**/extratrees_model.joblib") if p.is_file()]
    if joblib_candidates:
        joblib_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(joblib_candidates[0])

    pth_candidates = [p for p in ROOT.glob("runs/**/final_safe_system.pth") if p.is_file()]
    if pth_candidates:
        pth_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(pth_candidates[0])

    return str(ROOT / "runs" / "train_horizontal" / "final_safe_system.pth")


def _hip_centered_normalize(seq: np.ndarray) -> np.ndarray:
    result = np.zeros_like(seq)
    for t in range(seq.shape[0]):
        kp = seq[t]
        mid_hip = (kp[L_HIP] + kp[R_HIP]) / 2.0
        mid_ankle = (kp[L_ANKLE] + kp[R_ANKLE]) / 2.0
        skel_h = np.linalg.norm(kp[NOSE] - mid_ankle)
        if skel_h < 0.05:
            mid_sh = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2.0
            skel_h = np.linalg.norm(mid_sh - mid_hip) * 3.0
        skel_h = max(skel_h, 0.01)
        result[t] = (kp - mid_hip) / skel_h
    return result


def _bbox_aspect_ratio(seq: np.ndarray) -> np.ndarray:
    t_len = seq.shape[0]
    ar = np.ones((t_len, 1), dtype=np.float32)
    for t in range(t_len):
        kp = seq[t]
        valid = np.any(kp != 0, axis=1)
        if valid.sum() < 2:
            continue
        vk = kp[valid]
        w = vk[:, 0].max() - vk[:, 0].min()
        h = vk[:, 1].max() - vk[:, 1].min()
        ar[t, 0] = w / max(h, 1e-4)
    return ar


def prepare_sequence(buffer: deque, seq_len: int = SEQ_LEN, ma_window: int = 5) -> np.ndarray:
    frames = list(buffer)[-seq_len:]
    if len(frames) < seq_len:
        first = frames[0]
        frames = [first.copy()] * (seq_len - len(frames)) + frames
    seq = np.stack(frames, axis=0).astype(np.float32)

    ar = _bbox_aspect_ratio(seq)

    for k in range(17):
        for c in range(2):
            seq[:, k, c] = uniform_filter1d(seq[:, k, c], size=ma_window, mode="nearest")

    normed = _hip_centered_normalize(seq)

    delta = np.zeros_like(normed)
    delta[1:] = normed[1:] - normed[:-1]
    vel = np.linalg.norm(delta, axis=-1)
    acc = np.zeros_like(vel)
    acc[1:] = np.abs(vel[1:] - vel[:-1])

    features = np.concatenate([
        normed,
        vel[..., np.newaxis],
        acc[..., np.newaxis],
    ], axis=-1)

    flat = features.reshape(seq_len, -1)
    full = np.concatenate([flat, ar], axis=-1)
    return full[np.newaxis].astype(np.float32)


def prepare_sequence_fast(buffer: deque, seq_len: int = 96) -> np.ndarray:
    frames = list(buffer)[-seq_len:]
    if len(frames) < seq_len:
        first = frames[0]
        frames = [first.copy()] * (seq_len - len(frames)) + frames
    seq = np.stack(frames, axis=0).astype(np.float32)

    # Fast mode: skip temporal smoothing to reduce CPU load for tree-based model.
    ar = _bbox_aspect_ratio(seq)
    normed = _hip_centered_normalize(seq)

    delta = np.zeros_like(normed)
    delta[1:] = normed[1:] - normed[:-1]
    vel = np.linalg.norm(delta, axis=-1)
    acc = np.zeros_like(vel)
    acc[1:] = np.abs(vel[1:] - vel[:-1])

    features = np.concatenate(
        [
            normed,
            vel[..., np.newaxis],
            acc[..., np.newaxis],
        ],
        axis=-1,
    )

    flat = features.reshape(seq_len, -1)
    full = np.concatenate([flat, ar], axis=-1)
    return full[np.newaxis].astype(np.float32)


def build_extratrees_feature_vector(seq69: np.ndarray) -> np.ndarray:
    q25 = np.quantile(seq69, 0.25, axis=0)
    q75 = np.quantile(seq69, 0.75, axis=0)
    vel = np.diff(seq69, axis=0)
    feat = np.concatenate(
        [
            seq69.mean(axis=0),
            seq69.std(axis=0),
            seq69.min(axis=0),
            seq69.max(axis=0),
            seq69[0],
            seq69[-1],
            (seq69[-1] - seq69[0]),
            q25,
            q75,
            np.mean(np.abs(vel), axis=0),
            np.std(vel, axis=0),
        ],
        axis=0,
    )
    return feat.astype(np.float32)


class ActionRecognizerLite:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float,
        pred_stride: int,
        min_track_frames: int,
        smooth_window: int,
        fall_conf_boost: float = 0.10,
        sitting_conf_penalty: float = 0.15,
    ):
        self.label_map = LABEL_MAP.copy()
        self.backend = "torch"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.et_model = None
        self.feat_mean = None
        self.feat_std = None

        model_path_l = str(model_path).lower()
        if model_path_l.endswith(".joblib"):
            artifact = joblib.load(model_path)
            if isinstance(artifact, dict) and "model" in artifact:
                self.et_model = artifact["model"]
                lm = artifact.get("label_map")
                if isinstance(lm, dict):
                    self.label_map = {int(k): str(v) for k, v in lm.items()}
            else:
                self.et_model = artifact
            self.backend = "extratrees"
        else:
            ModelClass = load_action_model_class()
            self.model = ModelClass(
                input_dim=69,
                hidden_dim=128,
                num_layers=3,
                num_classes=5,
                num_heads=8,
                dropout=0.0,
            ).to(self.device)

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state)
            self.model.eval()

            if isinstance(checkpoint, dict):
                fm = checkpoint.get("feat_mean")
                fs = checkpoint.get("feat_std")
                if fm is not None and fs is not None:
                    self.feat_mean = np.array(fm, dtype=np.float32)
                    self.feat_std = np.array(fs, dtype=np.float32)

        self.conf_threshold = conf_threshold
        self.pred_stride = max(1, pred_stride)
        self.min_track_frames = max(1, min_track_frames)
        self.smooth_window = max(1, smooth_window)
        self.fall_conf_boost = max(0.0, fall_conf_boost)
        self.sitting_conf_penalty = max(0.0, sitting_conf_penalty)
        self.fast_mode = False
        self.fast_seq_len = 96

        self._buffers: Dict[int, deque] = {}
        self._frame_count: Dict[int, int] = {}
        self._last_pred: Dict[int, Tuple[int, float]] = {}
        self._frames_since_pred: Dict[int, int] = {}
        self._pred_history: Dict[int, list] = {}
        self._feat_cache: Dict[int, np.ndarray] = {}  # FIX: cache computed feature vectors to avoid recomputing when buffer unchanged
        self._buf_len_at_cache: Dict[int, int] = {}  # FIX: remember buffer length at cache time

    def update_track(self, track_id: int, kpts_norm: Optional[np.ndarray]):
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=SEQ_LEN)
            self._frame_count[track_id] = 0
            self._last_pred[track_id] = (-1, 0.0)
            self._frames_since_pred[track_id] = 0

        if kpts_norm is None:
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
        n_frames = self._frame_count.get(track_id, 0)
        since_pred = self._frames_since_pred.get(track_id, 0)

        if n_frames < self.min_track_frames or since_pred < self.pred_stride:
            lid, conf = self._last_pred.get(track_id, (-1, 0.0))
            return lid, conf, self.label_map.get(lid, "?")

        self._frames_since_pred[track_id] = 0

        buf_len = len(self._buffers[track_id])  # FIX: only recompute features if buffer has new frames since last prediction
        cached_len = self._buf_len_at_cache.get(track_id, -1)  # FIX: read last cached buffer length
        if track_id not in self._feat_cache or buf_len != cached_len:  # FIX: recompute only when needed
            if self.backend == "extratrees":  # FIX: keep existing backend-specific feature extraction path
                if self.fast_mode:  # FIX: preserve fast mode branch
                    x = prepare_sequence_fast(self._buffers[track_id], seq_len=self.fast_seq_len)  # FIX: fast sequence path
                else:  # FIX: preserve standard branch
                    x = prepare_sequence(self._buffers[track_id])  # FIX: standard sequence path
            else:  # FIX: torch backend feature extraction
                x = prepare_sequence(self._buffers[track_id])  # FIX: prepare sequence for torch backend
            if self.feat_mean is not None:  # FIX: keep normalization behavior
                x = (x - self.feat_mean) / self.feat_std  # FIX: normalize once before caching
            self._feat_cache[track_id] = x  # FIX: store computed features
            self._buf_len_at_cache[track_id] = buf_len  # FIX: store buffer length at cache time
        else:  # FIX: reuse cache when buffer unchanged
            x = self._feat_cache[track_id]  # FIX: cached features reused

        if self.backend == "extratrees":  # FIX: continue inference using cached/prepared x
            seq69 = x[0]  # FIX: feature tensor for tree model
            feat = build_extratrees_feature_vector(seq69)[np.newaxis, :]  # FIX: build tabular feature vector
            if hasattr(self.et_model, "predict_proba"):  # FIX: probability-capable branch
                probs = self.et_model.predict_proba(feat)[0]  # FIX: compute class probabilities
                label_id = int(np.argmax(probs))  # FIX: pick top class
                confidence = float(probs[label_id])  # FIX: keep confidence definition
            else:  # FIX: fallback branch when predict_proba unavailable
                label_id = int(self.et_model.predict(feat)[0])  # FIX: direct prediction
                confidence = 1.0  # FIX: fixed confidence fallback
        else:  # FIX: torch backend inference path
            xt = torch.FloatTensor(x).to(self.device)  # FIX: tensor conversion
            logits, _ = self.model(xt)  # FIX: forward pass
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()  # FIX: probability conversion
            label_id = int(np.argmax(probs))  # FIX: pick top class
            confidence = float(probs[label_id])  # FIX: keep confidence behavior

        # Class-specific gating:
        # - Fall: lower threshold a bit to catch rapid transitions.
        # - Sitting_Quickly: require higher confidence to reduce squat->sit confusion.
        class_threshold = self.conf_threshold
        if label_id == 0:  # Fall
            class_threshold = max(0.01, self.conf_threshold - self.fall_conf_boost)
        elif label_id == 2:  # Sitting_Quickly
            class_threshold = min(0.99, self.conf_threshold + self.sitting_conf_penalty)

        if confidence >= class_threshold:
            hist = self._pred_history.setdefault(track_id, [])
            # Fall should react quickly: allow immediate update for high-confidence fall.
            if label_id == 0 and confidence >= max(class_threshold, 0.35):
                hist.clear()
                hist.append(0)
                self._last_pred[track_id] = (0, confidence)
            else:
                hist.append(label_id)
                if len(hist) > self.smooth_window:
                    hist.pop(0)
                smoothed_id = Counter(hist).most_common(1)[0][0]
                self._last_pred[track_id] = (smoothed_id, confidence)

        lid, conf = self._last_pred.get(track_id, (-1, 0.0))
        return lid, conf, self.label_map.get(lid, "?")

    def remove_stale_tracks(self, active_ids: set[int]):
        dead = [tid for tid in self._buffers if tid not in active_ids]
        for tid in dead:
            del self._buffers[tid]
            del self._frame_count[tid]
            del self._last_pred[tid]
            del self._frames_since_pred[tid]
            self._pred_history.pop(tid, None)
            self._feat_cache.pop(tid, None)  # FIX: clean up cache for lost tracks to prevent memory leak
            self._buf_len_at_cache.pop(tid, None)  # FIX: clean up cached buffer length for lost tracks

    def get_last_prediction(self, track_id: int) -> Tuple[int, float, str]:
        lid, conf = self._last_pred.get(track_id, (-1, 0.0))
        return lid, conf, self.label_map.get(lid, "?")


def extract_kpts_for_track(result, track_id: int, w: int, h: int) -> Optional[np.ndarray]:
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

    if idx >= result.keypoints.xy.shape[0]:
        return None

    xy = result.keypoints.xy[idx].cpu().numpy().astype(np.float32)
    norm = xy.copy()
    if w > 0:
        norm[:, 0] /= w
    if h > 0:
        norm[:, 1] /= h
    norm = np.clip(norm, 0.0, 1.0)

    if np.all(norm == 0):
        return None
    return norm


def draw_skeleton(frame: np.ndarray, kpts_norm: np.ndarray, color: Tuple[int, int, int], w: int, h: int):
    pts = (kpts_norm * np.array([w, h])).astype(int)
    valid = ~np.all(kpts_norm == 0, axis=-1)

    for i, j in SKELETON:
        if valid[i] and valid[j]:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, 2, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        if valid[i]:
            cv2.circle(frame, tuple(pt), 4, color, -1, cv2.LINE_AA)


def draw_action_label(frame: np.ndarray, bbox: np.ndarray, track_id: int, label: str, conf: float, color: Tuple[int, int, int]):
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"#{track_id} {label} {conf:.0%}" if conf > 0 else f"#{track_id} ..."
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(frame, text, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def bbox_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.astype(np.float32)
    bx1, by1, bx2, by2 = box_b.astype(np.float32)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 1e-6:
        return 0.0
    return float(inter_area / union)


def bbox_center_distance_norm(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.astype(np.float32)
    bx1, by1, bx2, by2 = box_b.astype(np.float32)

    acx = (ax1 + ax2) * 0.5
    acy = (ay1 + ay2) * 0.5
    bcx = (bx1 + bx2) * 0.5
    bcy = (by1 + by2) * 0.5

    aw = max(ax2 - ax1, 1.0)
    ah = max(ay2 - ay1, 1.0)
    bw = max(bx2 - bx1, 1.0)
    bh = max(by2 - by1, 1.0)
    norm = max((aw + bw) * 0.5, (ah + bh) * 0.5, 1.0)
    dist = float(np.hypot(acx - bcx, acy - bcy))
    return dist / norm


def resolve_tracker_config(name: str) -> str:
    options = {
        "ByteTrack (custom)": ROOT / "bytetrack_custom.yaml",
        "ByteTrack (default)": ROOT / "config" / "bytetrack.yaml",
        "BoT-SORT (custom)": ROOT / "config" / "botsort_custom.yaml",
        "BoT-SORT (default)": ROOT / "config" / "botsort.yaml",
    }
    p = options[name]
    return str(p if p.exists() else (ROOT / "bytetrack_custom.yaml"))


def build_action_recognizer_if_available(
    model_path: str,
    conf_threshold: float,
    pred_stride: int,
    min_track_frames: int,
    smooth_window: int,
    fall_conf_boost: float,
    sitting_conf_penalty: float,
):
    try:
        if not Path(model_path).exists():
            return None, f"Action model not found: {model_path}"
        rec = ActionRecognizerLite(
            model_path=model_path,
            conf_threshold=conf_threshold,
            pred_stride=pred_stride,
            min_track_frames=min_track_frames,
            smooth_window=smooth_window,
            fall_conf_boost=fall_conf_boost,
            sitting_conf_penalty=sitting_conf_penalty,
        )
        return rec, None
    except Exception as e:
        return None, f"Action recognizer disabled: {e}"


def process_stream(
    cap: cv2.VideoCapture,
    output_path: Optional[Path],
    pose_model,
    recognizer,
    tracker_cfg: str,
    det_conf: float,
    det_iou: float,
    imgsz: int,
    max_det: int,
    draw_skeleton_flag: bool,
    max_frames: Optional[int] = None,
    total_frames: Optional[int] = None,
    show_preview: bool = True,
    preview_stride: int = 3,
    display_max_width: int = 960,
    preview_jpeg_quality: int = 88,
    preview_native_resolution: bool = False,
    normalize_timing_across_videos: bool = True,
    target_analysis_fps: float = 12.0,
    process_stride: int = 1,
    output_scale: float = 1.0,
    skip_action_model: bool = False,
    auto_tune_cpu: bool = True,
):
    frame_slot = st.empty()
    metrics_slot = st.empty()
    progress_slot = st.empty()
    progress = progress_slot.progress(0) if total_frames and total_frames > 0 else None

    cpu_only = not torch.cuda.is_available()
    is_upload_job = output_path is not None
    effective_process_stride = max(1, int(process_stride))
    effective_preview_stride = max(1, int(preview_stride))
    effective_max_det = int(max_det)
    effective_preview_jpeg_quality = int(np.clip(preview_jpeg_quality, 60, 95))
    effective_target_analysis_fps = max(0.0, float(target_analysis_fps))
    cpu_auto_tuned = False
    if auto_tune_cpu and cpu_only and effective_process_stride < 2:
        effective_process_stride = 2
        cpu_auto_tuned = True
    if auto_tune_cpu and cpu_only and is_upload_job and effective_process_stride < 3:
        effective_process_stride = 3
        cpu_auto_tuned = True
    if auto_tune_cpu and cpu_only and show_preview and effective_preview_stride < 5:
        effective_preview_stride = 5
        cpu_auto_tuned = True
    if not show_preview and effective_preview_stride < 12:
        effective_preview_stride = 12
    if auto_tune_cpu and cpu_only and effective_max_det > 20:
        effective_max_det = 20
        cpu_auto_tuned = True

    effective_action_pred_stride = None
    effective_action_update_stride = 1
    action_backend = None
    action_fast_mode = None
    if not skip_action_model and recognizer is not None:
        action_backend = getattr(recognizer, "backend", "unknown")
        if action_backend == "extratrees":
            fast_mode_target = bool(auto_tune_cpu and cpu_only and is_upload_job)
            setattr(recognizer, "fast_mode", fast_mode_target)
            action_fast_mode = fast_mode_target
        else:
            action_fast_mode = False
        current_pred_stride = int(getattr(recognizer, "pred_stride", 1))
        if auto_tune_cpu and cpu_only:
            if action_backend == "torch":
                min_pred_stride = 6 if is_upload_job else 4
            else:
                min_pred_stride = 4 if is_upload_job else 2
            if current_pred_stride < min_pred_stride:
                setattr(recognizer, "pred_stride", min_pred_stride)
                current_pred_stride = min_pred_stride
                cpu_auto_tuned = True
        effective_action_pred_stride = current_pred_stride
        if action_backend == "torch":
            effective_action_update_stride = 3 if is_upload_job else 2
        else:
            effective_action_update_stride = 2 if is_upload_job else 1
        if cpu_only and is_upload_job:
            effective_action_update_stride = max(effective_action_update_stride, 3)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if normalize_timing_across_videos and effective_target_analysis_fps > 0 and fps_src > effective_target_analysis_fps:
        effective_process_stride = max(1, max(effective_process_stride, int(round(fps_src / effective_target_analysis_fps))))
    if auto_tune_cpu and cpu_only and is_upload_job and fps_src >= 24.0 and effective_process_stride < 4:
        effective_process_stride = 4
        cpu_auto_tuned = True
    if normalize_timing_across_videos and fps_src > 0:
        target_action_fps = 8.0 if action_backend == "torch" else 10.0
        effective_action_update_stride = max(1, max(effective_action_update_stride, int(round(fps_src / target_action_fps))))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    
    # Apply output scaling if specified
    output_w = max(1, int(w * output_scale))
    output_h = max(1, int(h * output_scale))
    
    writer = None
    written_output_path = output_path
    if output_path is not None and w > 0 and h > 0:
        preferred_mp4 = output_path if output_path.suffix.lower() == ".mp4" else output_path.with_suffix(".mp4")
        fallback_avi = preferred_mp4.with_suffix(".avi")

        writer_specs = [
            (preferred_mp4, ("avc1", "mp4v")),
            (fallback_avi, ("XVID", "MJPG")),
        ]
        writer_ready = False
        for candidate_path, codecs in writer_specs:
            for codec_name in codecs:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                candidate_writer = cv2.VideoWriter(str(candidate_path), fourcc, fps_src, (output_w, output_h))
                if candidate_writer.isOpened():
                    writer = candidate_writer
                    written_output_path = candidate_path
                    writer_ready = True
                    break
                candidate_writer.release()
            if writer_ready:
                break

        if not writer_ready:
            writer = None
            written_output_path = None
            st.warning("Cannot initialize video writer. Processing will run but output video may be unavailable.")
    # FIX: only draw on frames that will actually be shown or written this iteration
    # FIX: We will handle per-frame drawing decision inside the loop instead
    draw_overlays = (writer is not None) or show_preview  # FIX: retain global capability flag

    action_counts = defaultdict(int)
    t_prev = time.time()
    t_start = t_prev
    fps_ema = fps_src if fps_src > 0 else 30.0
    frame_idx = 0
    detection_stats = {"frames_processed": 0, "frames_with_detections": 0, "frames_with_tracks": 0}
    last_active_ids = set()
    frames_with_people_all = 0

    raw_to_stable_id: Dict[int, int] = {}
    raw_last_seen: Dict[int, int] = {}
    stable_last_bbox: Dict[int, np.ndarray] = {}
    stable_last_seen: Dict[int, int] = {}
    unique_stable_ids: set[int] = set()
    next_stable_id = 1
    max_id_idle_frames = max(90, int(round(fps_src * 6.0)))
    track_hold_frames = max(3, int(round(fps_src * 0.35)))
    stable_reid_gap = max(8, int(round(fps_src * 0.75)))
    stable_reid_iou = 0.20
    stable_reid_dist = 0.75
    overlay_cache_by_id: Dict[int, Tuple[np.ndarray, str, float, Tuple[int, int, int]]] = {}

    def resolve_display_id(raw_tid: int, bbox: np.ndarray, assigned_stable_ids: set[int]) -> int:
        nonlocal next_stable_id
        sid = raw_to_stable_id.get(raw_tid)
        if sid is not None:
            return sid

        best_sid = None
        best_score = -1.0
        for candidate_sid, candidate_bbox in stable_last_bbox.items():
            if candidate_sid in assigned_stable_ids:
                continue
            last_seen = stable_last_seen.get(candidate_sid, -10_000)
            if frame_idx - last_seen > stable_reid_gap:
                continue
            iou = bbox_iou_xyxy(bbox, candidate_bbox)
            dist = bbox_center_distance_norm(bbox, candidate_bbox)
            if iou < stable_reid_iou and dist > stable_reid_dist:
                continue
            score = iou - 0.35 * dist
            if score > best_score:
                best_score = score
                best_sid = candidate_sid

        if best_sid is None:
            sid = next_stable_id
            next_stable_id += 1
        else:
            sid = best_sid

        raw_to_stable_id[raw_tid] = sid
        return sid

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        detection_stats["frames_processed"] += 1
        visible_ids = set()
        recognizer_active_ids = set()
        should_draw_this_frame = draw_overlays and (
            writer is not None or (show_preview and frame_idx % effective_preview_stride == 0)
        )
        should_process_frame = (frame_idx % effective_process_stride == 0)
        action_update_due = (frame_idx % effective_action_update_stride == 0)

        if should_process_frame:
            assigned_stable_ids: set[int] = set()
            results = pose_model.track(
                frame,
                persist=True,
                tracker=tracker_cfg,
                conf=det_conf,
                iou=det_iou,
                imgsz=imgsz,
                max_det=effective_max_det,
                classes=[0],
                half=torch.cuda.is_available(),
                verbose=False,
            )
            result = results[0] if results else None
            
            if result is not None and result.boxes is not None and result.boxes.id is not None:
                detection_stats["frames_with_detections"] += 1
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                bboxes = result.boxes.xyxy.cpu().numpy()

                for i, tid in enumerate(track_ids):
                    raw_tid = int(tid)
                    detection_stats["frames_with_tracks"] += 1
                    bbox = bboxes[i]
                    display_tid = resolve_display_id(raw_tid, bbox, assigned_stable_ids)
                    assigned_stable_ids.add(display_tid)
                    unique_stable_ids.add(display_tid)

                    raw_last_seen[raw_tid] = frame_idx
                    stable_last_bbox[display_tid] = bbox.copy()
                    stable_last_seen[display_tid] = frame_idx
                    visible_ids.add(display_tid)

                    needs_action_update = (
                        not skip_action_model
                        and recognizer is not None
                        and (action_update_due or display_tid not in recognizer._buffers)
                    )
                    needs_kpts = draw_skeleton_flag or needs_action_update
                    kpts = extract_kpts_for_track(result, raw_tid, frame.shape[1], frame.shape[0]) if needs_kpts else None
                    if not skip_action_model and recognizer is not None:
                        if needs_action_update:
                            recognizer.update_track(display_tid, kpts)
                            label_id, conf_val, label_name = recognizer.predict(display_tid)
                        else:
                            label_id, conf_val, label_name = recognizer.get_last_prediction(display_tid)
                    else:
                        # Fallback heuristic only when action model unavailable or disabled.
                        aspect = (bbox[2] - bbox[0]) / max((bbox[3] - bbox[1]), 1e-6)
                        label_id = 0 if aspect > 1.2 else 1
                        conf_val = 0.50
                        label_name = LABEL_MAP.get(label_id, "Walking")

                    if label_name not in ("?", "unknown"):
                        action_counts[label_name] += 1

                    color = LABEL_COLORS.get(label_id, (200, 200, 200))
                    overlay_cache_by_id[display_tid] = (bbox.copy(), label_name, conf_val, color)
                    if should_draw_this_frame:
                        if draw_skeleton_flag and kpts is not None:
                            draw_skeleton(frame, kpts, color, frame.shape[1], frame.shape[0])
                        draw_action_label(frame, bbox, display_tid, label_name, conf_val, color)

            stale_raw_ids = [rid for rid, last_seen in raw_last_seen.items() if frame_idx - last_seen > max_id_idle_frames]
            for rid in stale_raw_ids:
                raw_last_seen.pop(rid, None)
                raw_to_stable_id.pop(rid, None)

            stale_stable_ids = [sid for sid, last_seen in stable_last_seen.items() if frame_idx - last_seen > max_id_idle_frames]
            for sid in stale_stable_ids:
                stable_last_seen.pop(sid, None)
                stable_last_bbox.pop(sid, None)
                overlay_cache_by_id.pop(sid, None)

            held_ids = {
                sid for sid, last_seen in stable_last_seen.items()
                if frame_idx - last_seen <= track_hold_frames
            }
            if should_draw_this_frame:
                for held_tid in sorted(held_ids - visible_ids):
                    held_overlay = overlay_cache_by_id.get(held_tid)
                    if held_overlay is None:
                        continue
                    held_bbox, held_label, held_conf, held_color = held_overlay
                    draw_action_label(frame, held_bbox, held_tid, held_label, held_conf, held_color)
            recognizer_active_ids = held_ids.copy()
            last_active_ids = held_ids.copy()
        else:
            recognizer_active_ids = {
                sid for sid, last_seen in stable_last_seen.items()
                if frame_idx - last_seen <= track_hold_frames
            }
            if not recognizer_active_ids:
                recognizer_active_ids = last_active_ids.copy()
            if should_draw_this_frame:
                for cached_tid in sorted(recognizer_active_ids):
                    cached_overlay = overlay_cache_by_id.get(cached_tid)
                    if cached_overlay is None:
                        continue
                    cached_bbox, cached_label, cached_conf, cached_color = cached_overlay
                    draw_action_label(frame, cached_bbox, cached_tid, cached_label, cached_conf, cached_color)

        if recognizer_active_ids:
            frames_with_people_all += 1

        if not skip_action_model and recognizer is not None:
            recognizer.remove_stale_tracks(recognizer_active_ids)

        # Scale frame for output if needed
        if writer is not None:
            output_frame = frame
            if output_scale != 1.0:
                output_frame = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA)
            writer.write(output_frame)

        now = time.time()
        fps_ema = 0.92 * fps_ema + 0.08 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        if frame_idx % effective_preview_stride == 0:
            metrics_slot.info(
                f"Frame: {frame_idx} | FPS: {fps_ema:.1f} | Tracks: {len(visible_ids) if should_process_frame else len(recognizer_active_ids)} | "
                f"Falls: {action_counts.get('Fall', 0)} | Stride: {effective_process_stride} | MaxDet: {effective_max_det}"
            )

            if progress is not None:
                pct = min(100, int((frame_idx + 1) * 100 / max(total_frames, 1)))
                progress.progress(pct)

            if show_preview:
                preview_frame = frame
                if not preview_native_resolution and display_max_width > 0 and frame.shape[1] > display_max_width:
                    scale = display_max_width / float(frame.shape[1])
                    preview_h = max(1, int(frame.shape[0] * scale))
                    preview_frame = cv2.resize(frame, (display_max_width, preview_h), interpolation=cv2.INTER_AREA)
                _, jpeg_buf = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, effective_preview_jpeg_quality])
                frame_slot.image(jpeg_buf.tobytes(), width=preview_frame.shape[1])

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if progress is not None:
        progress.progress(100)

    elapsed = max(time.time() - t_start, 1e-6)
    fps_avg = frame_idx / elapsed if frame_idx > 0 else 0.0

    return {
        "total_frames": detection_stats["frames_processed"],
        "frames_with_detections": frames_with_people_all,
        "frames_with_detections_processed": detection_stats["frames_with_detections"],
        "total_tracks": len(unique_stable_ids),
        "total_detections": detection_stats["frames_with_tracks"],
        "fps": fps_avg,
        "fps_live_ema": fps_ema,
        "elapsed_sec": elapsed,
        "source_fps": fps_src,
        "effective_target_analysis_fps": effective_target_analysis_fps,
        "effective_process_stride": effective_process_stride,
        "effective_preview_stride": effective_preview_stride,
        "effective_max_det": effective_max_det,
        "effective_action_pred_stride": effective_action_pred_stride,
        "effective_action_update_stride": effective_action_update_stride,
        "action_backend": action_backend,
        "action_fast_mode": action_fast_mode,
        "cpu_auto_tuned": cpu_auto_tuned,
        "action_counts": dict(action_counts),
        "output_path": str(written_output_path) if written_output_path else None,
    }


def main():
    st.set_page_config(page_title="Fall Detection - Streamlit", layout="wide")
    st.title("Fall Detection & Action Recognition (Single-file Streamlit)")
    st.caption("No React/FastAPI/Spring data path. Frames are processed directly in Python.")

    if "last_output_video" not in st.session_state:
        st.session_state.last_output_video = None
    if "last_summary" not in st.session_state:
        st.session_state.last_summary = None

    with st.sidebar:
        st.header("Settings")

        gpu_available = torch.cuda.is_available()
        preset_profile_version = 3
        if st.session_state.get("preset_profile_version") != preset_profile_version:
            st.session_state.preset_imgsz = 640
            st.session_state.preset_process_stride = 1
            st.session_state.preset_output_scale = 1.0 if gpu_available else 0.75
            st.session_state.preset_skip_action = False
            st.session_state.preset_preview_stride = 6
            st.session_state.preset_display_max_width = 640
            st.session_state.preset_preview_jpeg_quality = 88
            st.session_state.preset_preview_native_resolution = False
            st.session_state.preset_normalize_timing = True
            st.session_state.preset_target_analysis_fps = 12.0
            st.session_state.preset_pred_stride = 3
            st.session_state.preset_max_det = 12
            st.session_state.preset_webcam_preview = bool(gpu_available)
            st.session_state.preset_cpu_auto_tune = True
            st.session_state.preset_profile_version = preset_profile_version

        # Initialize preset defaults (CPU-safe defaults when CUDA is unavailable).
        if "preset_imgsz" not in st.session_state:
            st.session_state.preset_imgsz = 640  # FIX: lock default image size to 640
        if "preset_process_stride" not in st.session_state:
            st.session_state.preset_process_stride = 1  # FIX: default quality mode for stable tracking
        if "preset_output_scale" not in st.session_state:
            st.session_state.preset_output_scale = 1.0 if gpu_available else 0.75
        if "preset_skip_action" not in st.session_state:
            st.session_state.preset_skip_action = False
        if "preset_preview_stride" not in st.session_state:
            st.session_state.preset_preview_stride = 6  # FIX: update preview every 6 frames by default
        if "preset_display_max_width" not in st.session_state:
            st.session_state.preset_display_max_width = 640  # FIX: smaller preview width for faster encoding
        if "preset_preview_jpeg_quality" not in st.session_state:
            st.session_state.preset_preview_jpeg_quality = 88
        if "preset_preview_native_resolution" not in st.session_state:
            st.session_state.preset_preview_native_resolution = False
        if "preset_normalize_timing" not in st.session_state:
            st.session_state.preset_normalize_timing = True
        if "preset_target_analysis_fps" not in st.session_state:
            st.session_state.preset_target_analysis_fps = 12.0
        if "preset_pred_stride" not in st.session_state:
            st.session_state.preset_pred_stride = 3  # FIX: action prediction every 3rd frame by default
        if "preset_max_det" not in st.session_state:
            st.session_state.preset_max_det = 12  # FIX: practical default for small-group scenes (about 4 people)
        if "preset_webcam_preview" not in st.session_state:
            st.session_state.preset_webcam_preview = bool(gpu_available)
        if "preset_cpu_auto_tune" not in st.session_state:
            st.session_state.preset_cpu_auto_tune = True

        # Get preset values
        default_imgsz = st.session_state.preset_imgsz
        default_process_stride = st.session_state.preset_process_stride
        default_output_scale = st.session_state.preset_output_scale
        default_skip_action = st.session_state.preset_skip_action
        default_preview_stride = st.session_state.preset_preview_stride
        default_display_max_width = st.session_state.preset_display_max_width
        default_preview_jpeg_quality = st.session_state.preset_preview_jpeg_quality
        default_preview_native_resolution = st.session_state.preset_preview_native_resolution
        default_normalize_timing = st.session_state.preset_normalize_timing
        default_target_analysis_fps = st.session_state.preset_target_analysis_fps
        default_pred_stride = st.session_state.preset_pred_stride
        default_max_det = st.session_state.preset_max_det
        default_webcam_preview = st.session_state.preset_webcam_preview
        default_cpu_auto_tune = st.session_state.preset_cpu_auto_tune

        source = st.radio("Input Source", ["Upload Video", "Webcam"])

        pose_weights = st.text_input("YOLO Pose Weights", value=str(ROOT / "yolov8n-pose.pt"))
        tracker_name = st.selectbox(
            "Tracker",
            ["ByteTrack (custom)", "ByteTrack (default)", "BoT-SORT (custom)", "BoT-SORT (default)"],
            index=2,  # FIX: default to BoT-SORT custom for better ID stability
        )
        det_conf = st.slider("Detection Confidence", 0.05, 0.95, 0.30, 0.01)  # FIX: tighter default reduces noisy detections
        det_iou = st.slider("NMS IoU", 0.10, 0.95, 0.50, 0.01)  # FIX: slightly higher IoU for stable association
        imgsz = st.select_slider("Image Size", [320, 480, 640, 960, 1280], value=default_imgsz)
        max_det = st.slider("Max Persons per Frame", 1, 200, default_max_det, 1)
        draw_skeleton_flag = st.checkbox("Draw Skeleton", value=False)

        st.markdown("---")
        st.subheader("Performance")
        st.info(f"💻 {'✅ GPU Available (CUDA)' if gpu_available else '❌ GPU Not Available - Using CPU'}")
        
        # Recommended settings info
        with st.expander("📋 RTX 3050 Recommended Settings", expanded=False):
            st.markdown("""
**RTX 3050 (4GB VRAM) Balanced Profile:**
- Tracker: **BoT-SORT (custom)** (more stable IDs in multi-person scenes)
- Image Size: **640** (good balance)
- Process every Nth frame: **1** (detect every frame)
- Output scale: **1.0** (full quality output)
- Action labels: **Enabled** (lightweight model)

**Expected Performance:**
- ~15-25 FPS on typical videos
- Smooth detections, good accuracy
- Safe GPU memory usage (<2GB)

Click **\"🎮 RTX 3050 (Balanced)\"** to apply these settings automatically!
            """)

        live_preview_upload = st.checkbox(  # FIX: default OFF — live preview during upload reduces FPS by 30–50%
            "Live preview while processing upload (⚠️ reduces FPS ~30–50%)",  # FIX:
            value=False,  # FIX:
        )  # FIX:
        webcam_live_preview = st.checkbox("Live preview while processing webcam", value=default_webcam_preview)
        preview_stride = st.slider("Preview update every N frames", 1, 15, default_preview_stride, 1)  # FIX: keep preset-driven value, default initialized to 6
        display_max_width = st.select_slider(
            "Preview max width",
            [480, 640, 854, 960, 1280],  # FIX: prioritize smaller preview widths for speed
            value=default_display_max_width,  # FIX: keep preset-driven value, default initialized to 640
        )
        preview_jpeg_quality = st.slider("Preview JPEG quality", 60, 95, default_preview_jpeg_quality, 1)
        preview_native_resolution = st.checkbox(
            "Preview at native resolution (slower)",
            value=default_preview_native_resolution,
        )
        normalize_timing_across_videos = st.checkbox(
            "Normalize timing across videos",
            value=default_normalize_timing,
            help="Use a target analysis FPS so videos with different source FPS behave more consistently.",
        )
        target_analysis_fps = st.slider(
            "Target analysis FPS",
            6.0,
            20.0,
            float(default_target_analysis_fps),
            1.0,
            disabled=not normalize_timing_across_videos,
        )

        st.divider()
        st.subheader("Speed Tuning")

        # GPU Profile Presets
        st.markdown("**⚡ GPU Profiles**")
        col_preset1, col_preset2, col_preset3 = st.columns(3)

        with col_preset1:
            if st.button("🎮 RTX 3050 (Balanced)", use_container_width=True):
                st.session_state.preset_imgsz = 640
                st.session_state.preset_process_stride = 1
                st.session_state.preset_output_scale = 1.0
                st.session_state.preset_skip_action = False
                st.session_state.preset_preview_stride = 6
                st.session_state.preset_display_max_width = 640
                st.session_state.preset_preview_jpeg_quality = 88
                st.session_state.preset_preview_native_resolution = False
                st.session_state.preset_normalize_timing = True
                st.session_state.preset_target_analysis_fps = 12.0
                st.session_state.preset_pred_stride = 3
                st.session_state.preset_max_det = 12
                st.session_state.preset_webcam_preview = True
                st.session_state.preset_cpu_auto_tune = True
                st.success("RTX 3050 preset applied!")
                st.rerun()

        with col_preset2:
            if st.button("⚡ Fast Mode (High Speed)", use_container_width=True):
                st.session_state.preset_imgsz = 480
                st.session_state.preset_process_stride = 2
                st.session_state.preset_output_scale = 0.5
                st.session_state.preset_skip_action = False
                st.session_state.preset_preview_stride = 5
                st.session_state.preset_display_max_width = 854
                st.session_state.preset_preview_jpeg_quality = 75
                st.session_state.preset_preview_native_resolution = False
                st.session_state.preset_normalize_timing = True
                st.session_state.preset_target_analysis_fps = 10.0
                st.session_state.preset_pred_stride = 4
                st.session_state.preset_max_det = 25
                st.session_state.preset_webcam_preview = False
                st.session_state.preset_cpu_auto_tune = True
                st.success("Fast mode preset applied!")
                st.rerun()

        with col_preset3:
            if st.button("🎯 Quality Mode (High Quality)", use_container_width=True):
                st.session_state.preset_imgsz = 960
                st.session_state.preset_process_stride = 1
                st.session_state.preset_output_scale = 1.0
                st.session_state.preset_skip_action = False
                st.session_state.preset_preview_stride = 2
                st.session_state.preset_display_max_width = 1280
                st.session_state.preset_preview_jpeg_quality = 92
                st.session_state.preset_preview_native_resolution = True
                st.session_state.preset_normalize_timing = False
                st.session_state.preset_target_analysis_fps = 12.0
                st.session_state.preset_pred_stride = 1
                st.session_state.preset_max_det = 100
                st.session_state.preset_webcam_preview = True
                st.session_state.preset_cpu_auto_tune = True
                st.success("Quality mode preset applied!")
                st.rerun()

        st.divider()
        st.markdown("**Manual Tuning**")
        process_stride = st.slider(
            "Process every Nth frame (skip frames for speed)",
            min_value=1,
            max_value=5,
            value=default_process_stride,
            help="Process only every Nth frame for detection. Output video will still have all frames, but detections are every Nth frame."
        )
        output_scale = st.select_slider(
            "Output resolution scale",
            [0.5, 0.75, 1.0],
            value=default_output_scale,
            help="Reduce output video resolution to increase speed (0.75 = 75% of original size)"
        )
        skip_action_model = st.checkbox(
            "Skip action recognition model (only detect, no labels)",
            value=default_skip_action,
            help="Disable action model to focus on detection speed"
        )
        cpu_auto_tune = st.checkbox(
            "CPU auto-tune (recommended)",
            value=default_cpu_auto_tune,
            help="When enabled on CPU, runtime enforces safer speed settings for smoother FPS.",
        )
        if not gpu_available and process_stride == 1 and cpu_auto_tune:
            st.caption("CPU mode detected: runtime may auto-raise effective process stride (up to 4 for upload) for smoother FPS.")

        st.markdown("---")
        st.subheader("Action Model")
        action_ckpt = st.text_input(
            "Action Model (.pth or .joblib)",
            value=resolve_default_action_model_path(),
        )
        min_track_frames = st.slider("Min Track Frames", 1, 128, 12, 1)
        pred_stride = st.slider("Prediction Stride", 1, 16, default_pred_stride, 1)
        action_conf = st.slider("Action Confidence Threshold", 0.01, 0.95, 0.30, 0.01)  # FIX: default stricter threshold
        smooth_window = st.slider("Smoothing Window", 1, 7, 3, 1)
        fall_conf_boost = st.slider(
            "Fast Fall Sensitivity",
            0.00,
            0.30,
            0.10,
            0.01,
            help="Higher value lowers threshold for Fall class to catch quick falls earlier.",
        )
        sitting_conf_penalty = st.slider(
            "Sitting Strictness",
            0.00,
            0.40,
            0.20,
            0.01,
            help="Higher value requires more confidence for Sitting_Quickly to reduce squat false positives.",
        )

    pose_weights_path = Path(pose_weights)
    if not pose_weights_path.exists():
        st.error(f"YOLO weights not found: {pose_weights_path}")
        st.stop()

    tracker_cfg = resolve_tracker_config(tracker_name)
    pose_model = load_pose_model(str(pose_weights_path))

    recognizer, rec_error = build_action_recognizer_if_available(
        model_path=action_ckpt,
        conf_threshold=action_conf,
        pred_stride=pred_stride,
        min_track_frames=min_track_frames,
        smooth_window=smooth_window,
        fall_conf_boost=fall_conf_boost,
        sitting_conf_penalty=sitting_conf_penalty,
    )
    if rec_error:
        st.warning(rec_error)

    if source == "Upload Video":
        file = st.file_uploader("Upload input video", type=["mp4", "avi", "mov", "mkv", "webm"])
        if file is not None:
            st.video(file)

        run_btn = st.button("Run Inference", type="primary", width="stretch")
        if run_btn:
            if not file:
                st.warning("Please upload a video first.")
                st.stop()

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix or ".mp4") as tmp_in:
                tmp_in.write(file.read())
                input_path = Path(tmp_in.name)

            stem = Path(file.name).stem if file.name else "upload"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"{stem}_{ts}_annotated.mp4"
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                st.error("Cannot open uploaded video.")
                st.stop()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            st.info("Processing uploaded video...")
            summary = process_stream(
                cap=cap,
                output_path=output_path,
                pose_model=pose_model,
                recognizer=recognizer,
                tracker_cfg=tracker_cfg,
                det_conf=det_conf,
                det_iou=det_iou,
                imgsz=imgsz,
                max_det=max_det,
                draw_skeleton_flag=draw_skeleton_flag,
                total_frames=total_frames,
                show_preview=live_preview_upload,
                preview_stride=preview_stride,
                display_max_width=display_max_width,
                preview_jpeg_quality=preview_jpeg_quality,
                preview_native_resolution=preview_native_resolution,
                normalize_timing_across_videos=normalize_timing_across_videos,
                target_analysis_fps=target_analysis_fps,
                process_stride=process_stride,
                output_scale=output_scale,
                skip_action_model=skip_action_model,
                auto_tune_cpu=cpu_auto_tune,
            )

            st.session_state.last_output_video = str(output_path)
            st.session_state.last_summary = summary

            st.success("Done")
            if summary.get("cpu_auto_tuned"):
                st.info(
                    "CPU auto-tuning active: "
                    f"effective_process_stride={summary.get('effective_process_stride')} | "
                    f"effective_preview_stride={summary.get('effective_preview_stride')} | "
                    f"effective_max_det={summary.get('effective_max_det')} | "
                    f"effective_action_pred_stride={summary.get('effective_action_pred_stride')} | "
                    f"effective_action_update_stride={summary.get('effective_action_update_stride')} | "
                    f"action_fast_mode={summary.get('action_fast_mode')}"
                )
            
            # Display detection diagnostics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Frames", summary.get("total_frames", 0))
            with col2:
                st.metric("Frames w/ People", summary.get("frames_with_detections", 0))
            with col3:
                st.metric("Unique Track IDs", summary.get("total_tracks", 0))
            with col4:
                st.metric("FPS", f"{summary.get('fps', 0):.1f}")
            st.caption(
                f"Detections: {summary.get('total_detections', 0)} | "
                f"Processed frames w/ detections: {summary.get('frames_with_detections_processed', 0)} | "
                f"Source FPS: {summary.get('source_fps', 0):.1f} | "
                f"Target analysis FPS: {summary.get('effective_target_analysis_fps', 0):.1f}"
            )
            
            if summary.get("frames_with_detections", 0) == 0:
                st.warning("⚠️ No people detected in video. Check detection confidence settings or video quality.")
            
            if summary.get("action_counts"):
                st.subheader("Actions Detected")
                for action, count in sorted(summary["action_counts"].items(), key=lambda x: x[1], reverse=True):
                    st.write(f"• {action}: **{count}** times")
            
            if output_path.exists():
                st.subheader("Annotated Video")
                st.video(str(output_path))
                with output_path.open("rb") as f:
                    st.download_button(
                        "Download annotated video",
                        data=f.read(),
                        file_name="video_action_streamlit.mp4",
                        mime="video/mp4",
                        width="stretch",
                    )

        saved_output = st.session_state.get("last_output_video")
        saved_summary = st.session_state.get("last_summary")
        if saved_output and Path(saved_output).exists():
            st.markdown("### Last Processed Video")
            if saved_summary:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Frames", saved_summary.get("total_frames", 0))
                with col2:
                    st.metric("Frames w/ People", saved_summary.get("frames_with_detections", 0))
                with col3:
                    st.metric("Unique Track IDs", saved_summary.get("total_tracks", 0))
                with col4:
                    st.metric("FPS", f"{saved_summary.get('fps', 0):.1f}")
                st.caption(
                    f"Detections: {saved_summary.get('total_detections', 0)} | "
                    f"Processed frames w/ detections: {saved_summary.get('frames_with_detections_processed', 0)} | "
                    f"Source FPS: {saved_summary.get('source_fps', 0):.1f} | "
                    f"Target analysis FPS: {saved_summary.get('effective_target_analysis_fps', 0):.1f}"
                )
            st.video(saved_output)
            if saved_summary and saved_summary.get("action_counts"):
                st.subheader("Actions Detected")
                for action, count in sorted(saved_summary["action_counts"].items(), key=lambda x: x[1], reverse=True):
                    st.write(f"• {action}: **{count}** times")
            with Path(saved_output).open("rb") as f:
                st.download_button(
                    "Download last processed video",
                    data=f.read(),
                    file_name=Path(saved_output).name,
                    mime="video/mp4",
                    width="stretch",
                )
    else:
        duration_sec = st.slider("Webcam capture duration (sec)", 3, 60, 10, 1)
        cam_idx = st.number_input("Webcam index", min_value=0, max_value=4, value=0, step=1)
        run_cam = st.button("Start Webcam", type="primary", width="stretch")
        if run_cam:
            cap = cv2.VideoCapture(int(cam_idx))
            if not cap.isOpened():
                st.error(f"Cannot open webcam index {cam_idx}.")
                st.stop()

            approx_fps = cap.get(cv2.CAP_PROP_FPS)
            if approx_fps is None or approx_fps <= 1:
                approx_fps = 25.0
            max_frames = int(duration_sec * approx_fps)

            st.info(f"Processing webcam for {duration_sec}s (~{max_frames} frames)...")
            summary = process_stream(
                cap=cap,
                output_path=None,
                pose_model=pose_model,
                recognizer=recognizer,
                tracker_cfg=tracker_cfg,
                det_conf=det_conf,
                det_iou=det_iou,
                imgsz=imgsz,
                max_det=max_det,
                draw_skeleton_flag=draw_skeleton_flag,
                max_frames=max_frames,
                total_frames=max_frames,
                show_preview=webcam_live_preview,
                preview_stride=preview_stride,
                display_max_width=display_max_width,
                preview_jpeg_quality=preview_jpeg_quality,
                preview_native_resolution=preview_native_resolution,
                normalize_timing_across_videos=normalize_timing_across_videos,
                target_analysis_fps=target_analysis_fps,
                process_stride=process_stride,
                output_scale=output_scale,
                skip_action_model=skip_action_model,
                auto_tune_cpu=cpu_auto_tune,
            )
            st.success("Webcam run finished")
            if summary.get("cpu_auto_tuned"):
                st.info(
                    "CPU auto-tuning active: "
                    f"effective_process_stride={summary.get('effective_process_stride')} | "
                    f"effective_preview_stride={summary.get('effective_preview_stride')} | "
                    f"effective_max_det={summary.get('effective_max_det')} | "
                    f"effective_action_pred_stride={summary.get('effective_action_pred_stride')} | "
                    f"effective_action_update_stride={summary.get('effective_action_update_stride')} | "
                    f"action_fast_mode={summary.get('action_fast_mode')}"
                )
            st.json(summary)


if __name__ == "__main__":
    main()
