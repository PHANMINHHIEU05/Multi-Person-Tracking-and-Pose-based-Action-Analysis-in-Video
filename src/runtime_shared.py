from __future__ import annotations

from collections import Counter, deque
import importlib.util
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

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


def resolve_default_pose_weights_path() -> str:
    engine_path = ROOT / "yolov8n-pose.engine"
    if engine_path.exists() and importlib.util.find_spec("tensorrt") is not None:
        return str(engine_path)
    return str(ROOT / "yolov8n-pose.pt")


def describe_pose_runtime(weights_path: str) -> Dict[str, str]:
    suffix = Path(weights_path).suffix.lower()
    if suffix == ".engine":
        backend = "TensorRT"
        if importlib.util.find_spec("tensorrt") is None:
            device = "missing-tensorrt"
        else:
            device = "cuda" if torch.cuda.is_available() else "missing-cuda"
    else:
        backend = "PyTorch"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "backend": backend,
        "device": device,
        "weights_path": str(weights_path),
    }


def load_pose_model(weights_path: str) -> YOLO:
    runtime_info = describe_pose_runtime(weights_path)
    if runtime_info["backend"] == "TensorRT" and runtime_info["device"] == "missing-tensorrt":
        raise RuntimeError(
            "TensorRT pose engine selected but the current Python environment does not have TensorRT bindings. "
            "Run the PyQt6 app from the dedicated TensorRT environment or switch back to the `.pt` pose model."
        )
    if runtime_info["backend"] == "TensorRT" and runtime_info["device"] != "cuda":
        raise RuntimeError("TensorRT pose engine requires CUDA. Please use a `.pt` pose model or enable NVIDIA CUDA/TensorRT.")

    model = YOLO(weights_path)
    if runtime_info["backend"] == "PyTorch" and runtime_info["device"] == "cuda":
        model.to("cuda")
    warmup = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(warmup, verbose=False)
    model._codex_runtime_info = runtime_info
    return model


def load_action_model_class():
    from train_professional_v3 import ActionRecognitionModel

    return ActionRecognitionModel


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


def prepare_sequence_fast(buffer: deque, seq_len: int = 96) -> np.ndarray:
    frames = list(buffer)[-seq_len:]
    if len(frames) < seq_len:
        first = frames[0]
        frames = [first.copy()] * (seq_len - len(frames)) + frames
    seq = np.stack(frames, axis=0).astype(np.float32)

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


def _valid_keypoint_mask(kpts_norm: np.ndarray) -> np.ndarray:
    return ~np.all(kpts_norm == 0, axis=1)


def _estimate_bbox_norm(
    kpts_norm: Optional[np.ndarray],
    bbox_norm: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], float, float]:
    if bbox_norm is not None:
        bbox = bbox_norm.astype(np.float32, copy=True)
        return bbox, max(float(bbox[2] - bbox[0]), 1e-3), max(float(bbox[3] - bbox[1]), 1e-3)
    if kpts_norm is None:
        return None, 1e-3, 1e-3
    valid = _valid_keypoint_mask(kpts_norm)
    if valid.sum() < 2:
        return None, 1e-3, 1e-3
    valid_points = kpts_norm[valid]
    x1 = float(valid_points[:, 0].min())
    y1 = float(valid_points[:, 1].min())
    x2 = float(valid_points[:, 0].max())
    y2 = float(valid_points[:, 1].max())
    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
    return bbox, max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)


def _estimate_body_center(kpts_norm: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if kpts_norm is None:
        return None
    valid = _valid_keypoint_mask(kpts_norm)
    if valid[L_HIP] and valid[R_HIP]:
        return ((kpts_norm[L_HIP] + kpts_norm[R_HIP]) * 0.5).astype(np.float32)
    if valid[L_SHOULDER] and valid[R_SHOULDER]:
        return ((kpts_norm[L_SHOULDER] + kpts_norm[R_SHOULDER]) * 0.5).astype(np.float32)
    if valid.any():
        return np.mean(kpts_norm[valid], axis=0).astype(np.float32)
    return None


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
        min_keypoint_ratio: float = 0.70,
        max_keypoint_jitter_ratio: float = 0.15,
        fall_priority_prob: float = 0.40,
        fall_velocity_ratio: float = 0.12,
        sitting_hold_frames: int = 5,
        sitting_height_ratio: float = 0.85,
        sitting_area_ratio: float = 0.78,
        track_time_budget_ms: float = 10.0,
        fast_track_threshold: int = 5,
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
            model_class = load_action_model_class()
            self.model = model_class(
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
        self._feat_cache: Dict[int, np.ndarray] = {}
        self._buf_len_at_cache: Dict[int, int] = {}
        self._quality_state: Dict[int, Dict[str, float | bool]] = {}
        self._last_valid_kpts: Dict[int, np.ndarray] = {}
        self._last_bbox_norm: Dict[int, np.ndarray] = {}
        self._pending_sitting_until: Dict[int, int] = {}
        self._last_predict_ms: Dict[int, float] = {}
        self._fast_mode_overload = False
        self._fast_mode_used = False

        self.min_keypoint_ratio = float(np.clip(min_keypoint_ratio, 0.10, 1.0))
        self.max_keypoint_jitter_ratio = float(max(max_keypoint_jitter_ratio, 0.01))
        self.fall_priority_prob = float(np.clip(fall_priority_prob, 0.05, 0.99))
        self.fall_velocity_ratio = float(max(fall_velocity_ratio, 0.01))
        self.sitting_hold_frames = max(1, int(sitting_hold_frames))
        self.sitting_height_ratio = float(np.clip(sitting_height_ratio, 0.30, 1.20))
        self.sitting_area_ratio = float(np.clip(sitting_area_ratio, 0.20, 1.20))
        self.fast_track_threshold = max(1, int(fast_track_threshold))
        self.track_time_budget_ms = float(max(track_time_budget_ms, 1.0))

    def set_active_track_count(self, count: int):
        self._fast_mode_overload = count > self.fast_track_threshold
        self._fast_mode_used = self._fast_mode_used or self._fast_mode_overload

    def is_fast_mode_active(self) -> bool:
        return self.fast_mode or self._fast_mode_overload

    def _build_quality_state(
        self,
        track_id: int,
        kpts_norm: Optional[np.ndarray],
        bbox_norm: Optional[np.ndarray],
    ) -> Tuple[Dict[str, float | bool], Optional[np.ndarray]]:
        quality: Dict[str, float | bool] = {
            "valid_ratio": 0.0,
            "jitter_ratio": 0.0,
            "downward_velocity": 0.0,
            "fall_velocity": False,
            "height_ratio": 1.0,
            "area_ratio": 1.0,
            "looks_sit_transition": False,
            "noisy": False,
        }
        previous_kpts = self._last_valid_kpts.get(track_id)
        previous_bbox = self._last_bbox_norm.get(track_id)
        current_bbox, bbox_w, bbox_h = _estimate_bbox_norm(kpts_norm, bbox_norm)

        if kpts_norm is None:
            return quality, current_bbox

        valid_mask = _valid_keypoint_mask(kpts_norm)
        quality["valid_ratio"] = float(valid_mask.sum() / max(kpts_norm.shape[0], 1))

        if previous_kpts is not None:
            prev_valid_mask = _valid_keypoint_mask(previous_kpts)
            common_mask = valid_mask & prev_valid_mask
            if common_mask.any():
                displacement = np.linalg.norm(kpts_norm[common_mask] - previous_kpts[common_mask], axis=1)
                quality["jitter_ratio"] = float(np.median(displacement) / max(bbox_w, bbox_h, 1e-3))

        current_center = _estimate_body_center(kpts_norm)
        previous_center = _estimate_body_center(previous_kpts) if previous_kpts is not None else None
        if current_center is not None and previous_center is not None:
            quality["downward_velocity"] = float((current_center[1] - previous_center[1]) / max(bbox_h, 1e-3))

        if previous_bbox is not None and current_bbox is not None:
            prev_w = max(float(previous_bbox[2] - previous_bbox[0]), 1e-3)
            prev_h = max(float(previous_bbox[3] - previous_bbox[1]), 1e-3)
            prev_area = max(prev_w * prev_h, 1e-3)
            quality["height_ratio"] = float(bbox_h / prev_h)
            quality["area_ratio"] = float((bbox_w * bbox_h) / prev_area)

        quality["fall_velocity"] = bool(quality["downward_velocity"] > self.fall_velocity_ratio)
        quality["looks_sit_transition"] = bool(
            quality["height_ratio"] < self.sitting_height_ratio or quality["area_ratio"] < self.sitting_area_ratio
        )
        quality["noisy"] = bool(quality["jitter_ratio"] > self.max_keypoint_jitter_ratio)
        return quality, current_bbox

    def update_track(self, track_id: int, kpts_norm: Optional[np.ndarray], bbox_norm: Optional[np.ndarray] = None):
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=SEQ_LEN)
            self._frame_count[track_id] = 0
            self._last_pred[track_id] = (-1, 0.0)
            self._frames_since_pred[track_id] = 0

        quality, current_bbox = self._build_quality_state(track_id, kpts_norm, bbox_norm)
        self._quality_state[track_id] = quality

        previous_valid = self._last_valid_kpts.get(track_id)
        if kpts_norm is None:
            if previous_valid is not None:
                buffer_kpts = previous_valid.copy()
            elif len(self._buffers[track_id]) > 0:
                buffer_kpts = self._buffers[track_id][-1].copy()
            else:
                buffer_kpts = np.zeros((17, 2), dtype=np.float32)
        elif bool(quality["valid_ratio"] < self.min_keypoint_ratio or quality["noisy"]):
            if previous_valid is not None:
                buffer_kpts = previous_valid.copy()
            else:
                buffer_kpts = kpts_norm.astype(np.float32, copy=True)
        else:
            buffer_kpts = kpts_norm.astype(np.float32, copy=True)
            self._last_valid_kpts[track_id] = buffer_kpts.copy()

        self._buffers[track_id].append(buffer_kpts)

        if current_bbox is not None:
            self._last_bbox_norm[track_id] = current_bbox.copy()

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

        quality = self._quality_state.get(track_id, {})
        if float(quality.get("valid_ratio", 0.0)) < self.min_keypoint_ratio or bool(quality.get("noisy", False)):
            lid, conf = self._last_pred.get(track_id, (-1, 0.0))
            return lid, conf, self.label_map.get(lid, "?")

        buf_len = len(self._buffers[track_id])
        cached_len = self._buf_len_at_cache.get(track_id, -1)
        if track_id not in self._feat_cache or buf_len != cached_len:
            if self.backend == "extratrees":
                if self.is_fast_mode_active():
                    x = prepare_sequence_fast(self._buffers[track_id], seq_len=self.fast_seq_len)
                else:
                    x = prepare_sequence(self._buffers[track_id])
            else:
                x = prepare_sequence(self._buffers[track_id])
            if self.feat_mean is not None:
                x = (x - self.feat_mean) / self.feat_std
            self._feat_cache[track_id] = x
            self._buf_len_at_cache[track_id] = buf_len
        else:
            x = self._feat_cache[track_id]

        probs = None
        started_at = time.perf_counter()
        if self.backend == "extratrees":
            seq69 = x[0]
            feat = build_extratrees_feature_vector(seq69)[np.newaxis, :]
            if hasattr(self.et_model, "predict_proba"):
                probs = self.et_model.predict_proba(feat)[0]
                label_id = int(np.argmax(probs))
                confidence = float(probs[label_id])
            else:
                label_id = int(self.et_model.predict(feat)[0])
                confidence = 1.0
        else:
            xt = torch.FloatTensor(x).to(self.device)
            logits, _ = self.model(xt)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            label_id = int(np.argmax(probs))
            confidence = float(probs[label_id])
        self._last_predict_ms[track_id] = (time.perf_counter() - started_at) * 1000.0
        if self._last_predict_ms[track_id] > self.track_time_budget_ms and self.backend == "extratrees":
            self._fast_mode_overload = True
            self._fast_mode_used = True

        fall_prob = float(probs[0]) if probs is not None and len(probs) > 0 else (confidence if label_id == 0 else 0.0)
        if fall_prob >= self.fall_priority_prob and bool(quality.get("fall_velocity", False)):
            label_id = 0
            confidence = max(confidence, fall_prob)

        class_threshold = self.conf_threshold
        if label_id == 0:
            class_threshold = max(0.01, self.conf_threshold - self.fall_conf_boost)
        elif label_id == 2:
            class_threshold = min(0.99, self.conf_threshold + self.sitting_conf_penalty)

        if confidence >= class_threshold:
            last_label_id = self._last_pred.get(track_id, (-1, 0.0))[0]
            if label_id == 2 and last_label_id == 1 and bool(quality.get("looks_sit_transition", False)) and not bool(quality.get("fall_velocity", False)):
                hold_until = self._pending_sitting_until.get(track_id)
                if hold_until is None:
                    self._pending_sitting_until[track_id] = n_frames + self.sitting_hold_frames
                    lid, conf = self._last_pred.get(track_id, (-1, 0.0))
                    return lid, conf, self.label_map.get(lid, "?")
                if n_frames < hold_until:
                    lid, conf = self._last_pred.get(track_id, (-1, 0.0))
                    return lid, conf, self.label_map.get(lid, "?")
            else:
                self._pending_sitting_until.pop(track_id, None)
            hist = self._pred_history.setdefault(track_id, [])
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

    def get_last_prediction(self, track_id: int) -> Tuple[int, float, str]:
        lid, conf = self._last_pred.get(track_id, (-1, 0.0))
        return lid, conf, self.label_map.get(lid, "?")

    def get_debug_state(self, track_id: int) -> Dict[str, float | bool | int | str]:
        quality = self._quality_state.get(track_id, {})
        lid, conf = self._last_pred.get(track_id, (-1, 0.0))
        frames_remaining = max(0, self._pending_sitting_until.get(track_id, 0) - self._frame_count.get(track_id, 0))
        return {
            "track_id": int(track_id),
            "label_id": int(lid),
            "label_name": self.label_map.get(lid, "?"),
            "confidence": float(conf),
            "valid_ratio": float(quality.get("valid_ratio", 0.0)),
            "jitter_ratio": float(quality.get("jitter_ratio", 0.0)),
            "downward_velocity": float(quality.get("downward_velocity", 0.0)),
            "fall_velocity": bool(quality.get("fall_velocity", False)),
            "looks_sit_transition": bool(quality.get("looks_sit_transition", False)),
            "noisy": bool(quality.get("noisy", False)),
            "predict_ms": float(self._last_predict_ms.get(track_id, 0.0)),
            "pending_sitting_frames": int(frames_remaining),
            "fast_mode_active": bool(self.is_fast_mode_active()),
        }

    def remove_stale_tracks(self, active_ids: set[int]):
        dead = [tid for tid in self._buffers if tid not in active_ids]
        for tid in dead:
            del self._buffers[tid]
            del self._frame_count[tid]
            del self._last_pred[tid]
            del self._frames_since_pred[tid]
            self._pred_history.pop(tid, None)
            self._feat_cache.pop(tid, None)
            self._buf_len_at_cache.pop(tid, None)
            self._quality_state.pop(tid, None)
            self._last_valid_kpts.pop(tid, None)
            self._last_bbox_norm.pop(tid, None)
            self._pending_sitting_until.pop(tid, None)
            self._last_predict_ms.pop(tid, None)


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
