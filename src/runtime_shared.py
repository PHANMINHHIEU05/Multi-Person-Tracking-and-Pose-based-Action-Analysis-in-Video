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

from src.action_model_common import (
    DEFAULT_ACTION_LABEL_MAP,
    EXTRATREES_FEATURE_SPEC_V1,
    LEGACY_ACTION_LABEL_MAP,
    build_extratrees_feature_vector,
    build_label_colors,
    get_action_color,
    normalize_label_map,
)

ROOT = Path(__file__).resolve().parent.parent
ACTIVE_ACTION_MODEL_PATH_FILE = ROOT / "runs" / "active_action_model_path.txt"

LABEL_MAP = DEFAULT_ACTION_LABEL_MAP.copy()
LABEL_COLORS = build_label_colors(LABEL_MAP)

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


def resolve_pose_inference_imgsz(requested_imgsz: int, weights_path: str) -> int:
    runtime_info = describe_pose_runtime(weights_path)
    imgsz = max(32, int(requested_imgsz))
    if runtime_info["backend"] == "TensorRT" and Path(weights_path).suffix.lower() == ".engine":
        # The current exported engine in this project is fixed to 640x640.
        return 640
    return imgsz


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
    if ACTIVE_ACTION_MODEL_PATH_FILE.exists():
        active_path = ACTIVE_ACTION_MODEL_PATH_FILE.read_text(encoding="utf-8").strip()
        if active_path and Path(active_path).is_file():
            return active_path

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
        model_path_l = str(model_path).lower()
        self.label_map = DEFAULT_ACTION_LABEL_MAP.copy() if model_path_l.endswith(".joblib") else LEGACY_ACTION_LABEL_MAP.copy()
        self.backend = "torch"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.et_model = None
        self.feat_mean = None
        self.feat_std = None
        self.feature_spec = EXTRATREES_FEATURE_SPEC_V1

        if model_path_l.endswith(".joblib"):
            artifact = joblib.load(model_path)
            if isinstance(artifact, dict) and "model" in artifact:
                self.et_model = artifact["model"]
                lm = artifact.get("label_map")
                if isinstance(lm, dict):
                    self.label_map = normalize_label_map(lm)
                self.feature_spec = str(artifact.get("feature_spec") or EXTRATREES_FEATURE_SPEC_V1)
            else:
                self.et_model = artifact
            self.backend = "extratrees"
            if hasattr(self.et_model, "set_params") and hasattr(self.et_model, "n_jobs"):
                self.et_model.set_params(n_jobs=1)
        else:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict):
                lm = checkpoint.get("label_map")
                if isinstance(lm, dict):
                    self.label_map = normalize_label_map(lm)
                num_classes = int(checkpoint.get("num_classes", len(self.label_map)))
            else:
                num_classes = len(self.label_map)
            model_class = load_action_model_class()
            self.model = model_class(
                input_dim=69,
                hidden_dim=128,
                num_layers=3,
                num_classes=num_classes,
                num_heads=8,
                dropout=0.0,
            ).to(self.device)

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
        self._tree_feat_cache: Dict[int, np.ndarray] = {}
        self._buf_len_at_tree_cache: Dict[int, int] = {}
        self._quality_state: Dict[int, Dict[str, float | bool]] = {}
        self._last_valid_kpts: Dict[int, np.ndarray] = {}
        self._last_bbox_norm: Dict[int, np.ndarray] = {}
        self._pending_sitting_until: Dict[int, int] = {}
        self._pending_fall_until: Dict[int, int] = {}
        self._fall_candidate_votes: Dict[int, int] = {}
        self._fall_recovery_votes: Dict[int, int] = {}
        self._last_predict_ms: Dict[int, float] = {}
        self._overload_track_count = False
        self._over_budget_predict = False

        self.min_keypoint_ratio = float(np.clip(min_keypoint_ratio, 0.10, 1.0))
        self.max_keypoint_jitter_ratio = float(max(max_keypoint_jitter_ratio, 0.01))
        self.fall_priority_prob = float(np.clip(fall_priority_prob, 0.05, 0.99))
        self.fall_velocity_ratio = float(max(fall_velocity_ratio, 0.01))
        self.sitting_hold_frames = max(1, int(sitting_hold_frames))
        self.sitting_height_ratio = float(np.clip(sitting_height_ratio, 0.30, 1.20))
        self.sitting_area_ratio = float(np.clip(sitting_area_ratio, 0.20, 1.20))
        self.fast_track_threshold = max(1, int(fast_track_threshold))
        self.track_time_budget_ms = float(max(track_time_budget_ms, 1.0))
        # Montage-style videos have short fall snippets and fast scene cuts:
        # keep Fall latched for a shorter window and release faster.
        self.fall_hold_frames = 6
        self.fall_enter_votes = 2
        self.fall_release_votes = 2
        self.fall_new_track_conf = 0.72
        self.fall_fastpath_conf = 0.45
        self.fall_live_fastpath_conf = 0.47
        self.fall_live_fastpath_velocity_ratio = self.fall_velocity_ratio * 0.92
        self.fall_transition_conf_floor = 0.43
        self.fall_decay_bbox_aspect_floor = 1.22
        self.fall_decay_area_ratio_ceiling = 0.78
        self._occlusion_lower_body_ratio = 0.50

        self._label_id_by_name = {str(name): int(label_id) for label_id, name in self.label_map.items()}
        self._fall_label_id = self._label_id_by_name.get("Fall")
        self._lying_label_id = self._label_id_by_name.get("Lying_Down")
        self._standing_label_id = self._label_id_by_name.get("Standing")
        self._walking_label_id = self._label_id_by_name.get("Walking")
        self._sitting_label_ids = {
            label_id
            for label_name, label_id in self._label_id_by_name.items()
            if label_name in {"Sitting", "Sitting_Quickly"}
        }

    def set_active_track_count(self, count: int):
        self._overload_track_count = count > self.fast_track_threshold

    def is_fast_mode_active(self) -> bool:
        return self.fast_mode

    def _display_label_name(self, label_id: int) -> str:
        label_name = self.label_map.get(label_id, "?")
        if label_name == "Sitting_Quickly":
            return "Sitting"
        if label_name == "Bending":
            return "?"
        return label_name

    def _display_prediction(self, label_id: int, confidence: float) -> Tuple[int, float, str]:
        return label_id, confidence, self._display_label_name(label_id)

    def _build_quality_state(
        self,
        track_id: int,
        kpts_norm: Optional[np.ndarray],
        bbox_norm: Optional[np.ndarray],
    ) -> Tuple[Dict[str, float | bool], Optional[np.ndarray]]:
        quality: Dict[str, float | bool] = {
            "valid_ratio": 0.0,
            "upper_body_ratio": 0.0,
            "lower_body_ratio": 0.0,
            "hip_to_ankle_ratio": 0.0,
            "jitter_ratio": 0.0,
            "downward_velocity": 0.0,
            "fall_velocity": False,
            "strong_fall_cue": False,
            "moderate_fall_cue": False,
            "chair_roll_cue": False,
            "height_ratio": 1.0,
            "area_ratio": 1.0,
            "bbox_aspect_ratio": 1.0,
            "occluded": False,
            "looks_sit_transition": False,
            "noisy": False,
            "rescue_applied": False,
            "rescue_reason": "",
        }
        previous_kpts = self._last_valid_kpts.get(track_id)
        previous_bbox = self._last_bbox_norm.get(track_id)
        current_bbox, bbox_w, bbox_h = _estimate_bbox_norm(kpts_norm, bbox_norm)
        quality["bbox_aspect_ratio"] = float(bbox_w / max(bbox_h, 1e-3))

        if kpts_norm is None:
            quality["occluded"] = True
            return quality, current_bbox

        valid_mask = _valid_keypoint_mask(kpts_norm)
        quality["valid_ratio"] = float(valid_mask.sum() / max(kpts_norm.shape[0], 1))
        upper_idxs = np.array([0, 5, 6, 7, 8, 9, 10], dtype=np.int32)
        lower_idxs = np.array([11, 12, 13, 14, 15, 16], dtype=np.int32)
        quality["upper_body_ratio"] = float(valid_mask[upper_idxs].sum() / max(len(upper_idxs), 1))
        quality["lower_body_ratio"] = float(valid_mask[lower_idxs].sum() / max(len(lower_idxs), 1))

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

        if valid_mask[L_HIP] and valid_mask[R_HIP] and valid_mask[L_ANKLE] and valid_mask[R_ANKLE]:
            hip_center = ((kpts_norm[L_HIP] + kpts_norm[R_HIP]) * 0.5).astype(np.float32)
            ankle_center = ((kpts_norm[L_ANKLE] + kpts_norm[R_ANKLE]) * 0.5).astype(np.float32)
            quality["hip_to_ankle_ratio"] = float((ankle_center[1] - hip_center[1]) / max(bbox_h, 1e-3))

        if previous_bbox is not None and current_bbox is not None:
            prev_w = max(float(previous_bbox[2] - previous_bbox[0]), 1e-3)
            prev_h = max(float(previous_bbox[3] - previous_bbox[1]), 1e-3)
            prev_area = max(prev_w * prev_h, 1e-3)
            quality["height_ratio"] = float(bbox_h / prev_h)
            quality["area_ratio"] = float((bbox_w * bbox_h) / prev_area)

        quality["fall_velocity"] = bool(quality["downward_velocity"] > self.fall_velocity_ratio)
        fall_trajectory = bool(quality["downward_velocity"] > self.fall_velocity_ratio * 0.88)
        collapse_shape = bool(
            quality["area_ratio"] < 0.78
            or quality["bbox_aspect_ratio"] > 1.18
            or (quality["height_ratio"] < 0.90 and quality["area_ratio"] < 0.88)
        )
        quality["strong_fall_cue"] = bool(fall_trajectory and collapse_shape)
        quality["moderate_fall_cue"] = bool(
            quality["downward_velocity"] > self.fall_velocity_ratio * 0.55
            and (
                quality["bbox_aspect_ratio"] > 0.92
                or quality["height_ratio"] < 0.96
                or quality["area_ratio"] < 0.92
            )
        )
        quality["chair_roll_cue"] = bool(
            quality["bbox_aspect_ratio"] > 1.00
            and quality["height_ratio"] < 0.95
            and quality["area_ratio"] < 0.96
        )
        quality["looks_sit_transition"] = bool(
            quality["height_ratio"] < self.sitting_height_ratio or quality["area_ratio"] < self.sitting_area_ratio
        )
        quality["noisy"] = bool(quality["jitter_ratio"] > self.max_keypoint_jitter_ratio)
        quality["occluded"] = bool(
            quality["valid_ratio"] < self.min_keypoint_ratio
            or quality["lower_body_ratio"] < self._occlusion_lower_body_ratio
        )
        return quality, current_bbox

    def _resolve_occlusion_label(
        self,
        track_id: int,
        label_id: int,
        confidence: float,
    ) -> Tuple[int, float, bool, str]:
        quality = self._quality_state.get(track_id, {})
        if not bool(quality.get("occluded", False)):
            return label_id, confidence, False, ""

        prev_label_id, prev_conf = self._last_pred.get(track_id, (-1, 0.0))
        bbox_aspect = float(quality.get("bbox_aspect_ratio", 1.0))
        downward_velocity = float(quality.get("downward_velocity", 0.0))
        fall_velocity = bool(quality.get("fall_velocity", False))
        area_ratio = float(quality.get("area_ratio", 1.0))
        lower_body_ratio = float(quality.get("lower_body_ratio", 0.0))
        strong_fall_cue = bool(quality.get("strong_fall_cue", False))

        # Strong fall cue: both hip downward velocity and body layout indicate collapse.
        if self._fall_label_id is not None and (
            strong_fall_cue
            or (fall_velocity and bbox_aspect >= 1.05 and area_ratio < 0.90)
        ):
            return self._fall_label_id, max(confidence, self.fall_priority_prob), True, "occ_fall_velocity"

        # Horizontal body under occlusion is more likely lying than sitting/walking.
        if self._lying_label_id is not None and bbox_aspect >= 1.40 and abs(downward_velocity) < self.fall_velocity_ratio:
            return self._lying_label_id, max(confidence, prev_conf, 0.45), True, "occ_bbox_horizontal"

        if prev_label_id >= 0:
            predicted_name = self.label_map.get(label_id, "")
            prev_name = self.label_map.get(prev_label_id, "")

            if (
                prev_label_id in {self._standing_label_id, self._walking_label_id}
                and lower_body_ratio < self._occlusion_lower_body_ratio
                and not fall_velocity
                and not strong_fall_cue
                and bbox_aspect < 1.25
            ):
                return prev_label_id, max(prev_conf, confidence * 0.85), True, "occ_missing_lower_body"

            if predicted_name in {"Fall", "Lying_Down"} and not fall_velocity and not strong_fall_cue and bbox_aspect < 1.20:
                return prev_label_id, max(prev_conf, confidence * 0.80), True, "occ_block_fall_lying"

            if (
                label_id in self._sitting_label_ids
                and prev_label_id in {self._standing_label_id, self._walking_label_id}
                and bbox_aspect < 1.05
            ):
                return prev_label_id, max(prev_conf, confidence * 0.80), True, "occ_block_sit"

            if area_ratio < 0.80 and prev_name not in {"Fall", "Lying_Down"}:
                return prev_label_id, max(prev_conf, confidence * 0.85), True, "occ_keep_prev_shrink"

            return prev_label_id, max(prev_conf, confidence * 0.75), True, "occ_keep_prev"

        return label_id, confidence, False, ""

    def _apply_sitting_hold_rule(
        self,
        track_id: int,
        label_id: int,
        confidence: float,
    ) -> Tuple[int, float, bool, str]:
        if label_id not in self._sitting_label_ids:
            self._pending_sitting_until.pop(track_id, None)
            return label_id, confidence, False, ""

        prev_label_id, prev_conf = self._last_pred.get(track_id, (-1, 0.0))
        if prev_label_id not in {self._standing_label_id, self._walking_label_id}:
            self._pending_sitting_until.pop(track_id, None)
            return label_id, confidence, False, ""

        quality = self._quality_state.get(track_id, {})
        if not bool(quality.get("occluded", False) or quality.get("noisy", False)):
            self._pending_sitting_until.pop(track_id, None)
            return label_id, confidence, False, ""
        if bool(quality.get("fall_velocity", False)):
            self._pending_sitting_until.pop(track_id, None)
            return label_id, confidence, False, ""

        looks_sit_transition = bool(quality.get("looks_sit_transition", False))
        frame_count = self._frame_count.get(track_id, 0)
        pending_until = self._pending_sitting_until.get(track_id)
        if not looks_sit_transition:
            if pending_until is None:
                self._pending_sitting_until[track_id] = frame_count + self.sitting_hold_frames
                return prev_label_id, max(prev_conf, confidence * 0.75), True, "sit_hold_start"
            if frame_count < pending_until:
                return prev_label_id, max(prev_conf, confidence * 0.75), True, "sit_hold_wait"
        self._pending_sitting_until.pop(track_id, None)
        return label_id, confidence, False, ""

    def _apply_posture_guardrails(
        self,
        track_id: int,
        label_id: int,
        confidence: float,
    ) -> Tuple[int, float, bool, str]:
        if self._lying_label_id is None or label_id != self._lying_label_id:
            return label_id, confidence, False, ""

        quality = self._quality_state.get(track_id, {})
        if bool(quality.get("fall_velocity", False)):
            return label_id, confidence, False, ""

        lower_body_ratio = float(quality.get("lower_body_ratio", 0.0))
        hip_to_ankle_ratio = float(quality.get("hip_to_ankle_ratio", 0.0))
        bbox_aspect = float(quality.get("bbox_aspect_ratio", 1.0))

        # Bent-forward posture: ankles remain significantly below hips, so avoid forcing lying.
        if lower_body_ratio >= 0.66 and hip_to_ankle_ratio >= 0.22 and bbox_aspect < 1.55:
            prev_label_id, prev_conf = self._last_pred.get(track_id, (-1, 0.0))
            if prev_label_id in {self._standing_label_id, self._walking_label_id}:
                return prev_label_id, max(prev_conf, confidence * 0.85), True, "posture_block_lying_to_upright"
            if self._standing_label_id is not None:
                return self._standing_label_id, max(0.40, confidence * 0.80), True, "posture_block_lying_to_standing"
        return label_id, confidence, False, ""

    def _apply_fall_priority_rule(
        self,
        track_id: int,
        label_id: int,
        confidence: float,
    ) -> Tuple[int, float, bool, str]:
        if self._fall_label_id is None:
            return label_id, confidence, False, ""

        quality = self._quality_state.get(track_id, {})
        frame_count = self._frame_count.get(track_id, 0)
        prev_label_id, prev_conf = self._last_pred.get(track_id, (-1, 0.0))
        is_prev_upright = prev_label_id in {self._standing_label_id, self._walking_label_id}
        strong_fall_cue = bool(quality.get("strong_fall_cue", False))
        moderate_fall_cue = bool(quality.get("moderate_fall_cue", False))
        chair_roll_cue = bool(quality.get("chair_roll_cue", False))
        fall_velocity = bool(quality.get("fall_velocity", False))
        downward_velocity = float(quality.get("downward_velocity", 0.0))
        height_ratio = float(quality.get("height_ratio", 1.0))
        area_ratio = float(quality.get("area_ratio", 1.0))
        bbox_aspect = float(quality.get("bbox_aspect_ratio", 1.0))
        very_severe_fall_cue = bool(
            strong_fall_cue
            and (
                downward_velocity > self.fall_velocity_ratio * 1.35
                or area_ratio < 0.70
                or bbox_aspect > 1.35
            )
        )
        valid_cue_driven_fall = bool(
            strong_fall_cue
            and (fall_velocity or area_ratio < 0.80 or bbox_aspect > 1.15)
            and (is_prev_upright or prev_label_id in {self._fall_label_id, self._lying_label_id})
        )
        pose_transition_fall = bool(
            label_id == self._fall_label_id
            and (
                is_prev_upright
                or prev_label_id in self._sitting_label_ids
                or prev_label_id in {self._fall_label_id, self._lying_label_id}
            )
            and confidence >= max(self.fall_priority_prob - 0.03, 0.36)
            and (moderate_fall_cue or chair_roll_cue or (fall_velocity and confidence >= (self.fall_transition_conf_floor - 0.06)))
        )

        recovery_cue = bool(
            downward_velocity < (-self.fall_velocity_ratio * 0.45)
            and height_ratio > 1.03
            and bbox_aspect < 0.95
        )

        predicted_fall_confident = bool(
            label_id == self._fall_label_id and confidence >= max(self.fall_priority_prob + 0.12, 0.55)
        )
        should_trigger_fall = bool(
            (predicted_fall_confident and (strong_fall_cue or (fall_velocity and confidence >= self.fall_new_track_conf)))
            or valid_cue_driven_fall
            or pose_transition_fall
        )

        if should_trigger_fall:
            votes = self._fall_candidate_votes.get(track_id, 0) + 1
            self._fall_candidate_votes[track_id] = votes
        else:
            self._fall_candidate_votes[track_id] = 0

        if should_trigger_fall:
            min_votes = self.fall_enter_votes
            rapid_transition_cue = bool(
                chair_roll_cue
                or (moderate_fall_cue and bbox_aspect >= 0.95 and area_ratio < 0.96)
                or downward_velocity > (self.fall_velocity_ratio * 1.25)
            )
            if pose_transition_fall and confidence >= max(self.fall_priority_prob, 0.42):
                if rapid_transition_cue:
                    min_votes = 1
            if prev_label_id < 0:
                min_votes = max(min_votes + 1, 3)
            if votes >= min_votes or very_severe_fall_cue:
                self._pending_fall_until[track_id] = frame_count + self.fall_hold_frames
                self._fall_recovery_votes[track_id] = 0
                self._fall_candidate_votes[track_id] = 0
                return self._fall_label_id, max(confidence, prev_conf, self.fall_priority_prob), True, "fall_priority"
            if prev_label_id in {self._standing_label_id, self._walking_label_id, self._lying_label_id}:
                return prev_label_id, max(prev_conf, confidence * 0.80), True, "fall_candidate_wait"
            return label_id, confidence, False, ""

        pending_until = self._pending_fall_until.get(track_id)
        if pending_until is not None and frame_count < pending_until:
            if label_id in {self._standing_label_id, self._walking_label_id}:
                if recovery_cue:
                    votes = self._fall_recovery_votes.get(track_id, 0) + 1
                    self._fall_recovery_votes[track_id] = votes
                    if votes >= self.fall_release_votes:
                        self._pending_fall_until.pop(track_id, None)
                        self._fall_recovery_votes.pop(track_id, None)
                        return label_id, max(confidence, 0.42), True, "fall_recovery_release"
                else:
                    self._fall_recovery_votes[track_id] = 0
                return self._fall_label_id, max(prev_conf, 0.42), True, "fall_hold"
            self._fall_recovery_votes[track_id] = 0
            if label_id == self._lying_label_id and not fall_velocity and not strong_fall_cue:
                return self._fall_label_id, max(prev_conf, 0.42), True, "fall_hold"
            return self._fall_label_id, max(prev_conf, confidence, 0.42), True, "fall_hold"

        self._pending_fall_until.pop(track_id, None)
        self._fall_candidate_votes[track_id] = 0
        if prev_label_id == self._fall_label_id and label_id in {self._standing_label_id, self._walking_label_id}:
            prone_post_fall = bool(
                bbox_aspect >= self.fall_decay_bbox_aspect_floor
                or area_ratio <= self.fall_decay_area_ratio_ceiling
            )
            if not recovery_cue and prone_post_fall:
                self._fall_recovery_votes[track_id] = 0
                if self._lying_label_id is not None:
                    return self._lying_label_id, max(prev_conf, 0.40), True, "fall_decay_to_lying"
                return self._fall_label_id, max(prev_conf, 0.40), True, "fall_decay"
        self._fall_recovery_votes.pop(track_id, None)
        return label_id, confidence, False, ""

    def _apply_live_fall_fastpath(self, track_id: int) -> None:
        if self._fall_label_id is None:
            return

        quality = self._quality_state.get(track_id, {})
        strong_fall_cue = bool(quality.get("strong_fall_cue", False))
        moderate_fall_cue = bool(quality.get("moderate_fall_cue", False))
        chair_roll_cue = bool(quality.get("chair_roll_cue", False))
        fall_velocity = bool(quality.get("fall_velocity", False))
        downward_velocity = float(quality.get("downward_velocity", 0.0))
        area_ratio = float(quality.get("area_ratio", 1.0))
        bbox_aspect = float(quality.get("bbox_aspect_ratio", 1.0))
        rapid_transition_cue = bool(
            chair_roll_cue
            or (
                moderate_fall_cue
                and (bbox_aspect >= 0.94 or area_ratio < 0.96)
            )
            or (
                fall_velocity
                and downward_velocity > self.fall_live_fastpath_velocity_ratio
                and (bbox_aspect > 0.92 or area_ratio < 0.97)
            )
        )
        if not (strong_fall_cue or rapid_transition_cue):
            return

        prev_label_id, prev_conf = self._last_pred.get(track_id, (-1, 0.0))
        if prev_label_id == self._fall_label_id:
            return
        if prev_label_id == self._lying_label_id and not strong_fall_cue:
            return
        if (
            prev_label_id >= 0
            and prev_label_id not in {self._standing_label_id, self._walking_label_id, self._lying_label_id}
            and not chair_roll_cue
        ):
            return

        proposed_conf = max(
            self.fall_live_fastpath_conf,
            self.fall_priority_prob + (0.04 if rapid_transition_cue else 0.0),
            prev_conf,
        )
        fall_label_id, fall_conf, fall_applied, _ = self._apply_fall_priority_rule(
            track_id,
            self._fall_label_id,
            proposed_conf,
        )
        if not fall_applied:
            return

        quality = self._quality_state.setdefault(track_id, {})
        quality["rescue_applied"] = True
        quality["rescue_reason"] = "fall_live_fastpath"
        hist = self._pred_history.setdefault(track_id, [])
        hist.clear()
        hist.append(int(fall_label_id))
        self._last_pred[track_id] = (int(fall_label_id), float(fall_conf))
        self._frames_since_pred[track_id] = 0

    def update_track(self, track_id: int, kpts_norm: Optional[np.ndarray], bbox_norm: Optional[np.ndarray] = None):
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=SEQ_LEN)
            self._frame_count[track_id] = 0
            self._last_pred[track_id] = (-1, 0.0)
            self._frames_since_pred[track_id] = 0

        quality, current_bbox = self._build_quality_state(track_id, kpts_norm, bbox_norm)
        self._quality_state[track_id] = quality

        if kpts_norm is None:
            if len(self._buffers[track_id]) > 0:
                buffer_kpts = self._buffers[track_id][-1].copy()
            else:
                buffer_kpts = np.zeros((17, 2), dtype=np.float32)
        else:
            buffer_kpts = kpts_norm.astype(np.float32, copy=True)

        if kpts_norm is not None:
            self._last_valid_kpts[track_id] = kpts_norm.astype(np.float32, copy=True)

        self._buffers[track_id].append(buffer_kpts)

        if current_bbox is not None:
            self._last_bbox_norm[track_id] = current_bbox.copy()

        self._frame_count[track_id] += 1
        self._frames_since_pred[track_id] += 1
        self._apply_live_fall_fastpath(track_id)

    @torch.no_grad()
    def _prepare_sequence_features(self, track_id: int) -> np.ndarray:
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
        return self._feat_cache[track_id]

    def _prepare_tree_features(self, track_id: int) -> np.ndarray:
        buf_len = len(self._buffers[track_id])
        cached_len = self._buf_len_at_tree_cache.get(track_id, -1)
        if track_id not in self._tree_feat_cache or buf_len != cached_len:
            x = self._prepare_sequence_features(track_id)
            seq69 = x[0]
            feat = build_extratrees_feature_vector(seq69, feature_spec=self.feature_spec)
            self._tree_feat_cache[track_id] = feat
            self._buf_len_at_tree_cache[track_id] = buf_len
        return self._tree_feat_cache[track_id]

    def _apply_prediction_result(self, track_id: int, label_id: int, confidence: float) -> Tuple[int, float, str]:
        quality = self._quality_state.get(track_id, {})
        class_threshold = self.conf_threshold
        label_name = self.label_map.get(label_id, "")
        if label_name == "Bending":
            lid, conf = self._last_pred.get(track_id, (-1, 0.0))
            return self._display_prediction(lid, conf)
        if label_id == 0:
            class_threshold = max(0.01, self.conf_threshold - self.fall_conf_boost)
        elif label_name in {"Sitting_Quickly", "Sitting"}:
            sit_penalty = self.sitting_conf_penalty
            if bool(quality.get("looks_sit_transition", False)) and float(quality.get("lower_body_ratio", 0.0)) >= 0.60:
                sit_penalty *= 0.35
            elif bool(quality.get("occluded", False)):
                sit_penalty *= 1.10
            class_threshold = min(0.99, self.conf_threshold + sit_penalty)

        fall_label_id, fall_conf, fall_applied, fall_reason = self._apply_fall_priority_rule(
            track_id, label_id, confidence
        )
        if fall_applied:
            quality = self._quality_state.setdefault(track_id, {})
            quality["rescue_applied"] = True
            quality["rescue_reason"] = fall_reason
            hist = self._pred_history.setdefault(track_id, [])
            hist.clear()
            hist.append(int(fall_label_id))
            self._last_pred[track_id] = (int(fall_label_id), float(fall_conf))
            return self._display_prediction(int(fall_label_id), float(fall_conf))
        label_id = fall_label_id
        confidence = fall_conf

        if label_id == self._fall_label_id:
            prev_label_id, prev_conf = self._last_pred.get(track_id, (-1, 0.0))
            fall_velocity = bool(quality.get("fall_velocity", False))
            strong_fall_cue = bool(quality.get("strong_fall_cue", False))
            moderate_fall_cue = bool(quality.get("moderate_fall_cue", False))
            chair_roll_cue = bool(quality.get("chair_roll_cue", False))
            high_conf_fall = confidence >= max(self.fall_new_track_conf, self.fall_priority_prob + 0.20)
            transitioning_into_fall = prev_label_id != self._fall_label_id
            fall_transition_floor = max(self.fall_transition_conf_floor, self.fall_priority_prob + 0.05)
            transition_has_cue = bool(fall_velocity or strong_fall_cue or moderate_fall_cue or chair_roll_cue)

            # Block weak/no-cue Fall transitions, especially in quick-cut videos.
            if transitioning_into_fall and not transition_has_cue and confidence < fall_transition_floor:
                quality = self._quality_state.setdefault(track_id, {})
                quality["rescue_applied"] = True
                quality["rescue_reason"] = "fall_gate_transition_low_conf"
                if prev_label_id >= 0:
                    self._last_pred[track_id] = (int(prev_label_id), float(max(prev_conf, confidence * 0.80)))
                else:
                    self._last_pred[track_id] = (int(prev_label_id), float(max(prev_conf, 0.0)))
                return self._display_prediction(self._last_pred[track_id][0], self._last_pred[track_id][1])

            if prev_label_id < 0 and not (transition_has_cue or high_conf_fall):
                quality = self._quality_state.setdefault(track_id, {})
                quality["rescue_applied"] = True
                quality["rescue_reason"] = "fall_gate_new_track"
                self._last_pred[track_id] = (int(prev_label_id), float(max(prev_conf, 0.0)))
                return self._display_prediction(self._last_pred[track_id][0], self._last_pred[track_id][1])
            if prev_label_id in {self._standing_label_id, self._walking_label_id} and not (transition_has_cue or high_conf_fall):
                quality = self._quality_state.setdefault(track_id, {})
                quality["rescue_applied"] = True
                quality["rescue_reason"] = "fall_gate_keep_prev"
                self._last_pred[track_id] = (int(prev_label_id), float(max(prev_conf, confidence * 0.75)))
                return self._display_prediction(self._last_pred[track_id][0], self._last_pred[track_id][1])

        guard_label_id, guard_conf, guard_applied, guard_reason = self._apply_posture_guardrails(
            track_id, label_id, confidence
        )
        if guard_applied:
            quality = self._quality_state.setdefault(track_id, {})
            quality["rescue_applied"] = True
            quality["rescue_reason"] = guard_reason
            self._last_pred[track_id] = (int(guard_label_id), float(guard_conf))
            return self._display_prediction(int(guard_label_id), float(guard_conf))
        label_id = guard_label_id
        confidence = guard_conf

        rescue_label_id, rescue_conf, rescue_applied, rescue_reason = self._resolve_occlusion_label(
            track_id, label_id, confidence
        )
        if rescue_applied:
            quality = self._quality_state.setdefault(track_id, {})
            quality["rescue_applied"] = True
            quality["rescue_reason"] = rescue_reason
            hist = self._pred_history.setdefault(track_id, [])
            hist.append(int(rescue_label_id))
            if len(hist) > self.smooth_window:
                hist.pop(0)
            self._last_pred[track_id] = (int(rescue_label_id), float(rescue_conf))
            lid, conf = self._last_pred.get(track_id, (-1, 0.0))
            return self._display_prediction(lid, conf)
        quality = self._quality_state.setdefault(track_id, {})
        quality["rescue_applied"] = False
        quality["rescue_reason"] = ""

        if confidence >= class_threshold:
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
                held_id, held_conf, held_applied, held_reason = self._apply_sitting_hold_rule(
                    track_id,
                    smoothed_id,
                    confidence,
                )
                if held_applied:
                    quality["rescue_applied"] = True
                    quality["rescue_reason"] = held_reason
                self._last_pred[track_id] = (int(held_id), float(held_conf))

        lid, conf = self._last_pred.get(track_id, (-1, 0.0))
        return self._display_prediction(lid, conf)

    def predict(self, track_id: int) -> Tuple[int, float, str]:
        n_frames = self._frame_count.get(track_id, 0)
        since_pred = self._frames_since_pred.get(track_id, 0)
        quality = self._quality_state.get(track_id, {})
        emergency_fall_cue = bool(quality.get("strong_fall_cue", False))

        if emergency_fall_cue and self._fall_label_id is not None:
            fall_label_id, fall_conf, fall_applied, _ = self._apply_fall_priority_rule(
                track_id,
                self._fall_label_id,
                max(self.fall_fastpath_conf, self.fall_priority_prob),
            )
            if fall_applied:
                quality = self._quality_state.setdefault(track_id, {})
                quality["rescue_applied"] = True
                quality["rescue_reason"] = "fall_emergency_fastpath"
                hist = self._pred_history.setdefault(track_id, [])
                hist.clear()
                hist.append(int(fall_label_id))
                self._last_pred[track_id] = (int(fall_label_id), float(fall_conf))
                self._frames_since_pred[track_id] = 0
                return self._display_prediction(int(fall_label_id), float(fall_conf))

        if n_frames < self.min_track_frames or since_pred < self.pred_stride:
            lid, conf = self._last_pred.get(track_id, (-1, 0.0))
            return self._display_prediction(lid, conf)

        self._frames_since_pred[track_id] = 0

        started_at = time.perf_counter()
        if self.backend == "extratrees":
            feat = self._prepare_tree_features(track_id)[np.newaxis, :]
            if hasattr(self.et_model, "predict_proba"):
                probs = self.et_model.predict_proba(feat)
                row = probs[0]
                label_id = int(np.argmax(row))
                confidence = float(row[label_id])
            else:
                label_id = int(self.et_model.predict(feat)[0])
                confidence = 1.0
        else:
            x = self._prepare_sequence_features(track_id)
            xt = torch.FloatTensor(x).to(self.device)
            logits, _ = self.model(xt)
            row = F.softmax(logits, dim=-1)[0].cpu().numpy()
            label_id = int(np.argmax(row))
            confidence = float(row[label_id])
        self._last_predict_ms[track_id] = (time.perf_counter() - started_at) * 1000.0
        if self._last_predict_ms[track_id] > self.track_time_budget_ms:
            self._over_budget_predict = True

        return self._apply_prediction_result(track_id, label_id, confidence)

    def collect_batch_features(self, track_ids: list[int]) -> Tuple[list[int], Optional[np.ndarray]]:
        if self.backend != "extratrees" or not track_ids:
            return [], None

        ready_ids: list[int] = []
        feature_rows: list[np.ndarray] = []
        for track_id in track_ids:
            if track_id not in self._buffers:
                continue
            n_frames = self._frame_count.get(track_id, 0)
            since_pred = self._frames_since_pred.get(track_id, 0)
            if n_frames < self.min_track_frames or since_pred < self.pred_stride:
                continue
            ready_ids.append(track_id)
            feature_rows.append(self._prepare_tree_features(track_id))

        if not ready_ids:
            return [], None

        feature_matrix = np.stack(feature_rows, axis=0).astype(np.float32, copy=False)
        return ready_ids, feature_matrix

    def mark_predictions_submitted(self, track_ids: list[int]) -> None:
        for track_id in track_ids:
            if track_id in self._frames_since_pred:
                self._frames_since_pred[track_id] = 0

    def apply_batch_prediction_results(
        self,
        track_ids: list[int],
        *,
        probs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        elapsed_ms: float = 0.0,
    ) -> Dict[int, Tuple[int, float, str]]:
        outputs: Dict[int, Tuple[int, float, str]] = {}
        if not track_ids:
            return outputs

        per_track_ms = float(elapsed_ms) / max(len(track_ids), 1)
        if probs is not None:
            for track_id, row in zip(track_ids, probs):
                if track_id not in self._buffers:
                    continue
                label_id = int(np.argmax(row))
                confidence = float(row[label_id])
                self._last_predict_ms[track_id] = per_track_ms
                if per_track_ms > self.track_time_budget_ms:
                    self._over_budget_predict = True
                outputs[track_id] = self._apply_prediction_result(track_id, label_id, confidence)
            return outputs

        if labels is not None:
            for track_id, label in zip(track_ids, labels):
                if track_id not in self._buffers:
                    continue
                self._last_predict_ms[track_id] = per_track_ms
                if per_track_ms > self.track_time_budget_ms:
                    self._over_budget_predict = True
                outputs[track_id] = self._apply_prediction_result(track_id, int(label), 1.0)
            return outputs

        for track_id in track_ids:
            if track_id in self._buffers:
                outputs[track_id] = self.get_last_prediction(track_id)
        return outputs

    def predict_batch(self, track_ids: list[int]) -> Dict[int, Tuple[int, float, str]]:
        if self.backend != "extratrees" or not track_ids:
            return {track_id: self.predict(track_id) for track_id in track_ids}

        outputs: Dict[int, Tuple[int, float, str]] = {}
        ready_ids, feature_matrix = self.collect_batch_features(track_ids)
        self.mark_predictions_submitted(ready_ids)

        if ready_ids and feature_matrix is not None:
            started_at = time.perf_counter()
            if hasattr(self.et_model, "predict_proba"):
                probs = self.et_model.predict_proba(feature_matrix)
                outputs.update(
                    self.apply_batch_prediction_results(
                        ready_ids,
                        probs=probs,
                        elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                    )
                )
            else:
                labels = self.et_model.predict(feature_matrix)
                outputs.update(
                    self.apply_batch_prediction_results(
                        ready_ids,
                        labels=labels,
                        elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                    )
                )

        for track_id in track_ids:
            outputs.setdefault(track_id, self.get_last_prediction(track_id))
        return outputs

    def get_last_prediction(self, track_id: int) -> Tuple[int, float, str]:
        lid, conf = self._last_pred.get(track_id, (-1, 0.0))
        return self._display_prediction(lid, conf)

    def get_debug_state(self, track_id: int) -> Dict[str, float | bool | int | str]:
        quality = self._quality_state.get(track_id, {})
        lid, conf = self._last_pred.get(track_id, (-1, 0.0))
        frames_remaining = max(0, self._pending_sitting_until.get(track_id, 0) - self._frame_count.get(track_id, 0))
        fall_frames_remaining = max(0, self._pending_fall_until.get(track_id, 0) - self._frame_count.get(track_id, 0))
        fall_candidate_votes = int(self._fall_candidate_votes.get(track_id, 0))
        fall_recovery_votes = int(self._fall_recovery_votes.get(track_id, 0))
        return {
            "track_id": int(track_id),
            "label_id": int(lid),
            "label_name": self.label_map.get(lid, "?"),
            "confidence": float(conf),
            "valid_ratio": float(quality.get("valid_ratio", 0.0)),
            "upper_body_ratio": float(quality.get("upper_body_ratio", 0.0)),
            "lower_body_ratio": float(quality.get("lower_body_ratio", 0.0)),
            "hip_to_ankle_ratio": float(quality.get("hip_to_ankle_ratio", 0.0)),
            "jitter_ratio": float(quality.get("jitter_ratio", 0.0)),
            "downward_velocity": float(quality.get("downward_velocity", 0.0)),
            "fall_velocity": bool(quality.get("fall_velocity", False)),
            "strong_fall_cue": bool(quality.get("strong_fall_cue", False)),
            "moderate_fall_cue": bool(quality.get("moderate_fall_cue", False)),
            "chair_roll_cue": bool(quality.get("chair_roll_cue", False)),
            "bbox_aspect_ratio": float(quality.get("bbox_aspect_ratio", 1.0)),
            "occluded": bool(quality.get("occluded", False)),
            "looks_sit_transition": bool(quality.get("looks_sit_transition", False)),
            "noisy": bool(quality.get("noisy", False)),
            "rescue_applied": bool(quality.get("rescue_applied", False)),
            "rescue_reason": str(quality.get("rescue_reason", "")),
            "predict_ms": float(self._last_predict_ms.get(track_id, 0.0)),
            "pending_sitting_frames": int(frames_remaining),
            "pending_fall_frames": int(fall_frames_remaining),
            "fall_candidate_votes": int(fall_candidate_votes),
            "fall_recovery_votes": int(fall_recovery_votes),
            "fast_mode_active": bool(self.is_fast_mode_active()),
            "overload_track_count": bool(self._overload_track_count),
            "over_budget_predict": bool(self._over_budget_predict),
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
            self._tree_feat_cache.pop(tid, None)
            self._buf_len_at_tree_cache.pop(tid, None)
            self._quality_state.pop(tid, None)
            self._last_valid_kpts.pop(tid, None)
            self._last_bbox_norm.pop(tid, None)
            self._pending_sitting_until.pop(tid, None)
            self._pending_fall_until.pop(tid, None)
            self._fall_candidate_votes.pop(tid, None)
            self._fall_recovery_votes.pop(tid, None)
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
