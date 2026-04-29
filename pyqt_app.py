from __future__ import annotations

import json
import sys
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QThread, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.runtime_shared import (
    ActionRecognizerLite,
    ROOT,
    describe_pose_runtime,
    bbox_center_distance_norm,
    bbox_iou_xyxy,
    draw_action_label,
    draw_skeleton,
    extract_kpts_for_track,
    get_action_color,
    load_pose_model,
    resolve_default_action_model_path,
    resolve_pose_inference_imgsz,
    resolve_default_pose_weights_path,
    resolve_tracker_config,
)

OUTPUT_DIR = ROOT / "runs" / "qt_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_POSE_MODEL_CACHE = {}
DEFAULT_PREVIEW_WIDTH = 960
DEFAULT_PREVIEW_HEIGHT = 540
VIDEO_READER_QUEUE_SIZE = 8


def load_pose_model_qt(weights_path: str):
    model = _POSE_MODEL_CACHE.get(weights_path)
    if model is None:
        model = load_pose_model(weights_path)
        _POSE_MODEL_CACHE[weights_path] = model
    return model


def frame_to_qimage(frame_bgr: np.ndarray, target_size=None) -> QImage:  # FIX: avoid QImage.copy() bottleneck
    # FIX: avoid QImage.copy() by keeping rgb array alive via closure
    if target_size is not None:  # FIX: resize only when caller requests target preview size
        target_w = max(1, int(target_size[0]))  # FIX: guard width against invalid values
        target_h = max(1, int(target_size[1]))  # FIX: guard height against invalid values
        frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)  # FIX: efficient resize path
    # FIX: convert once, keep reference alive so QImage doesn't dangle
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))  # FIX: contiguous RGB buffer for QImage
    h, w, ch = rgb.shape  # FIX: derive image shape from converted RGB array
    img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)  # FIX: wrap numpy buffer without extra copy
    img._rgb_keep_alive = rgb  # FIX: pin array to QImage lifetime, no copy needed
    return img  # FIX: return zero-copy QImage wrapper


@dataclass
class RuntimeConfig:
    source_mode: str
    video_path: str
    camera_index: int
    pose_weights: str
    action_model_path: str
    tracker_name: str
    det_conf: float
    det_iou: float
    imgsz: int
    max_det: int
    draw_skeleton: bool
    live_preview: bool
    preview_width: int
    preview_height: int
    webcam_duration_sec: int
    preview_stride: int
    process_stride: int
    output_scale: float
    save_output_video: bool
    skip_action_model: bool
    normalize_timing: bool
    auto_tune_cpu: bool
    target_analysis_fps: float
    min_track_frames: int
    pred_stride: int
    action_conf: float
    smooth_window: int
    fall_conf_boost: float
    sitting_conf_penalty: float
    keypoint_integrity_ratio: float
    keypoint_jitter_ratio: float
    fall_priority_prob: float
    fall_velocity_ratio: float
    sitting_hold_frames: int
    track_time_budget_ms: float
    fast_track_threshold: int


@dataclass
class VisualTrackState:
    bbox: np.ndarray
    label: str
    conf: float
    color: tuple[int, int, int]
    kpts: Optional[np.ndarray]
    prev_bbox: Optional[np.ndarray]
    prev_kpts: Optional[np.ndarray]
    last_frame_idx: int
    prev_frame_idx: int


def smooth_motion_array(
    previous: Optional[np.ndarray],
    current: Optional[np.ndarray],
    alpha: float,
) -> Optional[np.ndarray]:
    if current is None:
        return previous.copy() if previous is not None else None
    current_f32 = current.astype(np.float32, copy=True)
    if previous is None:
        return current_f32
    return ((1.0 - alpha) * previous + alpha * current_f32).astype(np.float32)


def extrapolate_motion_array(
    current: Optional[np.ndarray],
    previous: Optional[np.ndarray],
    current_frame_idx: int,
    previous_frame_idx: int,
    target_frame_idx: int,
    *,
    velocity_scale: float = 0.8,
    clamp_unit: bool = False,
) -> Optional[np.ndarray]:
    if current is None:
        return None
    predicted = current.astype(np.float32, copy=True)
    if previous is not None and current_frame_idx > previous_frame_idx >= 0 and target_frame_idx > current_frame_idx:
        frame_delta = float(current_frame_idx - previous_frame_idx)
        velocity = (predicted - previous) / max(frame_delta, 1.0)
        predicted = predicted + velocity * float(target_frame_idx - current_frame_idx) * velocity_scale
    if clamp_unit:
        predicted = np.clip(predicted, 0.0, 1.0)
    return predicted.astype(np.float32)


def resolve_visual_draw_state(state: VisualTrackState, target_frame_idx: int) -> tuple[np.ndarray, Optional[np.ndarray], str, float, tuple[int, int, int]]:
    bbox = extrapolate_motion_array(
        state.bbox,
        state.prev_bbox,
        state.last_frame_idx,
        state.prev_frame_idx,
        target_frame_idx,
    )
    kpts = extrapolate_motion_array(
        state.kpts,
        state.prev_kpts,
        state.last_frame_idx,
        state.prev_frame_idx,
        target_frame_idx,
        clamp_unit=True,
    )
    return bbox if bbox is not None else state.bbox.copy(), kpts, state.label, state.conf, state.color


@dataclass
class ActionBatchTask:
    task_id: int
    submitted_at: float
    track_ids: list[int]
    feature_matrix: np.ndarray


@dataclass
class ActionBatchResult:
    task_id: int
    submitted_at: float
    track_ids: list[int]
    probs: Optional[np.ndarray]
    labels: Optional[np.ndarray]
    elapsed_ms: float
    error: Optional[str] = None


class AsyncActionPredictor(threading.Thread):
    def __init__(self, model):
        super().__init__(daemon=True)
        self.model = model
        self._result_queue: queue.Queue[ActionBatchResult] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._task_condition = threading.Condition()
        self._latest_task: Optional[ActionBatchTask] = None
        self._next_task_id = 1
        self._processing = False

    def is_busy(self) -> bool:
        with self._task_condition:
            return self._processing or self._latest_task is not None or not self._result_queue.empty()

    def submit(self, track_ids: list[int], feature_matrix: Optional[np.ndarray]) -> int:
        if not track_ids or feature_matrix is None or self._stop_event.is_set():
            return 0
        with self._task_condition:
            task = ActionBatchTask(
                task_id=self._next_task_id,
                submitted_at=time.perf_counter(),
                track_ids=list(track_ids),
                feature_matrix=np.array(feature_matrix, copy=True),
            )
            self._next_task_id += 1
            # Latest-only behavior: replace any pending stale task.
            self._latest_task = task
            self._task_condition.notify()
            return task.task_id

    def poll_result(self) -> Optional[ActionBatchResult]:
        try:
            result = self._result_queue.get_nowait()
        except queue.Empty:
            return None
        with self._task_condition:
            if self._latest_task is None and not self._processing and self._result_queue.empty():
                self._task_condition.notify_all()
        return result

    def stop(self) -> None:
        self._stop_event.set()
        with self._task_condition:
            self._latest_task = None
            self._task_condition.notify_all()

    def _publish_result(self, result: ActionBatchResult) -> None:
        while not self._stop_event.is_set():
            try:
                self._result_queue.put(result, timeout=0.1)
                return
            except queue.Full:
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass

    def run(self) -> None:
        while not self._stop_event.is_set():
            with self._task_condition:
                while self._latest_task is None and not self._stop_event.is_set():
                    self._task_condition.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break
                task = self._latest_task
                self._latest_task = None
                self._processing = True

            if task is None:
                with self._task_condition:
                    self._processing = False
                continue

            started_at = time.perf_counter()
            probs = None
            labels = None
            error = None
            try:
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(task.feature_matrix)
                else:
                    labels = self.model.predict(task.feature_matrix)
            except Exception as exc:
                error = str(exc)

            self._publish_result(
                ActionBatchResult(
                    task_id=task.task_id,
                    submitted_at=task.submitted_at,
                    track_ids=task.track_ids,
                    probs=probs,
                    labels=labels,
                    elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                    error=error,
                )
            )
            with self._task_condition:
                self._processing = False
                if self._latest_task is None and self._result_queue.empty():
                    self._task_condition.notify_all()


class InferenceWorker(QThread):
    frame_ready = pyqtSignal()
    metrics_ready = pyqtSignal(dict)
    status_ready = pyqtSignal(str)
    finished_ready = pyqtSignal(dict)
    error_ready = pyqtSignal(str)

    def __init__(self, config: RuntimeConfig):
        super().__init__()
        self.config = config
        self._stop_requested = False
        self._preview_lock = threading.Lock()
        self._latest_preview_image: Optional[QImage] = None
        self._preview_signal_pending = False

    def stop(self) -> None:
        self._stop_requested = True

    def take_latest_preview(self) -> Optional[QImage]:
        with self._preview_lock:
            image = self._latest_preview_image
            self._latest_preview_image = None
            self._preview_signal_pending = False
        return image

    def _build_recognizer(self) -> Optional[ActionRecognizerLite]:
        if self.config.skip_action_model:
            return None
        if not Path(self.config.action_model_path).exists():
            raise FileNotFoundError(f"Action model not found: {self.config.action_model_path}")
        return ActionRecognizerLite(
            model_path=self.config.action_model_path,
            conf_threshold=self.config.action_conf,
            pred_stride=self.config.pred_stride,
            min_track_frames=self.config.min_track_frames,
            smooth_window=self.config.smooth_window,
            fall_conf_boost=self.config.fall_conf_boost,
            sitting_conf_penalty=self.config.sitting_conf_penalty,
            min_keypoint_ratio=self.config.keypoint_integrity_ratio,
            max_keypoint_jitter_ratio=self.config.keypoint_jitter_ratio,
            fall_priority_prob=self.config.fall_priority_prob,
            fall_velocity_ratio=self.config.fall_velocity_ratio,
            sitting_hold_frames=self.config.sitting_hold_frames,
            track_time_budget_ms=self.config.track_time_budget_ms,
            fast_track_threshold=self.config.fast_track_threshold,
        )

    def run(self) -> None:
        try:
            self.finished_ready.emit(self._run_inference())
        except Exception as exc:
            self.error_ready.emit(str(exc))

    def _run_inference(self) -> dict:
        cfg = self.config
        if cfg.source_mode == "video":
            if not Path(cfg.video_path).exists():
                raise FileNotFoundError(f"Video not found: {cfg.video_path}")
            cap = cv2.VideoCapture(cfg.video_path)
            output_stem = Path(cfg.video_path).stem
            max_frames = None
        else:
            cap = cv2.VideoCapture(int(cfg.camera_index))
            output_stem = f"webcam_{cfg.camera_index}"
            approx_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            max_frames = int(max(1, cfg.webcam_duration_sec) * max(1.0, approx_fps))

        if not cap.isOpened():
            raise RuntimeError("Cannot open selected input source.")

        self.status_ready.emit("Loading pose model...")
        pose_model = load_pose_model_qt(cfg.pose_weights)
        pose_runtime = getattr(pose_model, "_codex_runtime_info", describe_pose_runtime(cfg.pose_weights))
        effective_pose_imgsz = resolve_pose_inference_imgsz(cfg.imgsz, cfg.pose_weights)
        self.status_ready.emit(
            f"Pose runtime: {pose_runtime.get('backend', 'unknown')} on {pose_runtime.get('device', 'unknown')}"
        )
        if effective_pose_imgsz != int(cfg.imgsz):
            self.status_ready.emit(
                f"Pose input size adjusted from {int(cfg.imgsz)} to {effective_pose_imgsz} for this TensorRT engine."
            )

        self.status_ready.emit("Loading action model...")
        recognizer = self._build_recognizer()
        action_predictor: Optional[AsyncActionPredictor] = None

        tracker_cfg = resolve_tracker_config(cfg.tracker_name)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cpu_only = not torch.cuda.is_available()

        effective_process_stride = max(1, cfg.process_stride)
        effective_preview_stride = max(1, cfg.preview_stride)
        effective_max_det = max(1, cfg.max_det)
        effective_det_conf = float(cfg.det_conf)
        effective_target_analysis_fps = max(0.0, cfg.target_analysis_fps)
        cpu_auto_tuned = False
        if cfg.source_mode == "video" and not cfg.skip_action_model and effective_max_det < 3:
            effective_max_det = 3
            self.status_ready.emit("MaxDet raised to 3 for video action runs to reduce missed transition tracks.")
        if cfg.auto_tune_cpu and cpu_only and effective_process_stride < 2:
            effective_process_stride = 2
            cpu_auto_tuned = True
        if cfg.auto_tune_cpu and cpu_only and cfg.source_mode == "video" and effective_process_stride < 3:
            effective_process_stride = 3
            cpu_auto_tuned = True
        if cfg.auto_tune_cpu and cpu_only and cfg.live_preview and effective_preview_stride < 4:
            effective_preview_stride = 4
            cpu_auto_tuned = True
        if not cfg.live_preview and effective_preview_stride < 10:
            effective_preview_stride = 10
        if cfg.auto_tune_cpu and cpu_only and effective_max_det > 20:
            effective_max_det = 20
            cpu_auto_tuned = True
        if cfg.normalize_timing and effective_target_analysis_fps > 0 and fps_src > effective_target_analysis_fps:
            effective_process_stride = max(
                effective_process_stride,
                int(round(fps_src / effective_target_analysis_fps)),
            )
        if (
            cfg.source_mode == "video"
            and not cpu_only
            and effective_process_stride >= 3
            and effective_det_conf < 0.36
        ):
            effective_det_conf = 0.36
            self.status_ready.emit(
                f"Detection confidence raised to {effective_det_conf:.2f} for high-stride video to suppress false person tracks."
            )

        effective_action_update_stride = 1
        action_backend = None
        action_fast_mode = False
        action_fast_mode_used = False
        effective_action_pred_stride = None
        if recognizer is not None:
            action_backend = getattr(recognizer, "backend", "unknown")
            if action_backend == "extratrees":
                action_fast_mode = bool(cfg.auto_tune_cpu and cpu_only and cfg.source_mode == "video")
                recognizer.fast_mode = action_fast_mode
                action_fast_mode_used = action_fast_mode
            # Accuracy-first desktop video profile (video + GPU): allow densest action cadence.
            # Keep this strict so only explicit high-accuracy profiles use update stride = 1.
            accuracy_first_video = bool(
                action_backend == "extratrees"
                and cfg.source_mode == "video"
                and not cpu_only
                and int(cfg.process_stride) <= 1
                and int(cfg.pred_stride) <= 1
                and int(cfg.max_det) <= 12
                and not bool(cfg.normalize_timing)
                and not bool(cfg.auto_tune_cpu)
            )
            if action_backend == "torch":
                # Torch backend on GPU has enough headroom; keep action updates dense
                # to avoid missing short fall transitions.
                if cpu_only:
                    effective_action_update_stride = 2 if cfg.source_mode == "webcam" else 3
                else:
                    effective_action_update_stride = 1
            else:
                effective_action_update_stride = 1 if cfg.source_mode == "webcam" else 2
            if (
                action_backend == "extratrees"
                and cfg.source_mode == "video"
                and not cpu_only
                and not accuracy_first_video
                and fps_src >= 23.0
                and int(effective_process_stride) >= 2
            ):
                effective_action_update_stride = max(effective_action_update_stride, 2)
                self.status_ready.emit("Action update stride primed at 2 for dense 24+ FPS video responsiveness.")
            current_pred_stride = int(getattr(recognizer, "pred_stride", 1))
            if cfg.auto_tune_cpu and cpu_only:
                if action_backend == "torch":
                    min_pred_stride = 6 if cfg.source_mode == "video" else 4
                else:
                    min_pred_stride = 4 if cfg.source_mode == "video" else 2
                if current_pred_stride < min_pred_stride:
                    recognizer.pred_stride = min_pred_stride
                    current_pred_stride = min_pred_stride
                    cpu_auto_tuned = True
            if action_backend == "extratrees" and cfg.source_mode == "video" and not cpu_only:
                # ALWAYS run action prediction every frame for fall detection accuracy.
                # Sparse predictions cause label gaps and jumping - unacceptable for safety-critical action.
                target_pred_stride = 1
                if current_pred_stride != target_pred_stride:
                    recognizer.pred_stride = target_pred_stride
                    current_pred_stride = target_pred_stride
                    self.status_ready.emit(
                        f"Action pred stride set to {target_pred_stride} (every frame) for fall detection accuracy."
                    )
                current_min_frames = int(getattr(recognizer, "min_track_frames", cfg.min_track_frames))
                if int(effective_process_stride) >= 3:
                    target_min_frames = 4
                elif int(effective_process_stride) >= 2:
                    target_min_frames = 5
                else:
                    target_min_frames = 6
                if current_min_frames > target_min_frames:
                    recognizer.min_track_frames = target_min_frames
                    self.status_ready.emit(
                        f"Action min-track-frames lowered to {target_min_frames} for earlier Walking/Standing updates."
                    )
            effective_action_pred_stride = current_pred_stride
            if accuracy_first_video and effective_action_update_stride > 1:
                effective_action_update_stride = 1
                self.status_ready.emit("Action update stride set to 1 for accuracy-first video mode.")
            if cfg.normalize_timing and fps_src > 0:
                target_action_fps = 8.0 if action_backend == "torch" else 10.0
                effective_action_update_stride = max(
                    effective_action_update_stride,
                    int(round(fps_src / target_action_fps)),
                )
            if action_backend == "extratrees" and cfg.source_mode == "video" and not cpu_only and int(effective_process_stride) >= 3:
                # High process stride already drops temporal density; keep action refresh tighter.
                if effective_action_update_stride > 2:
                    effective_action_update_stride = 2
                    self.status_ready.emit("Action update stride capped to 2 for high-stride GPU video responsiveness.")
            if cfg.auto_tune_cpu and cpu_only and cfg.source_mode == "video":
                effective_action_update_stride = max(effective_action_update_stride, 3)
            if action_backend == "extratrees" and cfg.source_mode == "video" and not cpu_only:
                # For video with action recognition, prioritize responsiveness over queue optimization.
                # Sparse updates cause label gaps and missed falls. Keep stride tight for safety.
                if accuracy_first_video:
                    floor_stride = 1 if int(effective_process_stride) <= 1 else 1
                else:
                    # Even for normal mode, cap at stride 2 to avoid action starvation
                    floor_stride = 2
                    if int(effective_process_stride) >= 3:
                        floor_stride = 2  # Don't go higher than 2
                if effective_action_update_stride < floor_stride:
                    effective_action_update_stride = floor_stride
                    self.status_ready.emit(
                        f"Action update stride raised to {floor_stride} to reduce stale async results."
                    )
            if action_backend == "extratrees" and recognizer.et_model is not None:
                action_predictor = AsyncActionPredictor(recognizer.et_model)
                action_predictor.start()
                self.status_ready.emit("Async action inference: enabled")
        output_w = max(1, int(w * cfg.output_scale))
        output_h = max(1, int(h * cfg.output_scale))
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path = None
        writer = None
        if cfg.source_mode == "video" and cfg.save_output_video and w > 0 and h > 0:
            output_path = OUTPUT_DIR / f"{output_stem}_{run_ts}_qt_annotated.mp4"
            preferred_mp4 = output_path
            fallback_avi = output_path.with_suffix(".avi")
            for candidate_path, codecs in [
                (preferred_mp4, ("mp4v", "avc1")),
                (fallback_avi, ("XVID", "MJPG")),
            ]:
                for codec_name in codecs:
                    fourcc = cv2.VideoWriter_fourcc(*codec_name)
                    candidate_writer = cv2.VideoWriter(str(candidate_path), fourcc, fps_src, (output_w, output_h))
                    if candidate_writer.isOpened():
                        writer = candidate_writer
                        output_path = candidate_path
                        break
                    candidate_writer.release()
                if writer is not None:
                    break
            if writer is None:
                output_path = None
                self.status_ready.emit("Annotated video writer unavailable on this system. Continuing without saving output.")

        frame_idx = 0
        t_prev = time.time()
        t_start = t_prev
        fps_ema = fps_src if fps_src > 0 else 30.0
        action_counts = defaultdict(int)
        action_request_count = 0
        action_completed_count = 0
        action_busy_frames = 0
        action_skipped_busy_count = 0
        action_last_lag_ms = 0.0
        action_max_lag_ms = 0.0
        action_stale_result_count = 0
        latest_action_task_id = 0
        latest_action_submitted_at: Optional[float] = None
        latest_action_task_by_track: dict[int, int] = {}
        dynamic_action_update_stride = max(1, int(effective_action_update_stride))
        max_dynamic_action_update_stride = 2
        if cfg.auto_tune_cpu and cpu_only and cfg.source_mode == "video":
            max_dynamic_action_update_stride = 4
        elif int(effective_process_stride) >= 3:
            max_dynamic_action_update_stride = 3
        action_busy_streak = 0
        action_relief_streak = 0
        action_lag_relief_streak = 0
        detection_stats = {"frames_processed": 0, "frames_with_detections": 0, "total_detections": 0}
        reader: Optional[VideoFrameReader] = None
        if cfg.source_mode == "video":
            reader = VideoFrameReader(cap=cap, max_frames=max_frames)
            reader.start()
            self.status_ready.emit("Async video decode: enabled")

        raw_to_stable_id: dict[int, int] = {}
        raw_last_seen: dict[int, int] = {}
        stable_last_bbox: dict[int, np.ndarray] = {}
        stable_last_seen: dict[int, int] = {}
        visual_state_by_id: dict[int, VisualTrackState] = {}
        upright_switch_vote_by_id: dict[int, dict[str, int]] = {}
        unique_stable_ids: set[int] = set()
        next_stable_id = 1
        max_id_idle_frames = max(90, int(round(fps_src * 6.0)))
        track_hold_frames = max(4, int(round(fps_src * 0.60)))
        stable_reid_gap = max(12, int(round(fps_src * 1.50)))
        stable_reid_iou = 0.14
        stable_reid_dist = 0.85
        recent_track_count = 0
        # Scene-cut logic tuned to avoid over-resetting on motion-heavy clips.
        # We still detect hard transitions, but require stronger evidence + debounce.
        scene_stride_factor = max(1.0, float(effective_process_stride))
        # Montage videos often keep similar backgrounds; keep moderately sensitive cut detection
        # while avoiding excessive resets on fast motion segments.
        scene_cut_diff_threshold = 4.0 + 0.35 * (scene_stride_factor - 1.0)
        scene_cut_min_gap = max(12, int(round(fps_src * 0.50)))
        scene_cut_pixel_delta_threshold = 18.0
        scene_cut_pixel_change_ratio = 0.115
        last_scene_cut_frame = -10_000
        prev_scene_signature: Optional[np.ndarray] = None
        prev_scene_diff = 0.0
        scene_cut_reset_count = 0
        track_jump_vote_count = 0
        timeline_records: list[dict[str, object]] = []
        timeline_transitions: list[dict[str, object]] = []
        timeline_label_counts: dict[str, int] = defaultdict(int)
        timeline_last_label_by_tid: dict[int, str] = {}
        timeline_path: Optional[Path] = None

        def resolve_display_id(raw_tid: int, bbox: np.ndarray, assigned_stable_ids: set[int]) -> int:
            nonlocal next_stable_id
            sid = raw_to_stable_id.get(raw_tid)
            if sid is not None:
                return sid

            # Single-subject continuity fast-path:
            # if only one subject is visible, keep most recent stable ID to avoid state resets.
            # Keep this path conservative to avoid carrying labels across quick scene transitions.
            if recent_track_count <= 1 and not assigned_stable_ids and stable_last_seen:
                best_recent_sid = None
                best_recent_gap = 10_000
                for candidate_sid, last_seen in stable_last_seen.items():
                    gap = frame_idx - last_seen
                    if gap < 0:
                        continue
                    if gap < best_recent_gap:
                        best_recent_gap = gap
                        best_recent_sid = candidate_sid
                single_subject_gap_limit = max(6, int(round(fps_src * 0.40)))
                if best_recent_sid is not None and best_recent_gap <= single_subject_gap_limit:
                    candidate_bbox = stable_last_bbox.get(best_recent_sid)
                    if candidate_bbox is not None:
                        iou = bbox_iou_xyxy(bbox, candidate_bbox)
                        dist = bbox_center_distance_norm(bbox, candidate_bbox)
                        candidate_w = max(float(candidate_bbox[2] - candidate_bbox[0]), 1e-3)
                        candidate_h = max(float(candidate_bbox[3] - candidate_bbox[1]), 1e-3)
                        curr_w = max(float(bbox[2] - bbox[0]), 1e-3)
                        curr_h = max(float(bbox[3] - bbox[1]), 1e-3)
                        area_ratio = (curr_w * curr_h) / max(candidate_w * candidate_h, 1e-3)

                        prev_sensitive = False
                        if recognizer is not None:
                            _, _, prev_label_name = recognizer.get_last_prediction(best_recent_sid)
                            prev_sensitive = prev_label_name in {"Fall", "Lying_Down"}

                        likely_scene_jump = area_ratio < 0.62 or area_ratio > 1.62
                        strong_overlap = iou >= 0.08 and 0.60 <= area_ratio <= 1.65
                        close_motion = dist <= 0.45 and 0.72 <= area_ratio <= 1.38
                        if not likely_scene_jump and (strong_overlap or close_motion):
                            if not (prev_sensitive and iou < 0.14 and dist > 0.38):
                                raw_to_stable_id[raw_tid] = best_recent_sid
                                return best_recent_sid

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
                score = iou - 0.22 * dist
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

        self.status_ready.emit("Running inference...")

        while not self._stop_requested:
            if reader is not None:
                packet = reader.read(timeout=1.0)
                if packet is None:
                    break
                frame_idx, frame = packet
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and frame_idx >= max_frames:
                    break

            if action_predictor is not None and recognizer is not None:
                action_result = action_predictor.poll_result()
                if action_result is not None:
                    action_last_lag_ms = (time.perf_counter() - action_result.submitted_at) * 1000.0
                    action_max_lag_ms = max(action_max_lag_ms, action_last_lag_ms)
                    action_completed_count += 1
                    if action_result.task_id == latest_action_task_id:
                        latest_action_submitted_at = None
                    if action_result.error:
                        self.status_ready.emit(f"Async action inference error: {action_result.error}")
                    else:
                        valid_positions = [
                            pos
                            for pos, track_id in enumerate(action_result.track_ids)
                            if latest_action_task_by_track.get(track_id) == action_result.task_id
                        ]
                        dropped_positions = len(action_result.track_ids) - len(valid_positions)
                        action_stale_result_count += max(0, dropped_positions)
                        if valid_positions:
                            valid_track_ids = [action_result.track_ids[pos] for pos in valid_positions]
                            valid_probs = None
                            valid_labels = None
                            if action_result.probs is not None:
                                valid_probs = action_result.probs[valid_positions]
                            if action_result.labels is not None:
                                valid_labels = action_result.labels[valid_positions]
                            recognizer.apply_batch_prediction_results(
                                valid_track_ids,
                                probs=valid_probs,
                                labels=valid_labels,
                                elapsed_ms=action_result.elapsed_ms,
                            )
                            for track_id in valid_track_ids:
                                if latest_action_task_by_track.get(track_id) == action_result.task_id:
                                    latest_action_task_by_track.pop(track_id, None)
                    if action_backend == "extratrees" and cfg.source_mode == "video" and not cpu_only:
                        if action_last_lag_ms >= 240.0 and dynamic_action_update_stride < max_dynamic_action_update_stride:
                            dynamic_action_update_stride += 1
                            action_lag_relief_streak = 0
                            self.status_ready.emit(
                                f"Action lag {action_last_lag_ms:.0f}ms: update stride raised to {dynamic_action_update_stride}."
                            )
                        elif action_last_lag_ms <= 90.0:
                            action_lag_relief_streak += 1
                            if (
                                dynamic_action_update_stride > effective_action_update_stride
                                and action_lag_relief_streak >= 6
                            ):
                                dynamic_action_update_stride -= 1
                                action_lag_relief_streak = 0
                        else:
                            action_lag_relief_streak = 0

            detection_stats["frames_processed"] += 1
            visible_ids: set[int] = set()
            recognizer_active_ids: set[int] = set()
            should_process_frame = frame_idx % effective_process_stride == 0
            action_update_due = frame_idx % max(1, dynamic_action_update_stride) == 0
            preview_emit_due = cfg.live_preview and (frame_idx % max(1, effective_preview_stride) == 0)
            should_draw_this_frame = (writer is not None) or preview_emit_due

            if should_process_frame:
                scene_cut_diff = 0.0
                scene_cut_change_ratio = 0.0
                scene_cut_detected = False
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    scene_signature = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
                except Exception:
                    scene_signature = None
                if scene_signature is not None:
                    if prev_scene_signature is not None:
                        scene_abs_diff = cv2.absdiff(scene_signature, prev_scene_signature)
                        scene_cut_diff = float(np.mean(scene_abs_diff))
                        scene_cut_change_ratio = float(np.mean(scene_abs_diff >= scene_cut_pixel_delta_threshold))
                        cut_like_spike = (
                            scene_cut_diff >= max(scene_cut_diff_threshold, prev_scene_diff * 1.28)
                            and scene_cut_change_ratio >= scene_cut_pixel_change_ratio
                        )
                        hard_cut = (
                            scene_cut_diff >= (scene_cut_diff_threshold * 1.65)
                            and scene_cut_change_ratio >= (scene_cut_pixel_change_ratio * 0.92)
                        )
                        if (cut_like_spike or hard_cut) and (frame_idx - last_scene_cut_frame) >= scene_cut_min_gap:
                            scene_cut_detected = True
                        prev_scene_diff = scene_cut_diff
                    prev_scene_signature = scene_signature

                if scene_cut_detected:
                    last_scene_cut_frame = frame_idx
                    scene_cut_reset_count += 1
                    track_jump_vote_count = 0
                    raw_to_stable_id.clear()
                    raw_last_seen.clear()
                    stable_last_bbox.clear()
                    stable_last_seen.clear()
                    visual_state_by_id.clear()
                    upright_switch_vote_by_id.clear()
                    latest_action_task_by_track.clear()
                    if recognizer is not None:
                        recognizer.remove_stale_tracks(set())
                    self.status_ready.emit(
                        f"Scene cut reset at frame {frame_idx} (diff={scene_cut_diff:.1f})"
                    )

                assigned_stable_ids: set[int] = set()
                results = pose_model.track(
                    frame,
                    persist=True,
                    tracker=tracker_cfg,
                    conf=effective_det_conf,
                    iou=cfg.det_iou,
                    imgsz=effective_pose_imgsz,
                    max_det=effective_max_det,
                    classes=[0],
                    half=torch.cuda.is_available(),
                    verbose=False,
                )
                result = results[0] if results else None
                recent_track_count = 0

                if result is not None and result.boxes is not None and result.boxes.id is not None:
                    detection_stats["frames_with_detections"] += 1
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    bboxes = result.boxes.xyxy.cpu().numpy()
                    recent_track_count = int(len(track_ids))
                    timeline_debug_due = False  # FIX: default timeline debug gate for recognizer-disabled path
                    # Track-jump scene-cut fallback with debounce:
                    # trigger only when overlap collapses for consecutive processed frames.
                    track_jump_candidate = False
                    single_subject_hard_jump = False
                    if (
                        not scene_cut_detected
                        and stable_last_bbox
                        and len(track_ids) >= 1
                        and (frame_idx - last_scene_cut_frame) >= scene_cut_min_gap
                    ):
                        recent_prev_bboxes = [
                            candidate_bbox
                            for candidate_sid, candidate_bbox in stable_last_bbox.items()
                            if (frame_idx - stable_last_seen.get(candidate_sid, -10_000))
                            <= max(4, int(round(effective_process_stride + 2)))
                        ]
                        if len(track_ids) >= 4 and len(recent_prev_bboxes) >= 4:
                            low_overlap = 0
                            far_shift = 0
                            for curr_bbox in bboxes:
                                best_iou = 0.0
                                best_dist = 10.0
                                for prev_bbox in recent_prev_bboxes:
                                    iou_val = bbox_iou_xyxy(curr_bbox, prev_bbox)
                                    dist_val = bbox_center_distance_norm(curr_bbox, prev_bbox)
                                    if iou_val > best_iou:
                                        best_iou = iou_val
                                    if dist_val < best_dist:
                                        best_dist = dist_val
                                if best_iou < 0.02:
                                    low_overlap += 1
                                if best_dist > 0.70:
                                    far_shift += 1
                            low_overlap_ratio = low_overlap / max(len(bboxes), 1)
                            far_shift_ratio = far_shift / max(len(bboxes), 1)
                            track_jump_candidate = (
                                low_overlap_ratio >= 0.92
                                and far_shift_ratio >= 0.82
                                and scene_cut_change_ratio >= (scene_cut_pixel_change_ratio * 0.60)
                            )
                        elif len(track_ids) <= 2 and len(recent_prev_bboxes) >= 1:
                            severe_jump = 0
                            for curr_bbox in bboxes:
                                best_iou = 0.0
                                best_dist = 10.0
                                best_area_ratio = 1.0
                                for prev_bbox in recent_prev_bboxes:
                                    iou_val = bbox_iou_xyxy(curr_bbox, prev_bbox)
                                    dist_val = bbox_center_distance_norm(curr_bbox, prev_bbox)
                                    if iou_val > best_iou:
                                        best_iou = iou_val
                                        prev_w = max(float(prev_bbox[2] - prev_bbox[0]), 1e-3)
                                        prev_h = max(float(prev_bbox[3] - prev_bbox[1]), 1e-3)
                                        curr_w = max(float(curr_bbox[2] - curr_bbox[0]), 1e-3)
                                        curr_h = max(float(curr_bbox[3] - curr_bbox[1]), 1e-3)
                                        best_area_ratio = (curr_w * curr_h) / max(prev_w * prev_h, 1e-3)
                                    if dist_val < best_dist:
                                        best_dist = dist_val
                                abrupt_cut = (
                                    best_iou < 0.05
                                    and best_dist > 0.62
                                    and (
                                        best_area_ratio < 0.58
                                        or best_area_ratio > 1.82
                                        or scene_cut_change_ratio >= (scene_cut_pixel_change_ratio * 0.50)
                                    )
                                )
                                if abrupt_cut:
                                    severe_jump += 1
                            single_subject_hard_jump = severe_jump >= max(1, len(bboxes))
                            if single_subject_hard_jump:
                                track_jump_candidate = True

                    if track_jump_candidate:
                        vote_inc = 2 if single_subject_hard_jump else 1
                        track_jump_vote_count = min(track_jump_vote_count + vote_inc, 3)
                    else:
                        track_jump_vote_count = max(track_jump_vote_count - 1, 0)

                    if (
                        not scene_cut_detected
                        and track_jump_vote_count >= 2
                        and (frame_idx - last_scene_cut_frame) >= scene_cut_min_gap
                    ):
                        scene_cut_detected = True
                        last_scene_cut_frame = frame_idx
                        scene_cut_reset_count += 1
                        track_jump_vote_count = 0
                        raw_to_stable_id.clear()
                        raw_last_seen.clear()
                        stable_last_bbox.clear()
                        stable_last_seen.clear()
                        visual_state_by_id.clear()
                        upright_switch_vote_by_id.clear()
                        latest_action_task_by_track.clear()
                        if recognizer is not None:
                            recognizer.remove_stale_tracks(set())
                        self.status_ready.emit(
                            f"Scene cut reset at frame {frame_idx} (track-jump)"
                        )
                    if recognizer is not None:
                        recognizer.set_active_track_count(len(track_ids))
                        action_fast_mode_used = action_fast_mode_used or recognizer.is_fast_mode_active()
                    pending_prediction_ids: list[int] = []
                    track_entries: list[dict[str, object]] = []
                    if recognizer is not None:  # FIX: compute timeline debug gate only when recognizer is active
                        timeline_debug_due = bool(  # FIX: only collect heavy debug state on preview cadence
                            cfg.source_mode == "video" and frame_idx % max(1, effective_preview_stride) == 0
                        )  # FIX: decouple timeline debug collection from per-frame draw loop

                    for i, tid in enumerate(track_ids):
                        raw_tid = int(tid)
                        bbox = bboxes[i]
                        detection_stats["total_detections"] += 1

                        display_tid = resolve_display_id(raw_tid, bbox, assigned_stable_ids)
                        assigned_stable_ids.add(display_tid)
                        unique_stable_ids.add(display_tid)
                        needs_kpts = cfg.draw_skeleton or recognizer is not None
                        kpts = extract_kpts_for_track(result, raw_tid, frame.shape[1], frame.shape[0]) if needs_kpts else None
                        bbox_norm = np.array(
                            [
                                bbox[0] / max(frame.shape[1], 1),
                                bbox[1] / max(frame.shape[0], 1),
                                bbox[2] / max(frame.shape[1], 1),
                                bbox[3] / max(frame.shape[0], 1),
                            ],
                            dtype=np.float32,
                        )

                        prev_display_bbox = stable_last_bbox.get(display_tid)
                        prev_display_seen = stable_last_seen.get(display_tid, -10_000)
                        if recognizer is not None and prev_display_bbox is not None and frame_idx > prev_display_seen:
                            prev_visual_state = visual_state_by_id.get(display_tid)
                            prev_display_kpts = (
                                prev_visual_state.kpts
                                if prev_visual_state is not None and prev_visual_state.kpts is not None
                                else None
                            )
                            prev_w = max(float(prev_display_bbox[2] - prev_display_bbox[0]), 1e-3)
                            prev_h = max(float(prev_display_bbox[3] - prev_display_bbox[1]), 1e-3)
                            curr_w = max(float(bbox[2] - bbox[0]), 1e-3)
                            curr_h = max(float(bbox[3] - bbox[1]), 1e-3)
                            area_ratio = (curr_w * curr_h) / max(prev_w * prev_h, 1e-3)
                            iou = bbox_iou_xyxy(bbox, prev_display_bbox)
                            dist = bbox_center_distance_norm(bbox, prev_display_bbox)
                            reappear_gap = int(frame_idx - prev_display_seen)
                            _, _, prev_label_name = recognizer.get_last_prediction(display_tid)
                            carryover_sensitive = prev_label_name in {"Fall", "Lying_Down"}
                            carryover_hard_jump = iou < 0.05 and (dist > 0.55 or area_ratio < 0.55 or area_ratio > 1.90)
                            generic_hard_jump = iou < 0.03 and dist > 0.72 and (area_ratio < 0.50 or area_ratio > 2.20)
                            scene_context_jump = scene_cut_change_ratio >= (scene_cut_pixel_change_ratio * 0.45)
                            extreme_jump = dist > 0.95
                            pose_hard_jump = False
                            if prev_display_kpts is not None and kpts is not None:
                                valid_prev = ~np.all(prev_display_kpts == 0, axis=1)
                                valid_curr = ~np.all(kpts == 0, axis=1)
                                valid_joint_mask = valid_prev & valid_curr
                                if int(valid_joint_mask.sum()) >= 6:
                                    joint_delta = np.linalg.norm(
                                        kpts[valid_joint_mask] - prev_display_kpts[valid_joint_mask],
                                        axis=1,
                                    )
                                    median_joint_delta = float(np.median(joint_delta))
                                    p85_joint_delta = float(np.percentile(joint_delta, 85))
                                    pose_hard_jump = bool(
                                        (
                                            median_joint_delta > 0.24
                                            and p85_joint_delta > 0.36
                                            and iou < 0.30
                                        )
                                        or (median_joint_delta > 0.18 and iou < 0.12 and dist > 0.32)
                                        or (
                                            median_joint_delta > 0.16
                                            and reappear_gap >= max(10, int(round(fps_src * 0.40)))
                                            and iou < 0.28
                                        )
                                    )
                            reentry_sensitive_jump = bool(
                                carryover_sensitive
                                and reappear_gap >= max(12, int(round(fps_src * 0.50)))
                                and (iou < 0.20 or dist > 0.60)
                            )
                            reset_on_jump = (
                                (carryover_hard_jump or extreme_jump)
                                if carryover_sensitive
                                else (generic_hard_jump and scene_context_jump)
                            )
                            reset_on_jump = bool(reset_on_jump or pose_hard_jump or reentry_sensitive_jump)
                            if reset_on_jump:
                                recognizer.reset_track(display_tid)
                                latest_action_task_by_track.pop(display_tid, None)
                                visual_state_by_id.pop(display_tid, None)
                                upright_switch_vote_by_id.pop(display_tid, None)
                        raw_last_seen[raw_tid] = frame_idx
                        stable_last_bbox[display_tid] = bbox.copy()
                        stable_last_seen[display_tid] = frame_idx
                        visible_ids.add(display_tid)

                        needs_action_update = recognizer is not None and (
                            action_update_due or recognizer.get_last_prediction(display_tid)[0] < 0
                        )

                        if recognizer is not None:
                            recognizer.update_track(display_tid, kpts, bbox_norm=bbox_norm)
                            if needs_action_update:
                                pending_prediction_ids.append(display_tid)
                        track_entries.append(
                            {
                                "display_tid": display_tid,
                                "bbox": bbox,
                                "kpts": kpts,
                                "needs_action_update": needs_action_update if recognizer is not None else False,
                            }
                        )

                    batch_predictions: dict[int, Tuple[int, float, str]] = {}
                    if recognizer is not None and pending_prediction_ids:
                        if action_predictor is not None:
                            ready_ids, feature_matrix = recognizer.collect_batch_features(pending_prediction_ids)
                            if ready_ids:
                                was_busy = action_predictor.is_busy()
                                if was_busy:
                                    # Backpressure: keep latest in-flight batch, don't flood the queue.
                                    action_busy_frames += 1
                                    action_skipped_busy_count += 1
                                    action_busy_streak += 1
                                    action_relief_streak = 0
                                    action_lag_relief_streak = 0
                                    if (
                                        action_backend == "extratrees"
                                        and cfg.source_mode == "video"
                                        and not cpu_only
                                        and action_busy_streak >= 4
                                        and dynamic_action_update_stride < max_dynamic_action_update_stride
                                    ):
                                        dynamic_action_update_stride += 1

                                    # Avoid prolonged unknown labels while async worker is busy:
                                    # run a tiny synchronous fallback only for cold-start tracks.
                                    cold_start_ids = [
                                        track_id
                                        for track_id in ready_ids
                                        if recognizer.get_last_prediction(track_id)[0] < 0
                                    ]
                                    if cold_start_ids:
                                        fallback_limit = 2 if len(cold_start_ids) > 1 else 1
                                        fallback_ids = cold_start_ids[:fallback_limit]
                                        batch_predictions.update(recognizer.predict_batch(fallback_ids))
                                else:
                                    submitted_task_id = action_predictor.submit(ready_ids, feature_matrix)
                                    if submitted_task_id:
                                        recognizer.mark_predictions_submitted(ready_ids)
                                        action_request_count += 1
                                        latest_action_task_id = max(latest_action_task_id, submitted_task_id)
                                        latest_action_submitted_at = time.perf_counter()
                                        for track_id in ready_ids:
                                            latest_action_task_by_track[track_id] = submitted_task_id
                                        action_busy_streak = 0
                                        action_relief_streak += 1
                                        if (
                                            dynamic_action_update_stride > effective_action_update_stride
                                            and action_relief_streak >= 6
                                        ):
                                            dynamic_action_update_stride -= 1
                                            action_relief_streak = 0
                            else:
                                action_relief_streak += 1
                                action_busy_streak = 0
                                batch_predictions = recognizer.predict_batch(pending_prediction_ids)
                        else:
                            # Torch backend has no async predictor; run synchronous batch/loop prediction.
                            sync_started = time.perf_counter()
                            batch_predictions = recognizer.predict_batch(pending_prediction_ids)
                            sync_elapsed_ms = (time.perf_counter() - sync_started) * 1000.0
                            if batch_predictions:
                                action_request_count += 1
                                action_completed_count += 1
                                action_last_lag_ms = sync_elapsed_ms
                                action_max_lag_ms = max(action_max_lag_ms, sync_elapsed_ms)
                            action_busy_streak = 0
                            action_relief_streak += 1
                            action_lag_relief_streak = 0

                    for entry in track_entries:
                        display_tid = int(entry["display_tid"])
                        bbox = np.asarray(entry["bbox"], dtype=np.float32)
                        kpts = entry["kpts"]

                        if recognizer is not None:
                            if bool(entry["needs_action_update"]):
                                label_id, conf_val, label_name = batch_predictions.get(display_tid, recognizer.get_last_prediction(display_tid))
                            else:
                                label_id, conf_val, label_name = recognizer.get_last_prediction(display_tid)
                            active_label_map = dict(getattr(recognizer, "label_map", {}))

                            prev_state = visual_state_by_id.get(display_tid)
                            if prev_state is not None:
                                prev_label = str(prev_state.label)
                                prev_conf = float(prev_state.conf)

                                # Keep last stable label when current output is unknown to avoid label flicker/dropout.
                                if label_name in ("?", "unknown") and prev_label not in ("?", "unknown", ""):
                                    label_name = prev_label
                                    conf_val = max(prev_conf * 0.90, float(conf_val))
                                    for mapped_id, mapped_name in active_label_map.items():
                                        if mapped_name == prev_label:
                                            label_id = int(mapped_id)
                                            break

                                # Lightweight upright hysteresis at render layer:
                                # require short confirmation before Standing<->Walking switch.
                                if (
                                    prev_label in {"Standing", "Walking"}
                                    and label_name in {"Standing", "Walking"}
                                    and label_name != prev_label
                                ):
                                    votes = upright_switch_vote_by_id.setdefault(
                                        display_tid,
                                        {"Walking": 0, "Standing": 0},
                                    )
                                    votes[label_name] = min(int(votes.get(label_name, 0)) + 1, 6)
                                    votes[prev_label] = 0
                                    if label_name == "Standing":
                                        required_votes = 5 if float(conf_val) < max(prev_conf + 0.12, 0.76) else 4
                                    else:
                                        required_votes = 2 if float(conf_val) < max(prev_conf + 0.05, 0.60) else 1
                                    if votes[label_name] < required_votes:
                                        label_name = prev_label
                                        conf_val = max(prev_conf * 0.92, float(conf_val) * 0.82)
                                        for mapped_id, mapped_name in active_label_map.items():
                                            if mapped_name == prev_label:
                                                label_id = int(mapped_id)
                                                break
                                elif label_name in {"Standing", "Walking"}:
                                    upright_switch_vote_by_id.pop(display_tid, None)
                        else:
                            aspect = (bbox[2] - bbox[0]) / max((bbox[3] - bbox[1]), 1e-6)
                            label_id = 0 if aspect > 1.2 else 2
                            conf_val = 0.50
                            label_name = "Fall" if label_id == 0 else "Walking"

                        if recognizer is not None and timeline_debug_due:  # FIX: gate debug-state/timeline work by preview stride
                            debug_state = recognizer.get_debug_state(display_tid)
                            timeline_label = label_name if label_name not in ("", "unknown") else "?"
                            rec = {
                                "frame": int(frame_idx),
                                "sec": float(frame_idx / max(float(fps_src), 1e-6)),
                                "tid": int(display_tid),
                                "label": str(timeline_label),
                                "conf": float(conf_val),
                                "raw_label_name": str(debug_state.get("raw_label_name", "?")),
                                "raw_conf": float(debug_state.get("raw_confidence", 0.0)),
                                "final_label_name": str(debug_state.get("final_label_name", timeline_label)),
                                "postprocess_mode": str(debug_state.get("postprocess_mode", "")),
                                "postprocess_reason": str(debug_state.get("postprocess_reason", "")),
                                "fall_cue": bool(
                                    debug_state.get("strong_fall_cue", False)
                                    or debug_state.get("moderate_fall_cue", False)
                                    or debug_state.get("lateral_fall_cue", False)
                                ),
                                "fall_vel": bool(debug_state.get("fall_velocity", False)),
                                "fall_hold": int(debug_state.get("pending_fall_frames", 0)),
                                "fall_recovery_votes": int(debug_state.get("fall_recovery_votes", 0)),
                                "down_vel": float(debug_state.get("downward_velocity", 0.0)),
                                "bbox_ar": float(debug_state.get("bbox_aspect_ratio", 1.0)),
                                "edge": bool(debug_state.get("edge_contact", False)),
                                "edge_margin": float(debug_state.get("edge_margin", 1.0)),
                                "resc": str(debug_state.get("rescue_reason", "")),
                            }
                            timeline_records.append(rec)
                            timeline_label_counts[timeline_label] += 1
                            prev_timeline_label = timeline_last_label_by_tid.get(display_tid)
                            if prev_timeline_label != timeline_label:
                                timeline_transitions.append(dict(rec))
                                timeline_last_label_by_tid[display_tid] = timeline_label

                        if label_name not in ("?", "unknown"):
                            action_counts[label_name] += 1

                        color = get_action_color(label_name, label_id)
                        existing_state = visual_state_by_id.get(display_tid)
                        prev_bbox = existing_state.bbox.copy() if existing_state is not None else None
                        prev_kpts = existing_state.kpts.copy() if existing_state is not None and existing_state.kpts is not None else None
                        prev_frame_idx = existing_state.last_frame_idx if existing_state is not None else -1
                        smoothed_bbox = smooth_motion_array(prev_bbox, bbox, alpha=0.68)
                        smoothed_kpts = smooth_motion_array(prev_kpts, kpts, alpha=0.58)
                        visual_state_by_id[display_tid] = VisualTrackState(
                            bbox=smoothed_bbox if smoothed_bbox is not None else bbox.astype(np.float32, copy=True),
                            label=label_name,
                            conf=conf_val,
                            color=color,
                            kpts=smoothed_kpts,
                            prev_bbox=prev_bbox,
                            prev_kpts=prev_kpts,
                            last_frame_idx=frame_idx,
                            prev_frame_idx=prev_frame_idx,
                        )
                        if should_draw_this_frame:
                            draw_bbox, draw_kpts, draw_label, draw_conf, draw_color = resolve_visual_draw_state(
                                visual_state_by_id[display_tid],
                                frame_idx,
                            )
                            if cfg.draw_skeleton and draw_kpts is not None:
                                draw_skeleton(frame, draw_kpts, draw_color, frame.shape[1], frame.shape[0])
                            draw_action_label(frame, draw_bbox, display_tid, draw_label, draw_conf, draw_color)

                stale_raw_ids = [rid for rid, last_seen in raw_last_seen.items() if frame_idx - last_seen > max_id_idle_frames]
                for rid in stale_raw_ids:
                    raw_last_seen.pop(rid, None)
                    raw_to_stable_id.pop(rid, None)

                stale_stable_ids = [sid for sid, last_seen in stable_last_seen.items() if frame_idx - last_seen > max_id_idle_frames]
                for sid in stale_stable_ids:
                    stable_last_seen.pop(sid, None)
                    stable_last_bbox.pop(sid, None)
                    visual_state_by_id.pop(sid, None)
                    upright_switch_vote_by_id.pop(sid, None)

                held_ids = {sid for sid, last_seen in stable_last_seen.items() if frame_idx - last_seen <= track_hold_frames}
                if should_draw_this_frame:
                    for held_tid in sorted(held_ids - visible_ids):
                        held_state = visual_state_by_id.get(held_tid)
                        if held_state is None:
                            continue
                        held_bbox, held_kpts, held_label, held_conf, held_color = resolve_visual_draw_state(
                            held_state,
                            frame_idx,
                        )
                        if cfg.draw_skeleton and held_kpts is not None:
                            draw_skeleton(frame, held_kpts, held_color, frame.shape[1], frame.shape[0])
                        draw_action_label(frame, held_bbox, held_tid, held_label, held_conf, held_color)
                recognizer_active_ids = held_ids.copy()
            else:
                recognizer_active_ids = {sid for sid, last_seen in stable_last_seen.items() if frame_idx - last_seen <= track_hold_frames}
                if should_draw_this_frame:
                    for cached_tid in sorted(recognizer_active_ids):
                        cached_state = visual_state_by_id.get(cached_tid)
                        if cached_state is None:
                            continue
                        cached_bbox, cached_kpts, cached_label, cached_conf, cached_color = resolve_visual_draw_state(
                            cached_state,
                            frame_idx,
                        )
                        if cfg.draw_skeleton and cached_kpts is not None:
                            draw_skeleton(frame, cached_kpts, cached_color, frame.shape[1], frame.shape[0])
                        draw_action_label(frame, cached_bbox, cached_tid, cached_label, cached_conf, cached_color)

            if recognizer is not None:
                recognizer.remove_stale_tracks(recognizer_active_ids)
                stale_pending_track_ids = [track_id for track_id in latest_action_task_by_track if track_id not in recognizer_active_ids]
                for track_id in stale_pending_track_ids:
                    latest_action_task_by_track.pop(track_id, None)

            if writer is not None:
                output_frame = frame
                if cfg.output_scale != 1.0:
                    output_frame = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA)
                writer.write(output_frame)

            if cfg.live_preview and preview_emit_due:
                now = time.time()
                inst_fps = min(240.0, 1.0 / max(now - t_prev, 1e-4))
                fps_ema = 0.92 * fps_ema + 0.08 * inst_fps
                t_prev = now
                qt_img = frame_to_qimage(frame, target_size=(cfg.preview_width, cfg.preview_height))
                should_emit_preview = False
                with self._preview_lock:
                    self._latest_preview_image = qt_img
                    if not self._preview_signal_pending:
                        self._preview_signal_pending = True
                        should_emit_preview = True
                if should_emit_preview:
                    self.frame_ready.emit()
            elif should_process_frame:
                now = time.time()
                inst_fps = min(240.0, 1.0 / max(now - t_prev, 1e-4))
                fps_ema = 0.92 * fps_ema + 0.08 * inst_fps
                t_prev = now

            if frame_idx % max(1, effective_preview_stride) == 0:
                guardrail_debug_html = ""
                if recognizer is not None:
                    debug_ids = sorted(visible_ids if visible_ids else recognizer_active_ids)[:6]
                    if debug_ids:
                        rows = []
                        for debug_tid in debug_ids:
                            debug_state = recognizer.get_debug_state(debug_tid)
                            rows.append(
                                "<tr>"
                                f"<td>{debug_tid}</td>"
                                f"<td>{debug_state.get('label_name', '?')}</td>"
                                f"<td>{debug_state.get('confidence', 0.0):.0%}</td>"
                                f"<td>{debug_state.get('valid_ratio', 0.0):.0%}</td>"
                                f"<td>{debug_state.get('lower_body_ratio', 0.0):.0%}</td>"
                                f"<td>{debug_state.get('jitter_ratio', 0.0):.2f}</td>"
                                f"<td>{debug_state.get('downward_velocity', 0.0):.2f}</td>"
                                f"<td>{debug_state.get('hip_to_ankle_ratio', 0.0):.2f}</td>"
                                f"<td>{'Y' if debug_state.get('fall_velocity', False) else 'N'}</td>"
                                f"<td>{'Y' if debug_state.get('strong_fall_cue', False) else 'N'}</td>"
                                f"<td>{debug_state.get('pending_fall_frames', 0)}</td>"
                                f"<td>{debug_state.get('fall_candidate_votes', 0)}</td>"
                                f"<td>{debug_state.get('fall_recovery_votes', 0)}</td>"
                                f"<td>{debug_state.get('bbox_aspect_ratio', 1.0):.2f}</td>"
                                f"<td>{'Y' if debug_state.get('occluded', False) else 'N'}</td>"
                                f"<td>{'Y' if debug_state.get('noisy', False) else 'N'}</td>"
                                f"<td>{'Y' if debug_state.get('rescue_applied', False) else 'N'}</td>"
                                f"<td>{debug_state.get('pending_sitting_frames', 0)}</td>"
                                f"<td>{debug_state.get('predict_ms', 0.0):.1f}</td>"
                                f"<td>{'Y' if debug_state.get('overload_track_count', False) else 'N'}</td>"
                                f"<td>{'Y' if debug_state.get('over_budget_predict', False) else 'N'}</td>"
                                "</tr>"
                            )
                        guardrail_debug_html = (
                            "<table style='width:100%; border-collapse:collapse;'>"
                            "<thead><tr>"
                                "<th align='left'>ID</th><th align='left'>Label</th><th align='left'>Conf</th>"
                                "<th align='left'>Valid</th><th align='left'>LowerKP</th><th align='left'>Jitter</th><th align='left'>DownVel</th><th align='left'>HipAnk</th>"
                                "<th align='left'>FallVel</th><th align='left'>FallCue</th><th align='left'>FallHold</th><th align='left'>FallCandV</th><th align='left'>FallRecV</th><th align='left'>BBoxAR</th><th align='left'>Occ</th>"
                                "<th align='left'>Noisy</th><th align='left'>Rescue</th><th align='left'>SitHold</th>"
                                "<th align='left'>Pred ms</th><th align='left'>Overload</th><th align='left'>OverBudget</th>"
                                "</tr></thead><tbody>"
                                + "".join(rows)
                                + "</tbody></table>"
                        )
                self.metrics_ready.emit(
                    {
                        "frame": frame_idx,
                        "fps_live_ema": fps_ema,
                        "visible_tracks": len(visible_ids) if should_process_frame else len(recognizer_active_ids),
                        "unique_track_ids": len(unique_stable_ids),
                        "falls": action_counts.get("Fall", 0),
                        "source_fps": fps_src,
                        "effective_det_conf": effective_det_conf,
                        "effective_pose_imgsz": effective_pose_imgsz,
                        "effective_process_stride": effective_process_stride,
                        "effective_preview_stride": effective_preview_stride,
                        "effective_action_update_stride": effective_action_update_stride,
                        "dynamic_action_update_stride": dynamic_action_update_stride,
                        "effective_action_pred_stride": effective_action_pred_stride or 0,
                        "effective_max_det": effective_max_det,
                        "scene_cut_resets": scene_cut_reset_count,
                        "action_skipped_busy_count": action_skipped_busy_count,
                        "action_queue_busy": "Y" if action_predictor is not None and action_predictor.is_busy() else "N",
                        "action_inflight_ms": (
                            (time.perf_counter() - latest_action_submitted_at) * 1000.0
                            if latest_action_submitted_at is not None and action_predictor is not None and action_predictor.is_busy()
                            else 0.0
                        ),
                        "action_max_lag_ms": action_max_lag_ms,
                        "cpu_auto_tuned": cpu_auto_tuned,
                        "processed_frames": detection_stats["frames_processed"],
                        "total_frames": total_frames or max_frames or 0,
                        "guardrail_debug_html": guardrail_debug_html,
                    }
                )

            if reader is None:
                frame_idx += 1

        if reader is not None:
            reader.stop()
            reader.join(timeout=2.0)
        else:
            cap.release()
        if action_predictor is not None:
            action_predictor.stop()
            action_predictor.join(timeout=1.0)
        if writer is not None:
            writer.release()

        elapsed = max(time.time() - t_start, 1e-6)
        fps_avg = frame_idx / elapsed if frame_idx > 0 else 0.0
        if cfg.source_mode == "video" and recognizer is not None:
            device_tag = str(pose_runtime.get("device") or "unknown").replace("/", "_").replace(":", "_")
            timeline_path = OUTPUT_DIR / f"fall_debug_timeline_{device_tag}_{run_ts}.json"
            timeline_payload = {
                "label_counts": dict(timeline_label_counts),
                "records": timeline_records,
                "transitions": timeline_transitions,
            }
            try:
                with timeline_path.open("w", encoding="utf-8") as f:
                    json.dump(timeline_payload, f, ensure_ascii=False)
                self.status_ready.emit(f"Saved fall debug timeline: {timeline_path}")
            except Exception as exc:
                self.status_ready.emit(f"Failed to save fall debug timeline: {exc}")
                timeline_path = None
        return {
            "source_mode": cfg.source_mode,
            "processed_frames": detection_stats["frames_processed"],
            "frames_with_detections": detection_stats["frames_with_detections"],
            "total_detections": detection_stats["total_detections"],
            "unique_track_ids": len(unique_stable_ids),
            "fps": fps_avg,
            "fps_live_ema": fps_ema,
            "elapsed_sec": elapsed,
            "source_fps": fps_src,
            "effective_det_conf": effective_det_conf,
            "effective_pose_imgsz": effective_pose_imgsz,
            "effective_preview_stride": effective_preview_stride,
            "effective_max_det": effective_max_det,
            "scene_cut_resets": scene_cut_reset_count,
            "effective_action_pred_stride": effective_action_pred_stride,
            "effective_process_stride": effective_process_stride,
            "effective_action_update_stride": effective_action_update_stride,
            "dynamic_action_update_stride": dynamic_action_update_stride,
            "effective_target_analysis_fps": effective_target_analysis_fps,
            "pose_backend": pose_runtime.get("backend"),
            "pose_device": pose_runtime.get("device"),
            "pose_weights": pose_runtime.get("weights_path"),
            "action_backend": action_backend,
            "action_fast_mode": action_fast_mode_used,
            "action_queue_busy_frames": action_busy_frames,
            "action_skipped_busy_count": action_skipped_busy_count,
            "action_request_count": action_request_count,
            "action_completed_count": action_completed_count,
            "action_stale_result_count": action_stale_result_count,
            "action_last_lag_ms": action_last_lag_ms,
            "action_max_lag_ms": action_max_lag_ms,
            "cpu_auto_tuned": cpu_auto_tuned,
            "output_path": str(output_path) if output_path else None,
            "fall_debug_timeline_path": str(timeline_path) if timeline_path else None,
            "stopped": self._stop_requested,
            "action_counts": dict(action_counts),
        }


class VideoFrameReader(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture, max_frames: Optional[int], queue_size: int = VIDEO_READER_QUEUE_SIZE):
        super().__init__(daemon=True)
        self.cap = cap
        self.max_frames = max_frames
        self.queue: queue.Queue[Optional[Tuple[int, np.ndarray]]] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.error: Optional[BaseException] = None

    def stop(self) -> None:
        self.stop_event.set()

    def read(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
        while True:
            try:
                item = self.queue.get(timeout=timeout)
            except queue.Empty:
                if not self.is_alive():
                    if self.error is not None:
                        raise RuntimeError("Video reader thread failed.") from self.error
                    return None
                continue

            if item is None:
                if self.error is not None:
                    raise RuntimeError("Video reader thread failed.") from self.error
                return None
            return item

    def run(self) -> None:
        frame_idx = 0
        try:
            while not self.stop_event.is_set():
                if self.max_frames is not None and frame_idx >= max(self.max_frames, 0):
                    break
                ok, frame = self.cap.read()
                if not ok:
                    break
                while not self.stop_event.is_set():
                    try:
                        self.queue.put((frame_idx, frame), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                frame_idx += 1
        except BaseException as exc:
            self.error = exc
        finally:
            self.cap.release()
            while True:
                try:
                    self.queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.worker: Optional[InferenceWorker] = None
        self.last_output_path: Optional[str] = None
        self.setWindowTitle("PyQt6 Fall Detection & Action Recognition")
        self.resize(1560, 920)
        self._build_ui()
        self._apply_profile("balanced")
        self._sync_source_mode()
        self._sync_normalize_timing()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        control_scroll.setWidget(control_widget)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Upload Video", "Webcam"])
        self.source_combo.currentIndexChanged.connect(self._sync_source_mode)

        self.video_path_edit = QLineEdit(str(ROOT / "data" / "video" / "video1.mp4"))
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_video)
        video_row = QHBoxLayout()
        video_row.addWidget(self.video_path_edit)
        video_row.addWidget(browse_btn)
        self.video_row_widget = QWidget()
        self.video_row_widget.setLayout(video_row)

        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 8)
        self.camera_index_spin.setValue(0)
        self.webcam_duration_spin = QSpinBox()
        self.webcam_duration_spin.setRange(3, 300)
        self.webcam_duration_spin.setValue(10)
        self.camera_row_widget = QWidget()
        camera_row = QHBoxLayout()
        camera_row.setContentsMargins(0, 0, 0, 0)
        camera_row.addWidget(self.camera_index_spin)
        camera_row.addWidget(QLabel("Duration (sec)"))
        camera_row.addWidget(self.webcam_duration_spin)
        self.camera_row_widget.setLayout(camera_row)

        source_group = QGroupBox("Input")
        source_form = QFormLayout(source_group)
        source_form.addRow("Source", self.source_combo)
        source_form.addRow("Video Path", self.video_row_widget)
        source_form.addRow("Camera Index", self.camera_row_widget)
        control_layout.addWidget(source_group)

        self.pose_weights_edit = QLineEdit(resolve_default_pose_weights_path())
        self.action_model_edit = QLineEdit(resolve_default_action_model_path())
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["BoT-SORT (custom)", "ByteTrack (custom)", "BoT-SORT (default)", "ByteTrack (default)"])
        self.tracker_combo.setCurrentText("BoT-SORT (custom)")

        self.det_conf_spin = self._make_double_spin(0.05, 0.95, 0.30, 0.01)
        self.det_iou_spin = self._make_double_spin(0.10, 0.95, 0.50, 0.01)
        self.imgsz_combo = QComboBox()
        for value in [320, 480, 640, 960, 1280]:
            self.imgsz_combo.addItem(str(value), value)
        self.imgsz_combo.setCurrentText("640")
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 200)
        self.max_det_spin.setValue(12)

        model_group = QGroupBox("Models and Tracking")
        model_form = QFormLayout(model_group)
        model_form.addRow("YOLO Pose Weights", self.pose_weights_edit)
        model_form.addRow("Action Model", self.action_model_edit)
        model_form.addRow("Tracker", self.tracker_combo)
        model_form.addRow("Detection Confidence", self.det_conf_spin)
        model_form.addRow("NMS IoU", self.det_iou_spin)
        model_form.addRow("Image Size", self.imgsz_combo)
        model_form.addRow("Max Persons", self.max_det_spin)
        control_layout.addWidget(model_group)

        self.live_preview_checkbox = QCheckBox("Enable live preview")
        self.live_preview_checkbox.setChecked(True)
        self.draw_skeleton_checkbox = QCheckBox("Draw skeleton")
        self.skip_action_checkbox = QCheckBox("Skip action recognition model")
        self.normalize_timing_checkbox = QCheckBox("Normalize timing across videos")
        self.normalize_timing_checkbox.setChecked(True)
        self.normalize_timing_checkbox.toggled.connect(self._sync_normalize_timing)
        self.auto_tune_cpu_checkbox = QCheckBox("CPU auto-tune")
        self.auto_tune_cpu_checkbox.setChecked(True)
        self.process_stride_spin = QSpinBox()
        self.process_stride_spin.setRange(1, 5)
        self.process_stride_spin.setValue(1)
        self.preview_stride_spin = QSpinBox()
        self.preview_stride_spin.setRange(1, 15)
        self.preview_stride_spin.setValue(4)
        self.output_scale_combo = QComboBox()
        for value in [0.5, 0.75, 1.0]:
            self.output_scale_combo.addItem(str(value), value)
        self.output_scale_combo.setCurrentText("1.0")
        self.save_output_checkbox = QCheckBox("Save annotated output video")
        self.save_output_checkbox.setChecked(False)
        self.target_analysis_fps_spin = self._make_double_spin(6.0, 20.0, 12.0, 1.0)

        perf_group = QGroupBox("Performance")
        perf_form = QFormLayout(perf_group)
        perf_form.addRow(self.live_preview_checkbox)
        perf_form.addRow(self.draw_skeleton_checkbox)
        perf_form.addRow(self.skip_action_checkbox)
        perf_form.addRow(self.normalize_timing_checkbox)
        perf_form.addRow(self.auto_tune_cpu_checkbox)
        perf_form.addRow("Process Stride", self.process_stride_spin)
        perf_form.addRow("Preview Stride", self.preview_stride_spin)
        perf_form.addRow("Output Scale", self.output_scale_combo)
        perf_form.addRow(self.save_output_checkbox)
        perf_form.addRow("Target Analysis FPS", self.target_analysis_fps_spin)
        control_layout.addWidget(perf_group)

        self.min_track_frames_spin = QSpinBox()
        self.min_track_frames_spin.setRange(1, 128)
        self.min_track_frames_spin.setValue(12)
        self.pred_stride_spin = QSpinBox()
        self.pred_stride_spin.setRange(1, 16)
        self.pred_stride_spin.setValue(3)
        self.action_conf_spin = self._make_double_spin(0.01, 0.95, 0.30, 0.01)
        self.smooth_window_spin = QSpinBox()
        self.smooth_window_spin.setRange(1, 7)
        self.smooth_window_spin.setValue(3)
        self.fall_conf_boost_spin = self._make_double_spin(0.00, 0.30, 0.10, 0.01)
        self.sitting_conf_penalty_spin = self._make_double_spin(0.00, 0.40, 0.20, 0.01)
        self.keypoint_integrity_spin = self._make_double_spin(0.50, 0.95, 0.70, 0.01)
        self.keypoint_jitter_spin = self._make_double_spin(0.05, 0.40, 0.15, 0.01)
        self.fall_priority_prob_spin = self._make_double_spin(0.20, 0.80, 0.40, 0.01)
        self.fall_velocity_ratio_spin = self._make_double_spin(0.05, 0.30, 0.12, 0.01)
        self.sitting_hold_frames_spin = QSpinBox()
        self.sitting_hold_frames_spin.setRange(1, 12)
        self.sitting_hold_frames_spin.setValue(5)
        self.track_time_budget_spin = self._make_double_spin(2.0, 30.0, 10.0, 0.5)
        self.fast_track_threshold_spin = QSpinBox()
        self.fast_track_threshold_spin.setRange(1, 20)
        self.fast_track_threshold_spin.setValue(5)

        action_group = QGroupBox("Action Recognition")
        action_form = QFormLayout(action_group)
        action_form.addRow("Min Track Frames", self.min_track_frames_spin)
        action_form.addRow("Prediction Stride", self.pred_stride_spin)
        action_form.addRow("Action Confidence", self.action_conf_spin)
        action_form.addRow("Smoothing Window", self.smooth_window_spin)
        action_form.addRow("Fast Fall Sensitivity", self.fall_conf_boost_spin)
        action_form.addRow("Sitting Strictness", self.sitting_conf_penalty_spin)
        action_form.addRow("Keypoint Integrity", self.keypoint_integrity_spin)
        action_form.addRow("Keypoint Jitter Limit", self.keypoint_jitter_spin)
        action_form.addRow("Fall Priority Prob", self.fall_priority_prob_spin)
        action_form.addRow("Fall Velocity Limit", self.fall_velocity_ratio_spin)
        action_form.addRow("Sitting Hold Frames", self.sitting_hold_frames_spin)
        action_form.addRow("Track Budget ms", self.track_time_budget_spin)
        action_form.addRow("Fast Mode ID Limit", self.fast_track_threshold_spin)
        control_layout.addWidget(action_group)

        preset_group = QGroupBox("Profiles")
        preset_row = QHBoxLayout(preset_group)
        balanced_btn = QPushButton("RTX 3050 Balanced")
        fast_btn = QPushButton("Fast Mode")
        quality_btn = QPushButton("Quality Mode")
        balanced_btn.clicked.connect(lambda: self._apply_profile("balanced"))
        fast_btn.clicked.connect(lambda: self._apply_profile("fast"))
        quality_btn.clicked.connect(lambda: self._apply_profile("quality"))
        preset_row.addWidget(balanced_btn)
        preset_row.addWidget(fast_btn)
        preset_row.addWidget(quality_btn)
        control_layout.addWidget(preset_group)

        button_row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.open_output_btn = QPushButton("Open Output")
        self.stop_btn.setEnabled(False)
        self.open_output_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_run)
        self.stop_btn.clicked.connect(self._stop_run)
        self.open_output_btn.clicked.connect(self._open_output)
        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)
        button_row.addWidget(self.open_output_btn)
        control_layout.addLayout(button_row)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.preview_label = QLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(960, 540)
        self.preview_label.setStyleSheet("background-color: #101418; color: #d9e2ec; border: 1px solid #23323f;")
        right_layout.addWidget(self.preview_label, stretch=5)

        metrics_group = QGroupBox("Metrics")
        metrics_layout = QGridLayout(metrics_group)
        self.metric_labels: dict[str, QLabel] = {}
        metric_names = [
            ("frame", "Frame"),
            ("fps_live_ema", "Live FPS"),
            ("visible_tracks", "Visible Tracks"),
            ("unique_track_ids", "Unique Track IDs"),
            ("falls", "Falls"),
            ("source_fps", "Source FPS"),
            ("effective_pose_imgsz", "Pose ImgSz"),
            ("effective_process_stride", "Process Stride"),
            ("effective_preview_stride", "Preview Stride"),
            ("effective_max_det", "Effective MaxDet"),
            ("effective_action_pred_stride", "Action Pred Stride"),
            ("effective_action_update_stride", "Action Update Stride"),
            ("dynamic_action_update_stride", "Action Update Dyn"),
            ("scene_cut_resets", "Scene Cut Resets"),
            ("action_queue_busy", "Action Busy"),
            ("action_inflight_ms", "Action Lag ms"),
            ("action_max_lag_ms", "Max Action Lag"),
        ]
        for index, (key, title) in enumerate(metric_names):
            title_label = QLabel(title)
            value_label = QLabel("-")
            value_label.setStyleSheet("font-weight: 600;")
            metrics_layout.addWidget(title_label, index // 2, (index % 2) * 2)
            metrics_layout.addWidget(value_label, index // 2, (index % 2) * 2 + 1)
            self.metric_labels[key] = value_label
        right_layout.addWidget(metrics_group, stretch=1)

        self.summary_browser = QTextBrowser()
        self.summary_browser.setOpenExternalLinks(False)
        self.summary_browser.setPlaceholderText("Run summary will appear here.")
        right_layout.addWidget(self.summary_browser, stretch=2)

        self.guardrail_browser = QTextBrowser()
        self.guardrail_browser.setOpenExternalLinks(False)
        self.guardrail_browser.setPlaceholderText("Live guardrail diagnostics will appear here.")
        right_layout.addWidget(self.guardrail_browser, stretch=2)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        right_layout.addWidget(self.log_output, stretch=2)

        splitter.addWidget(control_scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([420, 1140])

    def _make_double_spin(self, minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(2 if step < 1 else 1)
        return spin

    def _sync_source_mode(self) -> None:
        is_video = self.source_combo.currentText() == "Upload Video"
        self.video_row_widget.setVisible(is_video)
        self.camera_row_widget.setVisible(not is_video)
        self.save_output_checkbox.setEnabled(is_video)

    def _sync_normalize_timing(self) -> None:
        self.target_analysis_fps_spin.setEnabled(self.normalize_timing_checkbox.isChecked())

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Input Video",
            str(ROOT),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)",
        )
        if path:
            self.video_path_edit.setText(path)

    def _build_config(self) -> RuntimeConfig:
        return RuntimeConfig(
            source_mode="video" if self.source_combo.currentText() == "Upload Video" else "webcam",
            video_path=self.video_path_edit.text().strip(),
            camera_index=self.camera_index_spin.value(),
            pose_weights=self.pose_weights_edit.text().strip(),
            action_model_path=self.action_model_edit.text().strip(),
            tracker_name=self.tracker_combo.currentText(),
            det_conf=float(self.det_conf_spin.value()),
            det_iou=float(self.det_iou_spin.value()),
            imgsz=int(self.imgsz_combo.currentData()),
            max_det=int(self.max_det_spin.value()),
            draw_skeleton=self.draw_skeleton_checkbox.isChecked(),
            live_preview=self.live_preview_checkbox.isChecked(),
            preview_width=max(self.preview_label.width(), self.preview_label.minimumWidth(), DEFAULT_PREVIEW_WIDTH),
            preview_height=max(self.preview_label.height(), self.preview_label.minimumHeight(), DEFAULT_PREVIEW_HEIGHT),
            webcam_duration_sec=int(self.webcam_duration_spin.value()),
            preview_stride=int(self.preview_stride_spin.value()),
            process_stride=int(self.process_stride_spin.value()),
            output_scale=float(self.output_scale_combo.currentData()),
            save_output_video=self.save_output_checkbox.isChecked(),
            skip_action_model=self.skip_action_checkbox.isChecked(),
            normalize_timing=self.normalize_timing_checkbox.isChecked(),
            auto_tune_cpu=self.auto_tune_cpu_checkbox.isChecked(),
            target_analysis_fps=float(self.target_analysis_fps_spin.value()),
            min_track_frames=int(self.min_track_frames_spin.value()),
            pred_stride=int(self.pred_stride_spin.value()),
            action_conf=float(self.action_conf_spin.value()),
            smooth_window=int(self.smooth_window_spin.value()),
            fall_conf_boost=float(self.fall_conf_boost_spin.value()),
            sitting_conf_penalty=float(self.sitting_conf_penalty_spin.value()),
            keypoint_integrity_ratio=float(self.keypoint_integrity_spin.value()),
            keypoint_jitter_ratio=float(self.keypoint_jitter_spin.value()),
            fall_priority_prob=float(self.fall_priority_prob_spin.value()),
            fall_velocity_ratio=float(self.fall_velocity_ratio_spin.value()),
            sitting_hold_frames=int(self.sitting_hold_frames_spin.value()),
            track_time_budget_ms=float(self.track_time_budget_spin.value()),
            fast_track_threshold=int(self.fast_track_threshold_spin.value()),
        )

    def _set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if running:
            self.open_output_btn.setEnabled(False)

    def _apply_profile(self, profile_name: str) -> None:
        if profile_name == "balanced":
            self.tracker_combo.setCurrentText("BoT-SORT (custom)")
            self.det_conf_spin.setValue(0.30)
            self.det_iou_spin.setValue(0.50)
            self.imgsz_combo.setCurrentText("640")
            self.max_det_spin.setValue(12)
            self.live_preview_checkbox.setChecked(True)
            self.process_stride_spin.setValue(2)
            self.preview_stride_spin.setValue(3)
            self.output_scale_combo.setCurrentText("1.0")
            self.save_output_checkbox.setChecked(False)
            self.normalize_timing_checkbox.setChecked(False)
            self.target_analysis_fps_spin.setValue(12.0)
            self.pred_stride_spin.setValue(2)
            self.min_track_frames_spin.setValue(5)
            self.action_conf_spin.setValue(0.30)
            self.smooth_window_spin.setValue(3)
            self.fall_conf_boost_spin.setValue(0.08)
            self.sitting_conf_penalty_spin.setValue(0.16)
            self.keypoint_integrity_spin.setValue(0.70)
            self.keypoint_jitter_spin.setValue(0.15)
            self.fall_priority_prob_spin.setValue(0.44)
            self.fall_velocity_ratio_spin.setValue(0.12)
            self.sitting_hold_frames_spin.setValue(5)
            self.track_time_budget_spin.setValue(9.0)
            self.fast_track_threshold_spin.setValue(6)
            self.auto_tune_cpu_checkbox.setChecked(False)
        elif profile_name == "fast":
            self.tracker_combo.setCurrentText("ByteTrack (custom)")
            self.det_conf_spin.setValue(0.30)
            self.det_iou_spin.setValue(0.45)
            self.imgsz_combo.setCurrentText("480")
            self.max_det_spin.setValue(12)
            self.live_preview_checkbox.setChecked(True)
            self.process_stride_spin.setValue(2)
            self.preview_stride_spin.setValue(3)  # FIX: minimum preview stride 3 for fast mode responsiveness
            self.output_scale_combo.setCurrentText("0.75")
            self.save_output_checkbox.setChecked(False)
            self.normalize_timing_checkbox.setChecked(False)
            self.target_analysis_fps_spin.setValue(12.0)
            self.pred_stride_spin.setValue(1)
            self.min_track_frames_spin.setValue(6)  # Short fall clips need labels before the person has already hit the ground.
            self.action_conf_spin.setValue(0.31)
            self.smooth_window_spin.setValue(3)
            self.fall_conf_boost_spin.setValue(0.08)
            self.sitting_conf_penalty_spin.setValue(0.05)  # FIX: allow valid sitting detections
            self.keypoint_integrity_spin.setValue(0.68)
            self.keypoint_jitter_spin.setValue(0.18)
            self.fall_priority_prob_spin.setValue(0.32)  # FIX: lower fall priority threshold for faster fall capture
            self.fall_velocity_ratio_spin.setValue(0.10)
            self.sitting_hold_frames_spin.setValue(4)
            self.track_time_budget_spin.setValue(8.0)
            self.fast_track_threshold_spin.setValue(5)
            self.auto_tune_cpu_checkbox.setChecked(False)
        else:
            self.tracker_combo.setCurrentText("BoT-SORT (custom)")
            pose_pt = ROOT / "yolov8n-pose.pt"
            if pose_pt.exists():
                self.pose_weights_edit.setText(str(pose_pt))
            self.det_conf_spin.setValue(0.25)
            self.det_iou_spin.setValue(0.50)
            self.imgsz_combo.setCurrentText("640")  # FIX: was 960; reduce YOLO inference cost on GPU
            self.max_det_spin.setValue(16)
            self.live_preview_checkbox.setChecked(True)
            self.process_stride_spin.setValue(1)
            self.preview_stride_spin.setValue(2)  # FIX: give pipeline breathing room while keeping smooth preview
            self.output_scale_combo.setCurrentText("1.0")
            self.save_output_checkbox.setChecked(True)
            self.normalize_timing_checkbox.setChecked(False)
            self.target_analysis_fps_spin.setValue(15.0)  # FIX: increase responsiveness target in accuracy-first GPU mode
            self.pred_stride_spin.setValue(2)
            self.min_track_frames_spin.setValue(30)  # FIX: require more frames for stable action sequence context
            self.action_conf_spin.setValue(0.30)
            self.smooth_window_spin.setValue(2)
            self.fall_conf_boost_spin.setValue(0.10)
            self.sitting_conf_penalty_spin.setValue(0.05)  # FIX: reduce sitting suppression in accuracy-first profile
            self.keypoint_integrity_spin.setValue(0.68)
            self.keypoint_jitter_spin.setValue(0.18)
            self.fall_priority_prob_spin.setValue(0.32)  # FIX: align fall priority with recommended runtime value
            self.fall_velocity_ratio_spin.setValue(0.12)
            self.sitting_hold_frames_spin.setValue(5)
            self.track_time_budget_spin.setValue(12.0)
            self.fast_track_threshold_spin.setValue(6)
            self.auto_tune_cpu_checkbox.setChecked(False)
        self._append_log(f"Applied profile: {profile_name}")

    def _open_output(self) -> None:
        if not self.last_output_path:
            return
        output_path = Path(self.last_output_path)
        target = output_path if output_path.exists() else output_path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _start_run(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        config = self._build_config()
        if not Path(config.pose_weights).exists():
            QMessageBox.critical(self, "Missing Weights", f"YOLO pose weights not found:\n{config.pose_weights}")
            return
        if config.source_mode == "video" and not Path(config.video_path).exists():
            QMessageBox.critical(self, "Missing Video", f"Input video not found:\n{config.video_path}")
            return

        self.preview_label.setText("Starting...")
        self.log_output.clear()
        self.summary_browser.clear()
        self.guardrail_browser.clear()
        self.last_output_path = None
        self._append_log("PyQt6 desktop run started.")

        self.worker = InferenceWorker(config)
        self.worker.frame_ready.connect(self._update_preview_frame)
        self.worker.metrics_ready.connect(self._update_metrics)
        self.worker.status_ready.connect(self._append_log)
        self.worker.finished_ready.connect(self._on_finished)
        self.worker.error_ready.connect(self._on_error)
        self.worker.finished.connect(self._cleanup_worker)
        self._set_running(True)
        self.worker.start()

    def _stop_run(self) -> None:
        if self.worker is None:
            return
        self._append_log("Stop requested...")
        self.worker.stop()
        self.stop_btn.setEnabled(False)

    def _update_preview_frame(self) -> None:
        worker = self.worker
        if worker is None:
            return
        image = worker.take_latest_preview()
        if image is None:
            return
        self.preview_label.setPixmap(QPixmap.fromImage(image))

    def _update_metrics(self, metrics: dict) -> None:
        guardrail_debug_html = metrics.get("guardrail_debug_html")
        if guardrail_debug_html:
            self.guardrail_browser.setHtml(guardrail_debug_html)
        for key, value in metrics.items():
            label = self.metric_labels.get(key)
            if label is None:
                continue
            if isinstance(value, float):
                label.setText(f"{value:.1f}")
            else:
                label.setText(str(value))

    def _append_log(self, message: str) -> None:
        self.log_output.append(message)

    def _on_finished(self, summary: dict) -> None:
        self._set_running(False)
        self._append_log("Run finished.")
        self.last_output_path = summary.get("output_path")
        self.open_output_btn.setEnabled(bool(self.last_output_path))
        if summary.get("output_path"):
            self._append_log(f"Output video: {summary['output_path']}")
        if summary.get("fall_debug_timeline_path"):
            self._append_log(f"Fall debug timeline: {summary['fall_debug_timeline_path']}")
        self._append_log(
            f"FPS={summary.get('fps', 0):.1f} | "
            f"Unique Track IDs={summary.get('unique_track_ids', 0)} | "
            f"Source FPS={summary.get('source_fps', 0):.1f}"
        )
        action_lines = []
        for action, count in sorted(summary.get("action_counts", {}).items(), key=lambda item: item[1], reverse=True):
            action_lines.append(f"<li><b>{action}</b>: {count}</li>")
        action_html = "<ul>" + "".join(action_lines) + "</ul>" if action_lines else "<p>No actions detected.</p>"
        self.summary_browser.setHtml(
            f"""
            <h3>Run Summary</h3>
            <p><b>Average FPS:</b> {summary.get('fps', 0):.1f}<br>
            <b>Live FPS EMA:</b> {summary.get('fps_live_ema', 0):.1f}<br>
            <b>Unique Track IDs:</b> {summary.get('unique_track_ids', 0)}<br>
            <b>Total Detections:</b> {summary.get('total_detections', 0)}<br>
            <b>Frames With Detections:</b> {summary.get('frames_with_detections', 0)}<br>
            <b>Source FPS:</b> {summary.get('source_fps', 0):.1f}<br>
            <b>Effective Det Conf:</b> {summary.get('effective_det_conf', 0):.2f}<br>
            <b>Effective Pose ImgSz:</b> {summary.get('effective_pose_imgsz', 0)}<br>
            <b>Effective Process Stride:</b> {summary.get('effective_process_stride', 0)}<br>
            <b>Effective Preview Stride:</b> {summary.get('effective_preview_stride', 0)}<br>
            <b>Effective Action Pred Stride:</b> {summary.get('effective_action_pred_stride', 0) or 0}<br>
            <b>Effective Action Update Stride:</b> {summary.get('effective_action_update_stride', 0)}<br>
            <b>Dynamic Action Update Stride:</b> {summary.get('dynamic_action_update_stride', summary.get('effective_action_update_stride', 0))}<br>
            <b>Effective MaxDet:</b> {summary.get('effective_max_det', 0)}<br>
            <b>Scene Cut Resets:</b> {summary.get('scene_cut_resets', 0)}<br>
            <b>Action Queue Busy Frames:</b> {summary.get('action_queue_busy_frames', 0)}<br>
            <b>Action Busy Skipped:</b> {summary.get('action_skipped_busy_count', 0)}<br>
            <b>Action Requests:</b> {summary.get('action_request_count', 0)}<br>
            <b>Action Completed:</b> {summary.get('action_completed_count', 0)}<br>
            <b>Action Stale Results Dropped:</b> {summary.get('action_stale_result_count', 0)}<br>
            <b>Last Action Lag ms:</b> {summary.get('action_last_lag_ms', 0):.1f}<br>
            <b>Max Action Lag ms:</b> {summary.get('action_max_lag_ms', 0):.1f}<br>
            <b>CPU Auto-Tuned:</b> {summary.get('cpu_auto_tuned', False)}<br>
            <b>Pose Backend:</b> {summary.get('pose_backend') or 'unknown'}<br>
            <b>Pose Device:</b> {summary.get('pose_device') or 'unknown'}<br>
            <b>Pose Weights:</b> {summary.get('pose_weights') or 'unknown'}<br>
            <b>Action Backend:</b> {summary.get('action_backend') or 'disabled'}<br>
            <b>Action Fast Seq Mode (CPU):</b> {summary.get('action_fast_mode', False)}<br>
            <b>Output:</b> {summary.get('output_path') or 'No file saved'}<br>
            <b>Fall Debug Timeline:</b> {summary.get('fall_debug_timeline_path') or 'No debug timeline saved'}</p>
            <h4>Action Counts</h4>
            {action_html}
            """
        )
        QMessageBox.information(
            self,
            "Processing Complete",
            f"Run complete.\n\n"
            f"Average FPS: {summary.get('fps', 0):.1f}\n"
            f"Unique Track IDs: {summary.get('unique_track_ids', 0)}\n"
            f"Output: {summary.get('output_path') or 'No file saved'}",
        )

    def _on_error(self, message: str) -> None:
        self._set_running(False)
        self.open_output_btn.setEnabled(bool(self.last_output_path))
        self._append_log(f"ERROR: {message}")
        QMessageBox.critical(self, "PyQt6 App Error", message)

    def _cleanup_worker(self) -> None:
        worker = self.worker
        if worker is None:
            return
        if worker.isRunning():
            QTimer.singleShot(50, self._cleanup_worker)
            return
        worker.deleteLater()
        self.worker = None

    def closeEvent(self, event: QCloseEvent) -> None:
        worker = self.worker
        if worker is not None and worker.isRunning():
            self._append_log("Waiting for background worker to stop before closing...")
            worker.stop()
            if not worker.wait(10000):
                QMessageBox.warning(
                    self,
                    "Processing In Progress",
                    "The background inference thread is still shutting down. Please wait a moment and try closing again.",
                )
                event.ignore()
                return
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
