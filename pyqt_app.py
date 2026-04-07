from __future__ import annotations

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
    LABEL_COLORS,
    LABEL_MAP,
    ROOT,
    describe_pose_runtime,
    bbox_center_distance_norm,
    bbox_iou_xyxy,
    draw_action_label,
    draw_skeleton,
    extract_kpts_for_track,
    load_pose_model,
    resolve_default_action_model_path,
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


def frame_to_qimage(frame_bgr: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> QImage:
    if target_size is not None:
        target_w = max(1, int(target_size[0]))
        target_h = max(1, int(target_size[1]))
        src_h, src_w = frame_bgr.shape[:2]
        scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1), 1.0)
        if scale < 0.999:
            resized_w = max(1, int(round(src_w * scale)))
            resized_h = max(1, int(round(src_h * scale)))
            frame_bgr = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return image.copy()


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

    def _store_latest_preview(self, image: QImage) -> None:
        should_emit = False
        with self._preview_lock:
            self._latest_preview_image = image
            if not self._preview_signal_pending:
                self._preview_signal_pending = True
                should_emit = True
        if should_emit:
            self.frame_ready.emit()

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
        self.status_ready.emit(
            f"Pose runtime: {pose_runtime.get('backend', 'unknown')} on {pose_runtime.get('device', 'unknown')}"
        )

        self.status_ready.emit("Loading action model...")
        recognizer = self._build_recognizer()

        tracker_cfg = resolve_tracker_config(cfg.tracker_name)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cpu_only = not torch.cuda.is_available()

        effective_process_stride = max(1, cfg.process_stride)
        effective_preview_stride = max(1, cfg.preview_stride)
        effective_max_det = max(1, cfg.max_det)
        effective_target_analysis_fps = max(0.0, cfg.target_analysis_fps)
        cpu_auto_tuned = False
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

        effective_action_update_stride = 1
        action_backend = None
        action_fast_mode = False
        effective_action_pred_stride = None
        if recognizer is not None:
            action_backend = getattr(recognizer, "backend", "unknown")
            if action_backend == "extratrees":
                action_fast_mode = bool(cfg.auto_tune_cpu and cpu_only and cfg.source_mode == "video")
                recognizer.fast_mode = action_fast_mode
            if action_backend == "torch":
                effective_action_update_stride = 2 if cfg.source_mode == "webcam" else 3
            else:
                effective_action_update_stride = 1 if cfg.source_mode == "webcam" else 2
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
            effective_action_pred_stride = current_pred_stride
            if cfg.normalize_timing and fps_src > 0:
                target_action_fps = 8.0 if action_backend == "torch" else 10.0
                effective_action_update_stride = max(
                    effective_action_update_stride,
                    int(round(fps_src / target_action_fps)),
                )
            if cfg.auto_tune_cpu and cpu_only and cfg.source_mode == "video":
                effective_action_update_stride = max(effective_action_update_stride, 3)

        output_w = max(1, int(w * cfg.output_scale))
        output_h = max(1, int(h * cfg.output_scale))

        output_path = None
        writer = None
        if cfg.source_mode == "video" and cfg.save_output_video and w > 0 and h > 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"{output_stem}_{ts}_qt_annotated.mp4"
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
        unique_stable_ids: set[int] = set()
        next_stable_id = 1
        max_id_idle_frames = max(90, int(round(fps_src * 6.0)))
        track_hold_frames = max(3, int(round(fps_src * 0.35)))
        stable_reid_gap = max(8, int(round(fps_src * 0.75)))
        stable_reid_iou = 0.20
        stable_reid_dist = 0.75

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

            detection_stats["frames_processed"] += 1
            visible_ids: set[int] = set()
            recognizer_active_ids: set[int] = set()
            should_process_frame = frame_idx % effective_process_stride == 0
            action_update_due = frame_idx % max(1, effective_action_update_stride) == 0
            preview_emit_due = cfg.live_preview
            should_draw_this_frame = (writer is not None) or preview_emit_due

            if should_process_frame:
                assigned_stable_ids: set[int] = set()
                results = pose_model.track(
                    frame,
                    persist=True,
                    tracker=tracker_cfg,
                    conf=cfg.det_conf,
                    iou=cfg.det_iou,
                    imgsz=cfg.imgsz,
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
                        bbox = bboxes[i]
                        detection_stats["total_detections"] += 1

                        display_tid = resolve_display_id(raw_tid, bbox, assigned_stable_ids)
                        assigned_stable_ids.add(display_tid)
                        unique_stable_ids.add(display_tid)
                        raw_last_seen[raw_tid] = frame_idx
                        stable_last_bbox[display_tid] = bbox.copy()
                        stable_last_seen[display_tid] = frame_idx
                        visible_ids.add(display_tid)

                        needs_action_update = (
                            recognizer is not None
                            and (action_update_due or display_tid not in recognizer._buffers)
                        )
                        needs_kpts = cfg.draw_skeleton or needs_action_update
                        kpts = extract_kpts_for_track(result, raw_tid, frame.shape[1], frame.shape[0]) if needs_kpts else None

                        if recognizer is not None:
                            if needs_action_update:
                                recognizer.update_track(display_tid, kpts)
                                label_id, conf_val, label_name = recognizer.predict(display_tid)
                            else:
                                label_id, conf_val, label_name = recognizer.get_last_prediction(display_tid)
                        else:
                            aspect = (bbox[2] - bbox[0]) / max((bbox[3] - bbox[1]), 1e-6)
                            label_id = 0 if aspect > 1.2 else 1
                            conf_val = 0.50
                            label_name = LABEL_MAP.get(label_id, "Walking")

                        if label_name not in ("?", "unknown"):
                            action_counts[label_name] += 1

                        color = LABEL_COLORS.get(label_id, (200, 200, 200))
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

            if writer is not None:
                output_frame = frame
                if cfg.output_scale != 1.0:
                    output_frame = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA)
                writer.write(output_frame)

            if cfg.live_preview and preview_emit_due:
                now = time.time()
                fps_ema = 0.92 * fps_ema + 0.08 * (1.0 / max(now - t_prev, 1e-6))
                t_prev = now
                self._store_latest_preview(
                    frame_to_qimage(frame, target_size=(cfg.preview_width, cfg.preview_height))
                )
            elif should_process_frame:
                now = time.time()
                fps_ema = 0.92 * fps_ema + 0.08 * (1.0 / max(now - t_prev, 1e-6))
                t_prev = now

            if frame_idx % max(1, effective_preview_stride) == 0:
                self.metrics_ready.emit(
                    {
                        "frame": frame_idx,
                        "fps_live_ema": fps_ema,
                        "visible_tracks": len(visible_ids) if should_process_frame else len(recognizer_active_ids),
                        "unique_track_ids": len(unique_stable_ids),
                        "falls": action_counts.get("Fall", 0),
                        "source_fps": fps_src,
                        "effective_process_stride": effective_process_stride,
                        "effective_preview_stride": effective_preview_stride,
                        "effective_action_update_stride": effective_action_update_stride,
                        "effective_action_pred_stride": effective_action_pred_stride or 0,
                        "effective_max_det": effective_max_det,
                        "cpu_auto_tuned": cpu_auto_tuned,
                        "processed_frames": detection_stats["frames_processed"],
                        "total_frames": total_frames or max_frames or 0,
                    }
                )

            if reader is None:
                frame_idx += 1

        if reader is not None:
            reader.stop()
            reader.join(timeout=2.0)
        else:
            cap.release()
        if writer is not None:
            writer.release()

        elapsed = max(time.time() - t_start, 1e-6)
        fps_avg = frame_idx / elapsed if frame_idx > 0 else 0.0
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
            "effective_preview_stride": effective_preview_stride,
            "effective_max_det": effective_max_det,
            "effective_action_pred_stride": effective_action_pred_stride,
            "effective_process_stride": effective_process_stride,
            "effective_action_update_stride": effective_action_update_stride,
            "effective_target_analysis_fps": effective_target_analysis_fps,
            "pose_backend": pose_runtime.get("backend"),
            "pose_device": pose_runtime.get("device"),
            "pose_weights": pose_runtime.get("weights_path"),
            "action_backend": action_backend,
            "action_fast_mode": action_fast_mode,
            "cpu_auto_tuned": cpu_auto_tuned,
            "output_path": str(output_path) if output_path else None,
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

        action_group = QGroupBox("Action Recognition")
        action_form = QFormLayout(action_group)
        action_form.addRow("Min Track Frames", self.min_track_frames_spin)
        action_form.addRow("Prediction Stride", self.pred_stride_spin)
        action_form.addRow("Action Confidence", self.action_conf_spin)
        action_form.addRow("Smoothing Window", self.smooth_window_spin)
        action_form.addRow("Fast Fall Sensitivity", self.fall_conf_boost_spin)
        action_form.addRow("Sitting Strictness", self.sitting_conf_penalty_spin)
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
            ("effective_process_stride", "Process Stride"),
            ("effective_preview_stride", "Preview Stride"),
            ("effective_max_det", "Effective MaxDet"),
            ("effective_action_pred_stride", "Action Pred Stride"),
            ("effective_action_update_stride", "Action Update Stride"),
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
            self.process_stride_spin.setValue(1)
            self.preview_stride_spin.setValue(4)
            self.output_scale_combo.setCurrentText("1.0")
            self.save_output_checkbox.setChecked(False)
            self.normalize_timing_checkbox.setChecked(True)
            self.target_analysis_fps_spin.setValue(12.0)
            self.pred_stride_spin.setValue(3)
            self.action_conf_spin.setValue(0.30)
            self.smooth_window_spin.setValue(3)
            self.fall_conf_boost_spin.setValue(0.10)
            self.sitting_conf_penalty_spin.setValue(0.20)
        elif profile_name == "fast":
            self.tracker_combo.setCurrentText("ByteTrack (custom)")
            self.det_conf_spin.setValue(0.28)
            self.det_iou_spin.setValue(0.45)
            self.imgsz_combo.setCurrentText("480")
            self.max_det_spin.setValue(20)
            self.live_preview_checkbox.setChecked(True)
            self.process_stride_spin.setValue(2)
            self.preview_stride_spin.setValue(5)
            self.output_scale_combo.setCurrentText("0.75")
            self.save_output_checkbox.setChecked(False)
            self.normalize_timing_checkbox.setChecked(True)
            self.target_analysis_fps_spin.setValue(10.0)
            self.pred_stride_spin.setValue(4)
            self.action_conf_spin.setValue(0.30)
            self.smooth_window_spin.setValue(3)
            self.fall_conf_boost_spin.setValue(0.10)
            self.sitting_conf_penalty_spin.setValue(0.20)
        else:
            self.tracker_combo.setCurrentText("BoT-SORT (custom)")
            self.det_conf_spin.setValue(0.30)
            self.det_iou_spin.setValue(0.50)
            self.imgsz_combo.setCurrentText("960")
            self.max_det_spin.setValue(100)
            self.live_preview_checkbox.setChecked(True)
            self.process_stride_spin.setValue(1)
            self.preview_stride_spin.setValue(2)
            self.output_scale_combo.setCurrentText("1.0")
            self.save_output_checkbox.setChecked(True)
            self.normalize_timing_checkbox.setChecked(False)
            self.target_analysis_fps_spin.setValue(12.0)
            self.pred_stride_spin.setValue(1)
            self.action_conf_spin.setValue(0.30)
            self.smooth_window_spin.setValue(3)
            self.fall_conf_boost_spin.setValue(0.10)
            self.sitting_conf_penalty_spin.setValue(0.20)
        self.auto_tune_cpu_checkbox.setChecked(True)
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
        self.last_output_path = None
        self._append_log("PyQt6 desktop run started.")

        self.worker = InferenceWorker(config)
        self.worker.frame_ready.connect(self._flush_preview_frame)
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

    def _update_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def _flush_preview_frame(self) -> None:
        worker = self.worker
        if worker is None:
            return
        image = worker.take_latest_preview()
        if image is None:
            return
        self._update_frame(image)

    def _update_metrics(self, metrics: dict) -> None:
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
            <b>Effective Process Stride:</b> {summary.get('effective_process_stride', 0)}<br>
            <b>Effective Preview Stride:</b> {summary.get('effective_preview_stride', 0)}<br>
            <b>Effective Action Pred Stride:</b> {summary.get('effective_action_pred_stride', 0) or 0}<br>
            <b>Effective Action Update Stride:</b> {summary.get('effective_action_update_stride', 0)}<br>
            <b>Effective MaxDet:</b> {summary.get('effective_max_det', 0)}<br>
            <b>CPU Auto-Tuned:</b> {summary.get('cpu_auto_tuned', False)}<br>
            <b>Pose Backend:</b> {summary.get('pose_backend') or 'unknown'}<br>
            <b>Pose Device:</b> {summary.get('pose_device') or 'unknown'}<br>
            <b>Pose Weights:</b> {summary.get('pose_weights') or 'unknown'}<br>
            <b>Action Backend:</b> {summary.get('action_backend') or 'disabled'}<br>
            <b>Action Fast Mode:</b> {summary.get('action_fast_mode', False)}<br>
            <b>Output:</b> {summary.get('output_path') or 'No file saved'}</p>
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
