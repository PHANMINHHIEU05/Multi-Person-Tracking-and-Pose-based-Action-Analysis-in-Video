from __future__ import annotations

import tempfile
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

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


class ActionRecognizerLite:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float,
        pred_stride: int,
        min_track_frames: int,
        smooth_window: int,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self.feat_mean = None
        self.feat_std = None
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

        self._buffers: Dict[int, deque] = {}
        self._frame_count: Dict[int, int] = {}
        self._last_pred: Dict[int, Tuple[int, float]] = {}
        self._frames_since_pred: Dict[int, int] = {}
        self._pred_history: Dict[int, list] = {}

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
            return lid, conf, LABEL_MAP.get(lid, "?")

        self._frames_since_pred[track_id] = 0

        x = prepare_sequence(self._buffers[track_id])
        if self.feat_mean is not None:
            x = (x - self.feat_mean) / self.feat_std

        xt = torch.FloatTensor(x).to(self.device)
        logits, _ = self.model(xt)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        label_id = int(np.argmax(probs))
        confidence = float(probs[label_id])

        if confidence >= self.conf_threshold:
            hist = self._pred_history.setdefault(track_id, [])
            hist.append(label_id)
            if len(hist) > self.smooth_window:
                hist.pop(0)
            smoothed_id = Counter(hist).most_common(1)[0][0]
            self._last_pred[track_id] = (smoothed_id, confidence)

        lid, conf = self._last_pred.get(track_id, (-1, 0.0))
        return lid, conf, LABEL_MAP.get(lid, "?")

    def remove_stale_tracks(self, active_ids: set[int]):
        dead = [tid for tid in self._buffers if tid not in active_ids]
        for tid in dead:
            del self._buffers[tid]
            del self._frame_count[tid]
            del self._last_pred[tid]
            del self._frames_since_pred[tid]
            self._pred_history.pop(tid, None)


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
):
    try:
        if not Path(model_path).exists():
            return None, f"Action checkpoint not found: {model_path}"
        rec = ActionRecognizerLite(
            model_path=model_path,
            conf_threshold=conf_threshold,
            pred_stride=pred_stride,
            min_track_frames=min_track_frames,
            smooth_window=smooth_window,
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
):
    frame_slot = st.empty()
    metrics_slot = st.empty()
    progress_slot = st.empty()
    progress = progress_slot.progress(0) if total_frames and total_frames > 0 else None

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = None
    if output_path is not None and w > 0 and h > 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_src, (w, h))

    action_counts = defaultdict(int)
    t_prev = time.time()
    fps_ema = fps_src if fps_src > 0 else 30.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        results_iter = pose_model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            conf=det_conf,
            iou=det_iou,
            imgsz=imgsz,
            max_det=max_det,
            classes=[0],
            half=torch.cuda.is_available(),
            verbose=False,
            stream=True,
        )
        result = next(results_iter, None)
        if result is None:
            frame_idx += 1
            continue

        active_ids = set()

        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            bboxes = result.boxes.xyxy.cpu().numpy()

            for i, tid in enumerate(track_ids):
                active_ids.add(int(tid))
                bbox = bboxes[i]

                kpts = extract_kpts_for_track(result, int(tid), frame.shape[1], frame.shape[0])
                if recognizer is not None:
                    recognizer.update_track(int(tid), kpts)
                    label_id, conf_val, label_name = recognizer.predict(int(tid))
                else:
                    # Fallback heuristic only when action model unavailable.
                    aspect = (bbox[2] - bbox[0]) / max((bbox[3] - bbox[1]), 1e-6)
                    label_id = 0 if aspect > 1.2 else 1
                    conf_val = 0.50
                    label_name = LABEL_MAP.get(label_id, "Walking")

                if label_name not in ("?", "unknown"):
                    action_counts[label_name] += 1

                color = LABEL_COLORS.get(label_id, (200, 200, 200))
                if draw_skeleton_flag and kpts is not None:
                    draw_skeleton(frame, kpts, color, frame.shape[1], frame.shape[0])
                draw_action_label(frame, bbox, int(tid), label_name, conf_val, color)

        if recognizer is not None:
            recognizer.remove_stale_tracks(active_ids)

        if writer is not None:
            writer.write(frame)

        now = time.time()
        fps_ema = 0.92 * fps_ema + 0.08 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        if frame_idx % max(1, preview_stride) == 0:
            metrics_slot.info(
                f"Frame: {frame_idx} | FPS: {fps_ema:.1f} | Tracks: {len(active_ids)} | "
                f"Falls: {action_counts.get('Fall', 0)}"
            )

            if progress is not None:
                pct = min(100, int((frame_idx + 1) * 100 / max(total_frames, 1)))
                progress.progress(pct)

            if show_preview:
                preview_frame = frame
                if display_max_width > 0 and frame.shape[1] > display_max_width:
                    scale = display_max_width / float(frame.shape[1])
                    preview_h = max(1, int(frame.shape[0] * scale))
                    preview_frame = cv2.resize(frame, (display_max_width, preview_h), interpolation=cv2.INTER_AREA)
                frame_slot.image(preview_frame, channels="BGR", width="stretch")

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if progress is not None:
        progress.progress(100)

    return {
        "frames": frame_idx,
        "fps": fps_ema,
        "counts": dict(action_counts),
        "output_path": str(output_path) if output_path else None,
    }


def main():
    st.set_page_config(page_title="Fall Detection - Streamlit", layout="wide")
    st.title("Fall Detection & Action Recognition (Single-file Streamlit)")
    st.caption("No React/FastAPI/Spring data path. Frames are processed directly in Python.")

    with st.sidebar:
        st.header("Settings")
        source = st.radio("Input Source", ["Upload Video", "Webcam"])

        pose_weights = st.text_input("YOLO Pose Weights", value=str(ROOT / "yolov8n-pose.pt"))
        tracker_name = st.selectbox(
            "Tracker",
            ["ByteTrack (custom)", "ByteTrack (default)", "BoT-SORT (custom)", "BoT-SORT (default)"],
            index=0,
        )
        det_conf = st.slider("Detection Confidence", 0.05, 0.95, 0.25, 0.01)
        det_iou = st.slider("NMS IoU", 0.10, 0.95, 0.45, 0.01)
        imgsz = st.select_slider("Image Size", [320, 480, 640, 960, 1280], value=640)
        max_det = st.slider("Max Persons per Frame", 1, 200, 50, 1)
        draw_skeleton_flag = st.checkbox("Draw Skeleton", value=False)

        st.markdown("---")
        st.subheader("Performance")
        live_preview_upload = st.checkbox("Live preview while processing upload", value=False)
        preview_stride = st.slider("Preview update every N frames", 1, 10, 3, 1)
        display_max_width = st.select_slider("Preview max width", [640, 854, 960, 1280, 1920], value=960)

        st.markdown("---")
        st.subheader("Action Model")
        action_ckpt = st.text_input("Action Checkpoint", value=str(ROOT / "runs" / "train_horizontal" / "final_safe_system.pth"))
        min_track_frames = st.slider("Min Track Frames", 1, 128, 24, 1)
        pred_stride = st.slider("Prediction Stride", 1, 16, 2, 1)
        action_conf = st.slider("Action Confidence Threshold", 0.01, 0.95, 0.25, 0.01)
        smooth_window = st.slider("Smoothing Window", 1, 7, 3, 1)

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

            output_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name)
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
            )

            st.success("Done")
            st.json(summary)
            if output_path.exists():
                st.video(str(output_path))
                with output_path.open("rb") as f:
                    st.download_button(
                        "Download annotated video",
                        data=f.read(),
                        file_name="video_action_streamlit.mp4",
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
                show_preview=True,
                preview_stride=preview_stride,
                display_max_width=display_max_width,
            )
            st.success("Webcam run finished")
            st.json(summary)


if __name__ == "__main__":
    main()
