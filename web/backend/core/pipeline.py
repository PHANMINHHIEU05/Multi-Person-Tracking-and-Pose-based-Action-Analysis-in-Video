"""Pipeline wrapper – runs module_c in a background thread and pushes
annotated JPEG frames + metrics into the session's asyncio.Queue.

Queue items are tuples:
  (jpeg_bytes: bytes, meta: dict)   for video frames
  (None, meta: dict)                for status messages

The WebSocket endpoint sends jpeg_bytes as a binary WS message followed
by the meta dict as a text (JSON) message.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import queue as stdlib_queue

import cv2
import numpy as np

# Only send every Nth frame over WebSocket to reduce JPEG-encode + network overhead.
# All frames are still processed for tracking accuracy and saved to output video.
STREAM_EVERY = 3  # 1=every frame, 3=every 3rd frame sent to browser


class _ThreadedWriter:
    """Writes annotated frames to disk in a daemon thread so H.264 software
    encoding never blocks the inference loop."""

    def __init__(self, path: str, fourcc: int, fps: float, size: tuple):
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        self._q: stdlib_queue.Queue = stdlib_queue.Queue(maxsize=60)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def write(self, frame: np.ndarray) -> None:
        """Non-blocking write — drops frame if the encoder can't keep up."""
        try:
            self._q.put_nowait(frame)  # use a copy so the inference loop can mutate freely
        except stdlib_queue.Full:
            pass

    def _loop(self) -> None:
        while True:
            frame = self._q.get()
            if frame is None:  # sentinel → exit
                break
            self._writer.write(frame)

    def release(self) -> None:
        self._q.put(None)          # send sentinel
        self._thread.join(timeout=30)
        self._writer.release()

# Add project root to path so module_c and train_professional_v3 are importable
ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.module_c_action import (
    ActionRecognizer,
    extract_kpts_for_track,
    draw_skeleton,
    draw_action_label,
    LABEL_MAP,
    LABEL_COLORS,
    SEQ_LEN,
)
from web.backend.core.session import RunSession


def _encode_jpeg_bytes(frame: np.ndarray, quality: int = 50, max_width: int = 720) -> bytes:
    """BGR numpy frame → raw JPEG bytes.

    Resizes to *max_width* pixels wide before encoding.  Returns raw bytes
    (not base64) so WebSocket can send them as a binary frame — 33 % smaller
    than base64 and no encode/decode overhead.
    """
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)),
                           interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return b""
    return buf.tobytes()


def _push_to_queue(session: "RunSession", item: tuple,
                   loop: asyncio.AbstractEventLoop) -> None:
    """Non-blocking put. Drops when queue is full.
    item is (jpeg_bytes | None, meta_dict).
    """
    def _put():
        try:
            session.queue.put_nowait(item)
        except asyncio.QueueFull:
            pass  # drop frame — client is too slow

    loop.call_soon_threadsafe(_put)


def run_pipeline(session: RunSession, model_path: str, conf: float,
                 imgsz: int, loop: asyncio.AbstractEventLoop):
    """Entry point for the background thread."""
    from ultralytics import YOLO

    try:
        session.status = "running"
        pose_model = YOLO(str(ROOT / "yolov8n-pose.pt"))
        recognizer = ActionRecognizer(
            model_path=model_path,
            device="auto",
            num_classes=5,
            hidden_dim=128,
            num_layers=3,
            num_heads=8,
        )

        cap = cv2.VideoCapture(session.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {session.video_path}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output video writer runs in its own daemon thread so H.264 encoding
        # never blocks the inference loop.
        out_path = Path(session.output_dir) / "video_action.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = _ThreadedWriter(str(out_path), fourcc, fps_src, (W, H))

        import csv
        csv_path = Path(session.output_dir) / "actions.csv"
        csv_f = open(csv_path, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["frame", "track_id", "action", "confidence", "x1", "y1", "x2", "y2"])

        action_counts: dict[str, int] = defaultdict(int)
        frame_idx = 0
        t_prev = time.time()
        fps_display = fps_src

        tracker_cfg = str(ROOT / "config" / "botsort.yaml")

        while True:
            if session.stop_event.is_set():
                session.status = "stopped"
                break

            ret, frame = cap.read()
            if not ret:
                session.status = "done"
                break

            results = pose_model.track(
                frame,
                persist=True,
                conf=conf,
                iou=0.45,
                imgsz=imgsz,
                max_det=50,
                classes=[0],
                tracker=tracker_cfg,
                verbose=False,
                half=True,   # FP16 inference on CUDA — ~2× faster, ignored on CPU
            )
            result = results[0]
            active_ids = set()

            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                bboxes = result.boxes.xyxy.cpu().numpy()
                tracks_payload = []

                for i, tid in enumerate(track_ids):
                    active_ids.add(tid)
                    bbox = bboxes[i]
                    kpts = extract_kpts_for_track(result, tid, W, H)
                    recognizer.update_track(tid, kpts)
                    label_id, conf_val, label_name = recognizer.predict(tid)
                    color = LABEL_COLORS.get(label_id, (200, 200, 200))

                    if kpts is not None:
                        draw_skeleton(frame, kpts, color, W, H)
                    draw_action_label(frame, bbox, tid, label_name, conf_val, color)

                    x1, y1, x2, y2 = bbox.astype(int)
                    csv_w.writerow([frame_idx, tid, label_name, f"{conf_val:.4f}",
                                    x1, y1, x2, y2])

                    if label_name != "?":
                        action_counts[label_name] += 1
                    if label_name == "Fall":
                        session.fall_count += 1

                    tracks_payload.append({
                        "id": int(tid),
                        "action": label_name,
                        "conf": round(float(conf_val), 3),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    })
            else:
                tracks_payload = []

            recognizer.remove_stale_tracks(active_ids)
            writer.write(frame)

            # FPS計算
            now = time.time()
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            frame_idx += 1
            session.total_frames = frame_idx

            # Only encode + stream every STREAM_EVERY frames to cut JPEG overhead.
            # Tracking and CSV writing still happen on every frame.
            if frame_idx % STREAM_EVERY == 0:
                fall_alert = bool(
                    tracks_payload and any(t["action"] == "Fall" for t in tracks_payload)
                )
                jpeg_bytes = _encode_jpeg_bytes(frame)
                meta = {
                    "type": "frame",
                    "frame_idx": frame_idx,
                    "fps": round(fps_display, 1),
                    "tracks": tracks_payload,
                    "action_counts": dict(action_counts),
                    "fall_alert": fall_alert,
                }
                _push_to_queue(session, (jpeg_bytes, meta), loop)

        cap.release()
        writer.release()
        csv_f.close()

    except Exception as exc:
        session.status = "error"
        session.error_msg = str(exc)
        _push_to_queue(session, (None, {"type": "status", "status": "error", "message": str(exc)}), loop)
        return

    # Send final status (always deliver even if queue was previously full)
    final_status = session.status  # "done" or "stopped"
    _push_to_queue(session, (None, {"type": "status", "status": final_status, "message": ""}), loop)


def start_pipeline_thread(session: RunSession, model_path: str,
                           conf: float, imgsz: int,
                           loop: asyncio.AbstractEventLoop) -> threading.Thread:
    t = threading.Thread(
        target=run_pipeline,
        args=(session, model_path, conf, imgsz, loop),
        daemon=True,
    )
    session.thread = t
    t.start()
    return t
