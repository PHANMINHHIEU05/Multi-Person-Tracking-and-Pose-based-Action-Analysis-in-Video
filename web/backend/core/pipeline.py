"""Pipeline wrapper – runs module_c in a background thread."""
from __future__ import annotations

import asyncio
import csv
import queue as stdlib_queue
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

# PERF: Stream preview every N frames (model still runs every frame).
STREAM_EVERY = 2
# PERF: Action recognizer runs every N frames and reuses cached actions between runs.
RECOGNIZE_EVERY_N = 3
# PERF: Lightweight drawing on preview/output frame.
DRAW_SKELETON = False
DRAW_LABEL = True
# PERF: Module-level cache required by spec (track_id -> (label_id, conf, label_name)).
_last_actions: dict[int, tuple[int, float, str]] = {}
# PERF: Module-level executor for non-blocking disk I/O.
io_executor = ThreadPoolExecutor(max_workers=2)
# PERF: Enable per-frame timing logs for bottleneck diagnosis.
PERF_LOG = True


class _ThreadedWriter:
    """PERF: Sequential video writer to preserve frame order and valid PTS."""

    def __init__(self, path: str, fourcc: int, fps: float, size: tuple[int, int]):
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        self._q: stdlib_queue.Queue = stdlib_queue.Queue(maxsize=64)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def write(self, frame: np.ndarray) -> None:
        """PERF: Non-blocking enqueue; drop oldest if writer lags."""
        try:
            self._q.put_nowait(frame)
        except stdlib_queue.Full:
            try:
                self._q.get_nowait()
            except stdlib_queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except stdlib_queue.Full:
                pass

    def _loop(self) -> None:
        while True:
            frame = self._q.get()
            if frame is None:
                break
            self._writer.write(frame)

    def release(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=10)
        self._writer.release()

# Add project root to path so module_c is importable.
ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.module_c_action import (  # noqa: E402
    ActionRecognizer,
    LABEL_COLORS,
    draw_action_label,
    draw_skeleton,
    extract_kpts_for_track,
)
from web.backend.core.session import RunSession  # noqa: E402
from web.backend.core.stream import push_preview_frame  # noqa: E402


def _decode_worker(cap: cv2.VideoCapture,
                   frame_queue: stdlib_queue.Queue,
                   stop_evt: threading.Event) -> None:
    """PERF: Dedicated decode thread. Keeps latest decoded frames in a small queue."""
    while not stop_evt.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame_queue.put_nowait(frame)
        except stdlib_queue.Full:
            try:
                frame_queue.get_nowait()  # PERF: Drop oldest decoded frame.
            except stdlib_queue.Empty:
                pass
            try:
                frame_queue.put_nowait(frame)
            except stdlib_queue.Full:
                pass
    try:
        frame_queue.put_nowait(None)  # PERF: Sentinel to stop main loop.
    except stdlib_queue.Full:
        pass


def encode_preview(frame: np.ndarray) -> bytes:
    """PERF: Resize to 854x480 and JPEG-encode with quality 72 for web preview."""
    resized = cv2.resize(frame, (854, 480), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if not ok:
        return b""
    return buf.tobytes()


def _push_meta(session: RunSession,
               payload: dict,
               loop: asyncio.AbstractEventLoop) -> None:
    """PERF: Non-blocking metadata push; drop stale metadata when queue is full."""

    def _put() -> None:
        try:
            session.queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    loop.call_soon_threadsafe(_put)


def run_pipeline(session: RunSession,
                 model_path: str,
                 conf: float,
                 imgsz: int,
                 loop: asyncio.AbstractEventLoop):
    """Entry point for background processing thread."""
    from ultralytics import YOLO

    cap: cv2.VideoCapture | None = None
    writer: _ThreadedWriter | None = None
    csv_f = None
    decode_stop = threading.Event()
    decode_thread = None
    pending_io = []

    try:
        session.status = "running"
        _last_actions.clear()  # PERF: Reset cache per run.

        pose_model = YOLO(str(ROOT / "yolov8n-pose.pt"))
        use_half = bool(torch.cuda.is_available())  # PERF: FP16 only when CUDA is available.

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

        out_path = Path(session.output_dir) / "video_action.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = _ThreadedWriter(str(out_path), fourcc, fps_src, (W, H))

        csv_path = Path(session.output_dir) / "actions.csv"
        csv_f = open(csv_path, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["frame", "track_id", "action", "confidence", "x1", "y1", "x2", "y2"])
        csv_lock = threading.Lock()

        def _csv_write_row(row: list) -> None:
            # PERF: Serialize csv writes to avoid race/corruption under thread pool.
            with csv_lock:
                csv_w.writerow(row)

        # PERF: Decode worker queue (maxsize=8) to decouple cap.read from inference loop.
        frame_queue: stdlib_queue.Queue = stdlib_queue.Queue(maxsize=8)
        decode_thread = threading.Thread(
            target=_decode_worker,
            args=(cap, frame_queue, decode_stop),
            daemon=True,
        )
        decode_thread.start()

        action_counts: dict[str, int] = defaultdict(int)
        frame_idx = 0
        t_prev = time.time()
        fps_display = fps_src

        while True:
            t0 = time.perf_counter()
            if session.stop_event.is_set():
                session.status = "stopped"
                break

            try:
                frame = frame_queue.get(timeout=1.0)
            except stdlib_queue.Empty:
                continue
            t1 = time.perf_counter()

            if frame is None:
                session.status = "done"
                break

            # PERF: Required YOLO args for faster GPU inference and lighter tracking.
            results_iter = pose_model.track(
                frame,
                persist=True,
                conf=conf,
                iou=0.45,
                imgsz=imgsz,
                max_det=50,
                classes=[0],
                tracker="bytetrack.yaml",
                verbose=False,
                half=use_half,
                stream=True,
            )
            result = next(results_iter, None)
            t2 = time.perf_counter()
            if result is None:
                continue

            active_ids = set()
            tracks_payload = []

            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                bboxes = result.boxes.xyxy.cpu().numpy()

                # PERF: Update per-track keypoint buffer every frame.
                for i, tid in enumerate(track_ids):
                    active_ids.add(int(tid))
                    kpts = extract_kpts_for_track(result, int(tid), W, H)
                    recognizer.update_track(int(tid), kpts)

                should_recognize = (frame_idx % RECOGNIZE_EVERY_N == 0)

                if should_recognize:
                    # PERF: Use batched recognizer path when available; fallback to per-track predict.
                    if hasattr(recognizer, "predict_batch"):
                        try:
                            batch_preds = recognizer.predict_batch(track_ids.tolist())  # type: ignore[attr-defined]
                            for tid, pred in batch_preds.items():
                                _last_actions[int(tid)] = pred
                        except Exception:
                            for tid in track_ids:
                                _last_actions[int(tid)] = recognizer.predict(int(tid))
                    else:
                        for tid in track_ids:
                            _last_actions[int(tid)] = recognizer.predict(int(tid))

                for i, tid in enumerate(track_ids):
                    tid = int(tid)
                    bbox = bboxes[i]
                    label_id, conf_val, label_name = _last_actions.get(tid, (-1, 0.0, "?"))
                    color = LABEL_COLORS.get(label_id, (200, 200, 200))

                    if DRAW_SKELETON:
                        kpts_draw = extract_kpts_for_track(result, tid, W, H)
                        if kpts_draw is not None:
                            draw_skeleton(frame, kpts_draw, color, W, H)
                    if DRAW_LABEL:
                        draw_action_label(frame, bbox, tid, label_name, conf_val, color)

                    x1, y1, x2, y2 = bbox.astype(int)
                    # PERF: CSV write submitted to I/O executor to avoid blocking main loop.
                    pending_io.append(
                        io_executor.submit(
                            _csv_write_row,
                            [frame_idx, tid, label_name, f"{conf_val:.4f}", x1, y1, x2, y2],
                        )
                    )

                    if label_name != "?":
                        action_counts[label_name] += 1
                    if label_name == "Fall":
                        session.fall_count += 1

                    tracks_payload.append({
                        "id": tid,
                        "action": label_name,
                        "conf": round(float(conf_val), 3),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    })

            t3 = time.perf_counter()

            recognizer.remove_stale_tracks(active_ids)
            for stale_tid in list(_last_actions.keys()):
                if stale_tid not in active_ids:
                    _last_actions.pop(stale_tid, None)

            # PERF: Video write is enqueued to single writer thread (ordered PTS).
            if writer is not None:
                writer.write(frame.copy())

            # PERF: Keep pending futures bounded.
            if len(pending_io) > 256:
                pending_io = [f for f in pending_io if not f.done()]

            now = time.time()
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            frame_idx += 1
            session.total_frames = frame_idx

            if frame_idx % STREAM_EVERY == 0:
                fall_alert = bool(tracks_payload and any(t["action"] == "Fall" for t in tracks_payload))
                enc_t_start = time.perf_counter()
                preview_bytes = encode_preview(frame)
                t4 = time.perf_counter()
                # PERF: Push latest preview frame through global drop queue.
                push_preview_frame(preview_bytes)
                meta = {
                    "type": "frame",
                    "frame_idx": frame_idx,
                    "fps": round(fps_display, 1),
                    "tracks": tracks_payload,
                    "action_counts": dict(action_counts),
                    "fall_alert": fall_alert,
                }
                _push_meta(session, meta, loop)
            else:
                t4 = time.perf_counter()

            if PERF_LOG:
                print(
                    f"[PERF] decode={1000*(t1-t0):.1f}ms | yolo={1000*(t2-t1):.1f}ms | "
                    f"action={1000*(t3-t2):.1f}ms | encode={1000*(t4-t3):.1f}ms | "
                    f"total={1000*(t4-t0):.1f}ms"
                )

        # PERF: Flush non-blocking I/O tasks before closing files.
        for fut in pending_io:
            try:
                fut.result(timeout=5)
            except Exception:
                pass

    except Exception as exc:
        session.status = "error"
        session.error_msg = str(exc)
        _push_meta(session, {"type": "status", "status": "error", "message": str(exc)}, loop)
        return

    finally:
        decode_stop.set()
        if decode_thread is not None:
            decode_thread.join(timeout=5)
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if csv_f is not None:
            csv_f.close()

    final_status = session.status
    _push_meta(session, {"type": "status", "status": final_status, "message": ""}, loop)


def start_pipeline_thread(session: RunSession,
                          model_path: str,
                          conf: float,
                          imgsz: int,
                          loop: asyncio.AbstractEventLoop) -> threading.Thread:
    t = threading.Thread(
        target=run_pipeline,
        args=(session, model_path, conf, imgsz, loop),
        daemon=True,
    )
    session.thread = t
    t.start()
    return t
