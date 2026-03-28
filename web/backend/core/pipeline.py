"""Pipeline wrapper - runs module_c in a background thread."""  # REFACTOR:
from __future__ import annotations  # REFACTOR:

import asyncio  # REFACTOR:
import csv  # REFACTOR:
import queue as stdlib_queue  # REFACTOR:
import sys  # REFACTOR:
import threading  # REFACTOR:
import time  # REFACTOR:
import traceback
from collections import defaultdict, deque  # REFACTOR: # FIX: use deque for explicit per-track temporal buffering
from concurrent.futures import ThreadPoolExecutor  # REFACTOR:
from pathlib import Path  # REFACTOR:

import cv2  # REFACTOR:
import numpy as np  # REFACTOR:
import torch  # REFACTOR:

STREAM_EVERY = 1  # REFACTOR: send metadata every frame for tighter overlay sync
INFER_EVERY = 3  # REFACTOR: # FIX: run recognizer every N frames to stabilize predictions and reduce compute
DRAW_SKELETON = False  # REFACTOR: optional drawing only for output video
DRAW_LABEL = True  # REFACTOR: keep action labels for output artifact
io_executor = ThreadPoolExecutor(max_workers=2)  # REFACTOR: non-blocking CSV/file I/O tasks
PERF_LOG = True  # REFACTOR: keep timing logs for diagnostics

ROOT = Path(__file__).parent.parent.parent.parent  # REFACTOR:
sys.path.insert(0, str(ROOT))  # REFACTOR:

from src.module_c_action import (  # noqa: E402  # REFACTOR:
    ActionRecognizer,
    LABEL_COLORS,
    MIN_TRACK_FRAMES,  # FIX: reuse module C minimum track context requirement
    SEQ_LEN,  # FIX: reuse module C sequence length trained for the model
    draw_action_label,
    draw_skeleton,
    extract_kpts_for_track,
)

WINDOW_SIZE = SEQ_LEN  # FIX: align temporal window with model training sequence length
MIN_FRAMES = MIN_TRACK_FRAMES  # FIX: require enough temporal context before emitting predictions
from web.backend.core.session import RunSession  # noqa: E402  # REFACTOR:


class _ThreadedWriter:  # REFACTOR: ordered single-thread writer avoids ffmpeg PTS errors
    """Sequential video writer preserving frame order."""  # REFACTOR:

    def __init__(self, path: str, fourcc: int, fps: float, size: tuple[int, int]):  # REFACTOR:
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)  # REFACTOR:
        self._q: stdlib_queue.Queue = stdlib_queue.Queue(maxsize=64)  # REFACTOR:
        self._thread = threading.Thread(target=self._loop, daemon=True)  # REFACTOR:
        self._thread.start()  # REFACTOR:

    def write(self, frame: np.ndarray) -> None:  # REFACTOR:
        try:
            self._q.put_nowait(frame)  # REFACTOR: enqueue frame for ordered write
        except stdlib_queue.Full:
            try:
                self._q.get_nowait()  # REFACTOR: drop oldest to keep pipeline realtime
            except stdlib_queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)  # REFACTOR:
            except stdlib_queue.Full:
                pass

    def _loop(self) -> None:  # REFACTOR:
        while True:
            frame = self._q.get()  # REFACTOR:
            if frame is None:  # REFACTOR:
                break
            self._writer.write(frame)  # REFACTOR:

    def release(self) -> None:  # REFACTOR:
        self._q.put(None)  # REFACTOR:
        self._thread.join(timeout=10)  # REFACTOR:
        self._writer.release()  # REFACTOR:


def _decode_worker(cap: cv2.VideoCapture, frame_queue: stdlib_queue.Queue,
                   stop_evt: threading.Event,
                   fps_src: float) -> None:  # REFACTOR:
    """Decode frames in a dedicated thread and keep only recent frames."""  # REFACTOR:
    src_frame_idx = -1  # REFACTOR:
    while not stop_evt.is_set():
        ret, frame = cap.read()  # REFACTOR:
        if not ret:
            if src_frame_idx < 0:
                print("[PIPELINE] decode ended immediately: cv2 could not read first frame")
            break
        src_frame_idx += 1  # REFACTOR:
        src_time_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)  # REFACTOR:
        if src_time_ms <= 0.0 and fps_src > 0:
            src_time_ms = (src_frame_idx / fps_src) * 1000.0  # REFACTOR: fallback when container has poor timestamps
        item = (src_frame_idx, src_time_ms, frame)  # REFACTOR:
        try:
            frame_queue.put_nowait(item)  # REFACTOR:
        except stdlib_queue.Full:
            try:
                frame_queue.get_nowait()  # REFACTOR: drop stale decoded frame
            except stdlib_queue.Empty:
                pass
            try:
                frame_queue.put_nowait(item)  # REFACTOR:
            except stdlib_queue.Full:
                pass
    try:
        frame_queue.put_nowait(None)  # REFACTOR: sentinel
    except stdlib_queue.Full:
        pass


def _push_meta(session: RunSession, payload: dict,
               loop: asyncio.AbstractEventLoop) -> None:  # REFACTOR:
    """Push JSON metadata into session queue from worker thread."""  # REFACTOR:

    def _put() -> None:  # REFACTOR:
        try:
            session.queue.put_nowait(payload)  # REFACTOR:
        except asyncio.QueueFull:
            pass  # REFACTOR:

    loop.call_soon_threadsafe(_put)  # REFACTOR:


def build_payload(frame_idx, fps, track_ids, boxes, actions,
                  fall_alert, action_counts,
                  inference_width=None, inference_height=None,
                  source_fps=None,
                  source_time_sec=None,
                  timestamp=None):  # REFACTOR: # FIX: explicit playback timestamp for frontend sync
    tracks = []  # REFACTOR:
    for tid, box, action in zip(track_ids, boxes, actions):  # REFACTOR:
        if hasattr(box, "xyxy"):
            bbox_vals = [round(v, 1) for v in box.xyxy[0].tolist()]  # REFACTOR:
            conf_val = round(float(box.conf[0]), 3)  # REFACTOR:
        else:
            bbox_arr, conf_arr = box  # REFACTOR:
            bbox_vals = [round(float(v), 1) for v in bbox_arr.tolist()]  # REFACTOR:
            conf_val = round(float(conf_arr), 3)  # REFACTOR:
        tracks.append({  # REFACTOR:
            "id": int(tid),
            "bbox": bbox_vals,
            "action": action,
            "conf": conf_val,
        })
    return {  # REFACTOR:
        "type": "frame",  # REFACTOR:
        "frame_idx": int(frame_idx),  # REFACTOR:
        "fps": round(float(fps), 1),  # REFACTOR:
        "tracks": tracks,  # REFACTOR:
        "fall_alert": bool(fall_alert),  # REFACTOR:
        "action_counts": dict(action_counts),  # REFACTOR:
        "inferenceWidth": int(inference_width) if inference_width else None,  # REFACTOR:
        "inferenceHeight": int(inference_height) if inference_height else None,  # REFACTOR:
        "sourceFps": round(float(source_fps), 3) if source_fps else None,  # REFACTOR:
        "sourceTimeSec": round(float(source_time_sec), 4) if source_time_sec is not None else None,  # REFACTOR:
        "timestamp": round(float(timestamp), 4) if timestamp is not None else None,  # FIX: sync key for canvas matching
    }


def run_pipeline(session: RunSession, model_path: str,
                 conf: float, imgsz: int,
                 loop: asyncio.AbstractEventLoop):  # REFACTOR:
    """Entry point for background processing thread."""  # REFACTOR:
    from ultralytics import YOLO

    cap: cv2.VideoCapture | None = None  # REFACTOR:
    writer: _ThreadedWriter | None = None  # REFACTOR:
    csv_f = None  # REFACTOR:
    decode_stop = threading.Event()  # REFACTOR:
    decode_thread = None  # REFACTOR:
    pending_io = []  # REFACTOR:

    try:
        session.status = "running"  # REFACTOR:

        pose_model = YOLO(str(ROOT / "yolov8n-pose.pt"))  # REFACTOR:
        use_half = bool(torch.cuda.is_available())  # REFACTOR:
        _dummy = np.zeros((640, 640, 3), dtype=np.uint8)  # FIX: CUDA/model warm-up to reduce first-frame latency spike
        pose_model.predict(_dummy, verbose=False)  # FIX: trigger kernels and graph setup before main loop
        del _dummy  # FIX: release warm-up tensor

        keypoint_buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))  # FIX: maintain explicit temporal context per track
        last_actions: dict[int, tuple[int, float, str]] = {}  # FIX: cache latest prediction per track_id
        _infer_counter = 0  # FIX: cadence counter for batched/cadenced inference

        recognizer = ActionRecognizer(  # REFACTOR:
            model_path=model_path,
            device="auto",
            num_classes=5,
            hidden_dim=128,
            num_layers=3,
            num_heads=8,
        )

        cap = cv2.VideoCapture(session.video_path)  # REFACTOR:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {session.video_path}")
        print(f"[PIPELINE] start run_id video={session.video_path}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0  # REFACTOR:
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # REFACTOR:
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # REFACTOR:
        print(f"[PIPELINE] source video info: {W}x{H} @ {fps_src:.3f}fps")

        out_path = Path(session.output_dir) / "video_action.mp4"  # REFACTOR:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # REFACTOR:
        writer = _ThreadedWriter(str(out_path), fourcc, fps_src, (W, H))  # REFACTOR:

        csv_path = Path(session.output_dir) / "actions.csv"  # REFACTOR:
        csv_f = open(csv_path, "w", newline="")  # REFACTOR:
        csv_w = csv.writer(csv_f)  # REFACTOR:
        csv_w.writerow(["frame", "track_id", "action", "confidence", "x1", "y1", "x2", "y2"])  # REFACTOR:
        csv_lock = threading.Lock()  # REFACTOR:

        def _csv_write_row(row: list) -> None:  # REFACTOR:
            with csv_lock:
                csv_w.writerow(row)  # REFACTOR:

        frame_queue: stdlib_queue.Queue = stdlib_queue.Queue(maxsize=8)  # REFACTOR:
        decode_thread = threading.Thread(target=_decode_worker,
                                         args=(cap, frame_queue, decode_stop, fps_src),
                                         daemon=True)  # REFACTOR:
        decode_thread.start()  # REFACTOR:

        action_counts: dict[str, int] = defaultdict(int)  # REFACTOR:
        processed_idx = 0  # REFACTOR:
        t_prev = time.time()  # REFACTOR:
        fps_display = fps_src  # REFACTOR:

        while True:
            t0 = time.perf_counter()  # REFACTOR:
            if session.stop_event.is_set():
                session.status = "stopped"  # REFACTOR:
                break

            try:
                item = frame_queue.get(timeout=1.0)  # REFACTOR:
            except stdlib_queue.Empty:
                continue
            t1 = time.perf_counter()  # REFACTOR:

            if item is None:
                session.status = "done"  # REFACTOR:
                break
            src_frame_idx, src_time_ms, frame = item  # REFACTOR:

            results_iter = pose_model.track(  # REFACTOR:
                frame,
                persist=True,
                conf=0.35,  # FIX: stabilize track continuity with lower detection gate
                iou=0.45,
                imgsz=imgsz,
                max_det=50,
                classes=[0],
                tracker=str(ROOT / "bytetrack_custom.yaml"),  # FIX: absolute path avoids tracker file resolution issues
                verbose=False,
                half=True,  # FIX: keep fp16 path enabled for CUDA acceleration
                stream=True,
            )
            result = next(results_iter, None)  # REFACTOR:
            t2 = time.perf_counter()  # REFACTOR:
            if result is None:
                continue

            active_ids = set()  # REFACTOR:
            track_ids_list: list[int] = []  # REFACTOR:
            boxes_payload: list[tuple[np.ndarray, float]] = []  # REFACTOR:
            actions_payload: list[str] = []  # REFACTOR:

            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)  # REFACTOR:
                bboxes = result.boxes.xyxy.cpu().numpy()  # REFACTOR:
                confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(track_ids))  # REFACTOR:

                _infer_counter += 1  # FIX: global cadence for expensive recognizer calls
                all_keypoints: list[np.ndarray | None] = []  # FIX: align keypoints with current track list
                for tid in track_ids:  # FIX: collect keypoints first for stable zipped processing
                    tid_i = int(tid)  # FIX:
                    kpts = extract_kpts_for_track(result, tid_i, W, H)  # FIX:
                    all_keypoints.append(kpts)  # FIX:

                for tid, keypoints in zip(track_ids, all_keypoints):  # FIX: buffered per-track update and infer gate
                    tid_i = int(tid)  # FIX:
                    active_ids.add(tid_i)  # FIX:
                    if keypoints is not None:  # FIX:
                        keypoint_buffers[tid_i].append(keypoints)  # FIX:
                    recognizer.update_track(tid_i, keypoints)  # FIX: preserve module_c_action API usage

                    if len(keypoint_buffers[tid_i]) >= MIN_FRAMES:  # FIX: only infer when enough temporal context exists
                        if _infer_counter % INFER_EVERY == 0:  # FIX:
                            last_actions[tid_i] = recognizer.predict(tid_i)  # FIX: keep existing recognizer signature
                        else:
                            last_actions.setdefault(tid_i, (-1, 0.0, "unknown"))  # FIX: hold last stable result between infer ticks
                    else:
                        last_actions[tid_i] = (-1, 0.0, "unknown")  # FIX: avoid premature guesses on short buffers

                for tid in list(keypoint_buffers.keys()):  # FIX: prune inactive tracks to prevent stale memory growth
                    if tid not in active_ids:
                        del keypoint_buffers[tid]  # FIX:
                        last_actions.pop(tid, None)  # FIX:

                for i, tid in enumerate(track_ids):
                    tid_i = int(tid)  # REFACTOR:
                    bbox = bboxes[i]  # REFACTOR:
                    label_id, conf_val, label_name = last_actions.get(tid_i, (-1, 0.0, "unknown"))  # FIX: consume cached action decisions instead of inline infer
                    color = LABEL_COLORS.get(label_id, (200, 200, 200))  # REFACTOR:

                    if DRAW_SKELETON:
                        kpts_draw = extract_kpts_for_track(result, tid_i, W, H)  # REFACTOR:
                        if kpts_draw is not None:
                            draw_skeleton(frame, kpts_draw, color, W, H)  # REFACTOR:
                    if DRAW_LABEL:
                        draw_action_label(frame, bbox, tid_i, label_name, conf_val, color)  # REFACTOR:

                    x1, y1, x2, y2 = bbox.astype(int)  # REFACTOR:
                    pending_io.append(io_executor.submit(_csv_write_row, [src_frame_idx, tid_i, label_name, f"{conf_val:.4f}", x1, y1, x2, y2]))  # REFACTOR:

                    if label_name not in ("?", "unknown"):  # FIX: ignore non-action placeholders in aggregate counts
                        action_counts[label_name] += 1  # REFACTOR:
                    if label_name == "Fall":
                        session.fall_count += 1  # REFACTOR:

                    track_ids_list.append(tid_i)  # REFACTOR:
                    boxes_payload.append((bbox, float(confs[i])))  # REFACTOR:
                    actions_payload.append(label_name)  # REFACTOR:

            t3 = time.perf_counter()  # REFACTOR:

            recognizer.remove_stale_tracks(active_ids)  # REFACTOR:
            for stale_tid in list(last_actions.keys()):  # FIX: keep local action cache consistent with active track set
                if stale_tid not in active_ids:
                    last_actions.pop(stale_tid, None)  # FIX:

            if writer is not None:
                writer.write(frame.copy())  # REFACTOR:

            if len(pending_io) > 256:
                pending_io = [f for f in pending_io if not f.done()]  # REFACTOR:

            now = time.time()  # REFACTOR:
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - t_prev, 1e-6))  # REFACTOR:
            t_prev = now  # REFACTOR:

            processed_idx += 1  # REFACTOR:
            session.total_frames = processed_idx  # REFACTOR:

            if processed_idx % STREAM_EVERY == 0:
                fall_alert = any(a.lower().find("fall") >= 0 for a in actions_payload)  # REFACTOR:
                inference_w = frame.shape[1]  # FIX: report inference-space width from processed frame
                inference_h = frame.shape[0]  # FIX: report inference-space height from processed frame
                payload = build_payload(  # REFACTOR:
                    frame_idx=src_frame_idx,
                    fps=fps_display,
                    track_ids=track_ids_list,
                    boxes=boxes_payload,
                    actions=actions_payload,
                    fall_alert=fall_alert,
                    action_counts=dict(action_counts),
                    inference_width=inference_w,
                    inference_height=inference_h,
                    source_fps=fps_src,
                    source_time_sec=(src_time_ms / 1000.0),
                    timestamp=(src_frame_idx / fps_src) if fps_src else 0.0,  # FIX: canonical playback timestamp (seconds)
                )
                _push_meta(session, payload, loop)  # REFACTOR:

            t4 = time.perf_counter()  # REFACTOR:
            if PERF_LOG:
                print(  # REFACTOR:
                    f"[PERF] decode={1000*(t1-t0):.1f}ms | yolo={1000*(t2-t1):.1f}ms | "
                    f"action={1000*(t3-t2):.1f}ms | encode={1000*(t4-t3):.1f}ms | "
                    f"total={1000*(t4-t0):.1f}ms"
                )

        for fut in pending_io:
            try:
                fut.result(timeout=5)  # REFACTOR:
            except Exception:
                pass

    except Exception as exc:
        session.status = "error"  # REFACTOR:
        session.error_msg = str(exc)  # REFACTOR:
        print(f"[PIPELINE] ERROR: {exc}")
        traceback.print_exc()
        _push_meta(session, {"type": "status", "status": "error", "message": str(exc)}, loop)  # REFACTOR:
        return

    finally:
        decode_stop.set()  # REFACTOR:
        if decode_thread is not None:
            decode_thread.join(timeout=5)  # REFACTOR:
        if cap is not None:
            cap.release()  # REFACTOR:
        if writer is not None:
            writer.release()  # REFACTOR:
        if csv_f is not None:
            csv_f.close()  # REFACTOR:

    _push_meta(session, {"type": "status", "status": session.status, "message": ""}, loop)  # REFACTOR:


def start_pipeline_thread(session: RunSession, model_path: str,
                          conf: float, imgsz: int,
                          loop: asyncio.AbstractEventLoop) -> threading.Thread:  # REFACTOR:
    t = threading.Thread(target=run_pipeline,
                         args=(session, model_path, conf, imgsz, loop),
                         daemon=True)  # REFACTOR:
    session.thread = t  # REFACTOR:
    t.start()  # REFACTOR:
    return t  # REFACTOR:
