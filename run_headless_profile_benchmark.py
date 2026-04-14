from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from pyqt_app import InferenceWorker, RuntimeConfig
from src.runtime_shared import resolve_default_action_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyQt inference worker headlessly across quality/fast/balanced profiles."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--profiles",
        default="quality,fast,balanced",
        help="Comma-separated profile names in order",
    )
    parser.add_argument(
        "--out_json",
        default="",
        help="Optional explicit output JSON path for combined summaries",
    )
    return parser.parse_args()


def _base_config(video_path: str) -> RuntimeConfig:
    # Force PyTorch pose weights for stable cross-profile benchmarking in environments
    # where TensorRT bindings may exist but CUDA runtime is unavailable.
    pose_pt = str((Path(__file__).resolve().parent / "yolov8n-pose.pt").resolve())
    return RuntimeConfig(
        source_mode="video",
        video_path=video_path,
        camera_index=0,
        pose_weights=pose_pt,
        action_model_path=resolve_default_action_model_path(),
        tracker_name="BoT-SORT (custom)",
        det_conf=0.25,
        det_iou=0.50,
        imgsz=640,
        max_det=12,
        draw_skeleton=True,
        live_preview=False,
        preview_width=960,
        preview_height=540,
        webcam_duration_sec=30,
        preview_stride=8,
        process_stride=1,
        output_scale=1.0,
        save_output_video=False,
        skip_action_model=False,
        normalize_timing=False,
        auto_tune_cpu=False,
        target_analysis_fps=12.0,
        min_track_frames=8,
        pred_stride=1,
        action_conf=0.30,
        smooth_window=2,
        fall_conf_boost=0.10,
        sitting_conf_penalty=0.22,
        keypoint_integrity_ratio=0.68,
        keypoint_jitter_ratio=0.18,
        fall_priority_prob=0.46,
        fall_velocity_ratio=0.12,
        sitting_hold_frames=5,
        track_time_budget_ms=12.0,
        fast_track_threshold=6,
    )


def apply_profile(cfg: RuntimeConfig, profile_name: str) -> RuntimeConfig:
    profile = profile_name.strip().lower()
    if profile == "balanced":
        cfg.tracker_name = "BoT-SORT (custom)"
        cfg.det_conf = 0.30
        cfg.det_iou = 0.50
        cfg.imgsz = 640
        cfg.max_det = 12
        cfg.process_stride = 1
        cfg.preview_stride = 4
        cfg.output_scale = 1.0
        cfg.normalize_timing = True
        cfg.target_analysis_fps = 12.0
        cfg.pred_stride = 3
        cfg.min_track_frames = 9
        cfg.action_conf = 0.30
        cfg.smooth_window = 3
        cfg.fall_conf_boost = 0.08
        cfg.sitting_conf_penalty = 0.20
        cfg.keypoint_integrity_ratio = 0.70
        cfg.keypoint_jitter_ratio = 0.15
        cfg.fall_priority_prob = 0.44
        cfg.fall_velocity_ratio = 0.12
        cfg.sitting_hold_frames = 5
        cfg.track_time_budget_ms = 10.0
        cfg.fast_track_threshold = 5
        cfg.auto_tune_cpu = True
        return cfg

    if profile == "fast":
        cfg.tracker_name = "ByteTrack (custom)"
        cfg.det_conf = 0.28
        cfg.det_iou = 0.45
        cfg.imgsz = 480
        cfg.max_det = 20
        cfg.process_stride = 2
        cfg.preview_stride = 5
        cfg.output_scale = 0.75
        cfg.normalize_timing = True
        cfg.target_analysis_fps = 10.0
        cfg.pred_stride = 4
        cfg.min_track_frames = 8
        cfg.action_conf = 0.31
        cfg.smooth_window = 3
        cfg.fall_conf_boost = 0.08
        cfg.sitting_conf_penalty = 0.20
        cfg.keypoint_integrity_ratio = 0.68
        cfg.keypoint_jitter_ratio = 0.18
        cfg.fall_priority_prob = 0.42
        cfg.fall_velocity_ratio = 0.10
        cfg.sitting_hold_frames = 4
        cfg.track_time_budget_ms = 8.0
        cfg.fast_track_threshold = 5
        cfg.auto_tune_cpu = True
        return cfg

    # quality default
    cfg.tracker_name = "BoT-SORT (custom)"
    cfg.pose_weights = str((Path(__file__).resolve().parent / "yolov8n-pose.pt").resolve())
    cfg.det_conf = 0.25
    cfg.det_iou = 0.50
    cfg.imgsz = 960
    cfg.max_det = 16
    cfg.process_stride = 1
    cfg.preview_stride = 1
    cfg.output_scale = 1.0
    cfg.normalize_timing = False
    cfg.target_analysis_fps = 12.0
    cfg.pred_stride = 2
    cfg.min_track_frames = 8
    cfg.action_conf = 0.30
    cfg.smooth_window = 2
    cfg.fall_conf_boost = 0.10
    cfg.sitting_conf_penalty = 0.22
    cfg.keypoint_integrity_ratio = 0.68
    cfg.keypoint_jitter_ratio = 0.18
    cfg.fall_priority_prob = 0.46
    cfg.fall_velocity_ratio = 0.12
    cfg.sitting_hold_frames = 5
    cfg.track_time_budget_ms = 12.0
    cfg.fast_track_threshold = 6
    cfg.auto_tune_cpu = False
    return cfg


def main() -> None:
    args = parse_args()
    video_path = str(Path(args.video).expanduser().resolve())
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    if not profiles:
        raise ValueError("No profiles specified.")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined: List[Dict[str, object]] = []

    print("=" * 72)
    print("HEADLESS PROFILE BENCHMARK")
    print("=" * 72)
    print(f"Video: {video_path}")
    print(f"Profiles: {profiles}")

    for profile in profiles:
        cfg = apply_profile(_base_config(video_path), profile)
        worker = InferenceWorker(cfg)
        print("-" * 72)
        print(f"Running profile: {profile}")
        summary = worker._run_inference()
        item = {
            "profile": profile,
            "config": asdict(cfg),
            "summary": summary,
        }
        combined.append(item)
        print(
            f"Done {profile}: FPS={summary.get('fps', 0):.2f}, "
            f"action_counts={summary.get('action_counts', {})}, "
            f"timeline={summary.get('fall_debug_timeline_path')}"
        )

    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
    else:
        out_path = (Path("runs/qt_outputs") / f"profile_benchmark_headless_{run_ts}.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"video": video_path, "runs": combined}, indent=2), encoding="utf-8")

    print("-" * 72)
    print(f"Saved combined summary: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
