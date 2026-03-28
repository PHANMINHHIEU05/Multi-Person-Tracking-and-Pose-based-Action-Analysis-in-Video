"""
main.py – Unified Pipeline Entry Point
=======================================
Multi-Person Tracking & Pose-based Action Analysis

Modes
-----
  action (default)  — YOLOv8n-Pose + BoT-SORT + Bi-GRU action recognition
  track             — YOLOv8n + BoT-SORT tracking only

Input Sources
-------------
  • Video file:    data/video/video1.mp4
  • Webcam:        0 (default), 1, 2, ... (camera index)
  • RTSP stream:   rtsp://user:pass@camera_ip:554/stream
  • HTTP stream:   http://camera_ip:port/mjpeg or rtmp://...

Usage Examples
--------------
  # File – Full pipeline (action recognition)
  python main.py --video data/video/video1.mp4

  # Webcam – Live tracking + action recognition with preview
  python main.py --video 0 --preview

  # Webcam – Tracking only, higher resolution
  python main.py --video 0 --mode track --imgsz 1280 --preview

  # RTSP stream from IP camera
  python main.py --video "rtsp://admin:password@192.168.1.100:554/stream" --preview

  # Specify output and device
  python main.py --video 0 --out runs/action/webcam_demo --device cuda --preview

  # Disable skeleton overlay (faster on slow networks)
  python main.py --video 0 --no_skeleton --preview

  # Adjust detection confidence (lower = more detections)
  python main.py --video 0 --conf 0.15 --preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-Person Tracking & Pose-based Action Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core I/O
    p.add_argument("--video",   required=True,
                   help="Path to input video file")
    p.add_argument("--out",     default="runs/action/run1",
                   help="Output directory for all results")
    p.add_argument("--device",  default="auto",
                   help="Device: auto | cpu | cuda | 0 (GPU index)")

    # Pipeline mode
    p.add_argument("--mode",    default="action",
                   choices=["action", "track"],
                   help="action: full pipeline with action recognition | "
                        "track: BoT-SORT tracking only (no action)")

    # Action recognition options (mode=action)
    p.add_argument("--model_path",  default="runs/train_horizontal/final_safe_system.pth",
                   help="Trained Bi-GRU checkpoint (.pth) — mode=action only")
    p.add_argument("--pose_model",  default="yolov8n-pose.pt",
                   help="YOLOv8-Pose weights — mode=action only")

    # Tracking options (mode=track)
    p.add_argument("--det_model",   default="yolov8n.pt",
                   help="YOLOv8 detection weights — mode=track only")
    p.add_argument("--tracker",     default="config/botsort_custom.yaml",
                   help="Tracker config yaml — mode=track only")

    # Shared inference parameters
    p.add_argument("--conf",    type=float, default=0.25,
                   help="YOLO confidence threshold")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="IoU threshold for NMS")
    p.add_argument("--imgsz",   type=int,   default=640,
                   help="Inference image size (use 1280 for small/distant people)")
    p.add_argument("--max_det", type=int,   default=50,
                   help="Maximum persons to detect per frame")

    # Display
    p.add_argument("--preview",     action="store_true",
                   help="Show real-time preview window")
    p.add_argument("--no_skeleton", action="store_true",
                   help="Disable skeleton drawing on output video (mode=action only)")

    return p


# --------------------------------------------------------------------------- #
#  Mode: action (Module C)
# --------------------------------------------------------------------------- #
def _run_action(args: argparse.Namespace) -> None:
    """Full pipeline: pose estimation + tracking + action recognition (Module C)."""
    from src.module_c_action import build_parser as c_build_parser, process_video

    # Build module_c args from explicit list so we get all defaults correct,
    # then override boolean flags that don't have --no- variants in module_c.
    c_argv = [
        "--video",      args.video,
        "--out",        args.out,
        "--model_path", args.model_path,
        "--pose_model", args.pose_model,
        "--device",     args.device,
        "--conf",       str(args.conf),
        "--iou",        str(args.iou),
        "--imgsz",      str(args.imgsz),
        "--max_det",    str(args.max_det),
    ]
    if args.preview:
        c_argv.append("--preview")

    c_args = c_build_parser().parse_args(c_argv)
    c_args.draw_skeleton = not args.no_skeleton

    process_video(c_args)


# --------------------------------------------------------------------------- #
#  Mode: track (Module B)
# --------------------------------------------------------------------------- #
def _run_track(args: argparse.Namespace) -> None:
    """Tracking only: YOLOv8n + BoT-SORT (Module B)."""
    from src.module_b_botsort_stable import parse_args as b_parse_args, run_tracking

    # Patch sys.argv so that module_b's argparse reads the values we want,
    # picking up all module_b-specific defaults automatically.
    b_argv = [
        "module_b",
        "--video",   args.video,
        "--out",     args.out,
        "--model",   args.det_model,
        "--device",  args.device,
        "--tracker", args.tracker,
        "--conf",    str(args.conf),
        "--iou",     str(args.iou),
        "--imgsz",   str(args.imgsz),
        "--max_det", str(args.max_det),
    ]
    if args.preview:
        b_argv.append("--enable_preview")

    old_argv = sys.argv
    try:
        sys.argv = b_argv
        b_args = b_parse_args()
    finally:
        sys.argv = old_argv

    run_tracking(b_args)


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- Input validation ---
    if not Path(args.video).exists():
        print(f"[ERROR] Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "action":
        if not Path(args.model_path).exists():
            print(f"[ERROR] Action model not found: {args.model_path}", file=sys.stderr)
            print("[HINT ] Run train_professional_v3.py first, or pass --model_path",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[main] mode=action  |  video={args.video}  |  out={args.out}")
        _run_action(args)
    else:
        print(f"[main] mode=track   |  video={args.video}  |  out={args.out}")
        _run_track(args)


if __name__ == "__main__":
    main()
