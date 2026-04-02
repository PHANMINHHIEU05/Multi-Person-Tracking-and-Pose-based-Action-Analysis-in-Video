from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the YOLOv8 pose model to a TensorRT engine for the PyQt6 app.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "yolov8n-pose.pt",
        help="Source YOLO pose `.pt` weights to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "yolov8n-pose.engine",
        help="Target TensorRT engine path.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size. Keep this aligned with the PyQt6 runtime default unless you know you want another size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="TensorRT export batch size.",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="TensorRT workspace size in GB.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device id for export, for example `0`.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable FP16 export and keep TensorRT engine in FP32.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic shapes during export.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    weights_path = args.weights.resolve()
    output_path = args.output.resolve()

    if weights_path.suffix.lower() != ".pt":
        raise SystemExit("TensorRT export requires a `.pt` source model.")
    if not weights_path.exists():
        raise SystemExit(f"Pose weights not found: {weights_path}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to export a TensorRT engine.")

    missing_packages = []
    dependency_checks = [
        ("onnx", "onnx<2.0.0"),
        ("onnxslim", "onnxslim>=0.1.71"),
        ("tensorrt", "tensorrt-cu12>=7.0.0"),
    ]
    for module_name, package_name in dependency_checks:
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(package_name)
    if missing_packages:
        missing_str = " ".join(missing_packages)
        compatibility_hint = ""
        if importlib.util.find_spec("tensorrt") is None and sys.version_info >= (3, 13):
            compatibility_hint = (
                f"\nCurrent project virtualenv uses Python {sys.version_info.major}.{sys.version_info.minor}. "
                "If `tensorrt-cu12` does not install cleanly here, rerun this export from a dedicated "
                "RTX export environment that uses a TensorRT-friendly Python version "
                "(commonly 3.10-3.12), then copy the generated `yolov8n-pose.engine` back into the project root."
            )
        raise SystemExit(
            "Missing TensorRT export dependencies in the project virtualenv. "
            f"Install them first with:\n./.venv/bin/pip install {missing_str}{compatibility_hint}"
        )

    # Try to preload packaged TensorRT shared libraries when available.
    if importlib.util.find_spec("tensorrt_libs") is not None:
        __import__("tensorrt_libs")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    exported_path = Path(
        model.export(
            format="engine",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workspace=args.workspace,
            half=not args.fp32,
            dynamic=args.dynamic,
            verbose=True,
        )
    ).resolve()

    if exported_path != output_path:
        shutil.copy2(exported_path, output_path)

    print(f"TensorRT engine ready: {output_path}")


if __name__ == "__main__":
    main()
