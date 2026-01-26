"""
Environment Test for:
- YOLOv8 (ultralytics)
- OpenCV
- ByteTrack dependencies (lap, cython_bbox)
- MediaPipe Pose
- Scikit-learn / Pandas / SciPy

Run:
    python env_test.py
"""

def test_basic():
    print("=== BASIC LIBS ===")
    import numpy as np
    import cv2
    print("NumPy:", np.__version__)
    print("OpenCV:", cv2.__version__)
    print("Basic libs OK\n")


def test_yolo():
    print("=== YOLOv8 ===")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("YOLOv8 loaded OK\n")


def test_tracking_deps():
    print("=== TRACKING DEPS (ByteTrack) ===")
    import lap
    import cython_bbox
    print("lap version:", lap.__version__ if hasattr(lap, "__version__") else "OK")
    print("cython_bbox OK\n")


def test_mediapipe():
    print("=== MEDIAPIPE POSE ===")
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False
    )
    print("MediaPipe Pose initialized OK\n")


def test_ml_stack():
    print("=== ML STACK ===")
    import sklearn
    import pandas as pd
    import scipy
    print("scikit-learn:", sklearn.__version__)
    print("pandas:", pd.__version__)
    print("scipy:", scipy.__version__)
    print("ML stack OK\n")


if __name__ == "__main__":
    print("\n===== ENVIRONMENT TEST START =====\n")
    test_basic()
    test_yolo()
    test_tracking_deps()
    test_mediapipe()
    test_ml_stack()
    print("===== ALL TESTS PASSED =====\n")
