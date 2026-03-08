# Multi-Person Tracking & Pose-based Action Analysis

## 1. Project Overview

This project implements a real-time multi-person tracking and pose-based action recognition system. It detects people in a video stream, maintains consistent identities (IDs) across frames, estimates body pose, and classifies actions per person using a trained deep learning model.

The system is built as a modular computer vision pipeline combining YOLOv8, BoT-SORT tracking, and a Bi-GRU + Multi-Head Self-Attention action recognition model.

**Core capabilities:**
- Detect and track multiple persons simultaneously with stable IDs
- Handle partial occlusions and ID switches via Feature Memory Bank
- Extract 17-keypoint COCO pose per tracked person every frame
- Classify actions in real-time: **Fall | Walking | Sitting Quickly | Bending | Lying Down**
- Operates on CPU-only systems (no GPU required)

---

## 2. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (action recognition) on a video
python main.py --video data/video/video1.mp4

# Specify output directory
python main.py --video data/video/video1.mp4 --out runs/action/my_run

# Tracking only (no action recognition)
python main.py --video data/video/video1.mp4 --mode track --out runs/track/my_run

# Use GPU
python main.py --video data/video/video1.mp4 --device cuda

# Show real-time preview
python main.py --video data/video/video1.mp4 --preview
```

**Outputs** (saved to `--out` directory):
- `video_action.mp4` — annotated video with bounding boxes, skeleton, action labels
- `actions.csv` — per-frame: `frame, track_id, action, confidence, x1, y1, x2, y2`
- `summary.json` — run metadata

---

## 3. System Architecture

```
Video Input
    │
    ▼
[Module A]  src/module_a_detect.py
    YOLOv8n — detection only (CSV/JSONL + annotated video)
    │
    ▼
[Module B]  src/module_b_botsort_stable.py
    YOLOv8n + BoT-SORT + Feature Memory Bank + Offline Merge
    → tracks.csv, tracks_merged.csv, video_track.mp4
    │
    ▼
[Module C]  src/module_c_action.py           ← main pipeline
    YOLOv8n-Pose + BoT-SORT + Bi-GRU Action Recognition
    → actions.csv, video_action.mp4
```

`main.py` is the unified entry point that delegates to Module B or Module C based on `--mode`.

---

## 4. Action Recognition Model

| Property | Value |
|----------|-------|
| Architecture | Bi-GRU (3 layers, 128 hidden) + Multi-Head Self-Attention |
| Loss | Focal Loss (γ=2.0) + class weights |
| Optimiser | AdamW + CosineAnnealingWarmRestarts |
| Input shape | (N, 128, 69) — 17 keypoints × 4 features + aspect ratio |
| Classes | Fall / Walking / Sitting_Quickly / Bending / Lying_Down |
| Checkpoint | `runs/train_v3/final_safe_system.pth` |

**Training pipeline:**
```bash
python extract_pose.py                 # Extract keypoints from UR_Fall + Multicam
python data_prepare_v3.py             # Build training data (128 frames, 69 features)
python train_professional_v3.py       # Train Bi-GRU model → runs/train_v3/
```

---

## 5. Computational Constraints

Designed for laptop-class hardware without GPU:

| Constraint | Value |
|------------|-------|
| Hardware | CPU-only (no CUDA required) |
| Detection model | YOLOv8n (lightweight) |
| Inference speed | ~5–7 f/s on CPU (640px), ~3–4 f/s (1080p+) |
| Max tracked persons | 50 (configurable via `--max_det`) |
| Action warmup | 80 frames per track before first prediction |

---

## 6. Module Reference

Run individual modules directly:

```bash
# Module A — detection only
python src/module_a_detect.py --video data/video/input.mp4 --out runs/detect/run1

# Module B — tracking only (BoT-SORT + Memory Bank)
python src/module_b_botsort_stable.py --video data/video/input.mp4 --out runs/track/run1

# Module C — pose + action recognition
python src/module_c_action.py --video data/video/input.mp4 --out runs/action/run1 \
    --model_path runs/train_v3/final_safe_system.pth
```

---

## 7. Dataset

- **UR Fall Detection Dataset** (`data/UR_Fall/`) — image sequences, multiple ADL/fall scenarios
- **Multicam Dataset** (`data/Multicam/`) — video clips with fall/non-fall annotations
- Processed keypoints: `data/processed_pose/*.npy` (shape: `(num_frames, 17, 2)`)
- Training-ready: `data/train_ready_horizontal/` (shape: `(N, 128, 69)`)

---

## 8. Reproducibility

```bash
pip install -r requirements.txt
python main.py --video data/video/video1.mp4 --out runs/action/demo --device cpu
```

The system produces deterministic results on CPU. All components are modular and independently runnable.
