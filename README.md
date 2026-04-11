# Multi-Person Tracking & Pose-based Action Analysis

## 1. Project Overview

This project implements a real-time multi-person tracking and pose-based action recognition system. It detects people in a video stream, maintains consistent identities (IDs) across frames, estimates body pose, and classifies actions per person using a trained pose-sequence model.

The system is built as a modular computer vision pipeline combining YOLOv8, BoT-SORT tracking, and pose-sequence action recognition models. The active desktop runtime now uses the repaired ExtraTrees 6-class artifact internally, while presenting a simpler 5-action view in the UI.

As of April 9, 2026, the runtime strategy has been adjusted for higher real-video stability: the app now restores the stronger 6-class internal artifact and uses label display mapping so users still see `Fall / Standing / Walking / Sitting / Lying_Down`, while the internal `Bending` class acts as an ambiguity buffer for occluded or transitional poses.

**Core capabilities:**
- Detect and track multiple persons simultaneously with stable IDs
- Handle partial occlusions and ID switches via Feature Memory Bank
- Extract 17-keypoint COCO pose per tracked person every frame
- Classify actions in real-time: **Fall | Standing | Walking | Sitting | Lying Down**
- Operates on CPU-only systems (no GPU required)

---

## 2. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the primary PyQt6 desktop UI
source .trt-export-venv/bin/activate
python pyqt_app.py

# Fallback runtime without TensorRT
source .venv/bin/activate
python pyqt_app.py

# Optional: export TensorRT pose engine on an NVIDIA RTX machine
# Uses the dedicated Python 3.12 TensorRT export environment created for this project.
source .trt-export-venv/bin/activate
python export_pose_engine.py

# Rebuild the clean 5-class focus dataset
source .venv/bin/activate
python repair_action_dataset.py \
  --mode five_action_round4 \
  --base_dir data/train_ready_action_repair_v2_unicomfacauca \
  --sit_external_dir data/train_ready_action_repair_v4_sit_only \
  --out_dir data/train_ready_action_repair_v6_five_action_round4

# Add focused hard-case walking augmentations for the 5-class task
python prepare_action_hardcases_round2.py \
  --base_dir data/train_ready_action_repair_v6_five_action_round4 \
  --out_dir data/train_ready_action_repair_v6_five_action_round4_hardcases \
  --walk_partial_body_copies 120 \
  --bending_boundary_copies 0

# Retrain the current active ExtraTrees artifact
python train_extratrees_action.py \
  --data_dir data/train_ready_action_repair_v6_five_action_round4_hardcases \
  --out_dir runs/train_extratrees_action_repair_v6_five_action_round4_hardcases_v1spec \
  --feature_spec mean,std,min,max,first,last,delta,q25,q75,abs_vel_mean,vel_std

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

## 2.1 Desktop UI

`pyqt_app.py` is the local UI for this project.
It is better suited for low-latency preview because frames are rendered directly in a Qt window instead of being pushed through browser updates.

Current PyQt6 UI includes:
- Upload video mode
- Webcam mode
- Live preview in a native desktop window
- Action recognition toggle
- Tracking/detection/runtime controls
- Annotated output video saving for uploaded videos

If `yolov8n-pose.engine` exists in the project root, the PyQt6 app auto-prefers it when the current Python environment has TensorRT available.
If TensorRT is not available in the current environment, the app falls back to `yolov8n-pose.pt`.

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
    YOLOv8n-Pose + BoT-SORT + Action Recognition
    → actions.csv, video_action.mp4
```

`main.py` is the unified entry point that delegates to Module B or Module C based on `--mode`.

---

## 4. Action Recognition Model

| Property | Value |
|----------|-------|
| Active runtime model | ExtraTrees on 128x69 pose-sequence features |
| Active artifact | `runs/train_extratrees_action_repair_v2_unicomfacauca_v1spec/extratrees_model.joblib` |
| Runtime selector | `runs/active_action_model_path.txt` |
| Input shape | (N, 128, 69) — 17 keypoints × 4 features + aspect ratio |
| Internal classes | Fall / Standing / Walking / Sitting_Quickly / Bending / Lying_Down |
| UI display classes | Fall / Standing / Walking / Sitting / Lying_Down |
| Legacy research model | Bi-GRU (3 layers, 128 hidden) + Multi-Head Self-Attention |
| Legacy checkpoint | `runs/train_v3/final_safe_system.pth` |

**Current active 5-class repair pipeline:**
```bash
python repair_action_dataset.py \
  --mode five_action_round4 \
  --base_dir data/train_ready_action_repair_v2_unicomfacauca \
  --sit_external_dir data/train_ready_action_repair_v4_sit_only \
  --out_dir data/train_ready_action_repair_v6_five_action_round4

python prepare_action_hardcases_round2.py \
  --base_dir data/train_ready_action_repair_v6_five_action_round4 \
  --out_dir data/train_ready_action_repair_v6_five_action_round4_hardcases \
  --walk_partial_body_copies 120 \
  --bending_boundary_copies 0

python train_extratrees_action.py \
  --data_dir data/train_ready_action_repair_v6_five_action_round4_hardcases \
  --out_dir runs/train_extratrees_action_repair_v6_five_action_round4_hardcases_v1spec \
  --feature_spec mean,std,min,max,first,last,delta,q25,q75,abs_vel_mean,vel_std
```

The current active artifact is selected through `runs/active_action_model_path.txt`, so the PyQt6 app and runtime helpers can move to a better ExtraTrees artifact without hard-coding a single training directory. The desktop runtime currently aliases `Sitting_Quickly -> Sitting` for display and suppresses direct `Bending` display so ambiguous poses are less likely to be forced into the wrong visible action.

**Alternative 5-class experiment kept for reference:**
- `runs/train_extratrees_action_repair_v6_five_action_round4_hardcases_v1spec/extratrees_model.joblib`
- grouped-CV macro-F1: `0.7929`
- useful for offline comparison, but the runtime was moved back to the 6-class internal artifact for better real-video ambiguity handling

**Experimental taxonomy round 3 (not active):**
```bash
python repair_action_dataset.py \
  --mode taxonomy_round3 \
  --base_dir data/train_ready_action_repair_v2_unicomfacauca \
  --sit_external_dir data/train_ready_action_repair_v4_sit_only \
  --out_dir data/train_ready_action_repair_v5_taxonomy_round3

python train_extratrees_action.py \
  --data_dir data/train_ready_action_repair_v5_taxonomy_round3 \
  --out_dir runs/train_extratrees_action_repair_v5_taxonomy_round3_v3physics \
  --feature_spec v3_physics_pose_stats
```

This round introduced `Sitting` as a seventh class and improved some seated-static holdout slices, but it did not beat the now-active 5-class focus artifact on grouped validation, so it remains a non-promoted experiment.

**Optional hardcase round-2 experiment:**
```bash
python prepare_action_hardcases_round2.py
python train_extratrees_action.py \
  --data_dir data/train_ready_action_repair_v3_hardcases \
  --out_dir runs/train_extratrees_action_repair_v3_hardcases_v1spec \
  --feature_spec mean,std,min,max,first,last,delta,q25,q75,abs_vel_mean,vel_std
```

This experiment is currently kept as a non-active candidate because it improved synthetic hardcases but remained below the promoted 5-class runtime artifact.

**Legacy deep-model training pipeline:**
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
    --model_path runs/train_extratrees_action_repair_v6_five_action_round4_hardcases_v1spec/extratrees_model.joblib
```

---

## 7. Dataset

- **UR Fall Detection Dataset** (`data/UR_Fall/`) — image sequences, multiple ADL/fall scenarios
- **Multicam Dataset** (`data/Multicam/`) — video clips with fall/non-fall annotations
- **Unicomfacauca Dataset Modification** (`data/external/unicomfacauca/extracted/Fall-Detection-Dataset-Modification/`) — targeted external supplement now ingested conservatively for `Standing`, `Walking`, and `Lying_Down`
- Processed keypoints: `data/processed_pose/*.npy` (shape: `(num_frames, 17, 2)`)
- Training-ready: `data/train_ready_horizontal/` (shape: `(N, 128, 69)`)
- Previous repaired 6-class train-ready dataset: `data/train_ready_action_repair_v2_unicomfacauca/`
- Active focused 5-class train-ready dataset: `data/train_ready_action_repair_v6_five_action_round4_hardcases/`

---

## 8. Reproducibility

```bash
pip install -r requirements.txt
python main.py --video data/video/video1.mp4 --out runs/action/demo --device cpu
```

The system produces deterministic results on CPU. All components are modular and independently runnable.
