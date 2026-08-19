# Multi-Person Tracking and Pose-based Action Recognition

## 1. Project Overview

This project is a real-time computer vision system for multi-person tracking and pose-based human action recognition. It detects people, estimates 17 COCO body keypoints, maintains track IDs across frames, and classifies each person's action from a temporal keypoint sequence.

Current visible action classes:

| ID | Action |
|---:|---|
| 0 | Fall |
| 1 | Standing |
| 2 | Walking |
| 3 | Sitting |
| 4 | Lying_Down |

Current active action model:

```text
runs/train_bigru_prod5_best_v1/final_safe_system.pth
```

The active model is a PyTorch Bi-GRU + Multi-Head Self-Attention model trained on pose-sequence features.

Core capabilities:

- Real-time person detection and pose estimation with YOLOv8n-Pose.
- Multi-person tracking with ByteTrack or BoT-SORT.
- Per-track keypoint buffering and temporal action recognition.
- PyQt6 desktop UI for video/webcam demo.
- Optional TensorRT acceleration for YOLO pose inference.
- Debug timeline output for analyzing model/runtime errors.

---

## 2. Main Runtime

Recommended PyQt6 runtime:

```bash
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .trt-export-venv/bin/activate
python pyqt_app.py
```

Fallback runtime without TensorRT environment:

```bash
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .venv/bin/activate
python pyqt_app.py
```

If installing from a clean machine:

```bash
pip install -r requirements.txt
python pyqt_app.py
```

The app automatically uses TensorRT pose inference if `yolov8n-pose.engine` exists and the current Python environment supports TensorRT. Otherwise, it falls back to PyTorch inference with `yolov8n-pose.pt`.

---

## 3. Runtime Pipeline

```text
Video / Webcam
    -> YOLOv8n-Pose person + keypoint detection
    -> ByteTrack or BoT-SORT tracking
    -> Track ID based keypoint buffer
    -> Feature sequence creation: 128 frames x 69 features
    -> Bi-GRU action model
    -> Runtime smoothing / state rules
    -> PyQt6 visualization + optional annotated output video
```

Feature sequence format:

| Feature group | Dimension |
|---|---:|
| 17 normalized keypoints x,y | 34 |
| 17 keypoint velocities | 17 |
| 17 keypoint accelerations | 17 |
| Bounding-box aspect ratio | 1 |
| Total per frame | 69 |

---

## 4. Desktop UI Profiles

The PyQt6 app provides three runtime profiles:

| Profile | Typical use | Tracker |
|---|---|---|
| Fast Mode | Best realtime smoothness | ByteTrack custom |
| RTX 3050 Balanced | Balance between stability and quality | BoT-SORT custom |
| Quality Mode | Higher pose resolution, slower | BoT-SORT custom |

In recent testing, Fast Mode with TensorRT pose backend has been the most practical realtime configuration.

---

## 5. Action Recognition Model

| Property | Current value |
|---|---|
| Active backend | PyTorch / torch |
| Active architecture | Bi-GRU + Multi-Head Self-Attention |
| Active artifact | `runs/train_bigru_prod5_best_v1/final_safe_system.pth` |
| Runtime selector | `runs/active_action_model_path.txt` |
| Input shape | `(N, 128, 69)` |
| Number of classes | 5 |
| Training script | `train_professional_v3.py` |

Model architecture:

- Input LayerNorm.
- Linear projection from 69 features to hidden dimension.
- 3-layer bidirectional GRU.
- Multi-Head Self-Attention pooling.
- MLP classifier for 5 action classes.

Training loss and optimization:

- Focal Loss with gamma = 2.0.
- Class weights for imbalanced classes.
- AdamW optimizer.
- CosineAnnealingWarmRestarts learning rate scheduler.
- Early stopping.
- Gradient clipping.
- CUDA mixed precision when available.

---

## 6. Active Training Dataset

Active training dataset:

```text
config/data/train_ready_bigru_prod5_best_v1
```

Dataset summary:

| Item | Value |
|---|---:|
| Samples | 5,636 |
| Sequence length | 128 frames |
| Feature dimension | 69 |
| Classes | 5 |

Class distribution:

| Class | Samples |
|---|---:|
| Fall | 1,665 |
| Standing | 476 |
| Walking | 1,720 |
| Sitting | 1,012 |
| Lying_Down | 763 |

Data sources:

| Source | Samples |
|---|---:|
| UR_Fall | 2,714 |
| Multicam | 1,048 |
| Augment_Fall | 504 |
| Multicam_AllCams | 451 |
| GMDCSA24 | 403 |
| Unicomfacauca | 339 |
| NTU_pseudo | 177 |

The dataset is stored as processed pose sequences, not raw RGB video. Raw datasets and train-ready arrays are excluded from the lightweight submission ZIP because they are large.

---

## 7. Training Command

Train the active Bi-GRU model:

```bash
python train_professional_v3.py \
  --data_dir config/data/train_ready_bigru_prod5_best_v1 \
  --save_dir runs/train_bigru_prod5_best_v1
```

Default training configuration:

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 8e-4 |
| Batch size | 64 |
| Max epochs | 300 |
| Early stopping patience | 15 |
| Dropout | 0.4 |
| Weight decay | 2e-4 |
| Validation split | 15% |

Best recorded validation result on the active dataset:

| Metric | Value |
|---|---:|
| Accuracy | 96.10% |
| Weighted F1 | 96.11% |
| Macro F1 | 96.09% |

---

## 8. TensorRT / PyTorch Pose Backend

Pose model files:

| File | Purpose |
|---|---|
| `yolov8n-pose.pt` | PyTorch YOLOv8 pose fallback |
| `yolov8n-pose.engine` | Optional TensorRT acceleration artifact |
| `yolov8n-pose.onnx` | Optional export intermediate |

TensorRT export command:

```bash
source .trt-export-venv/bin/activate
python export_pose_engine.py
```

Notes:

- TensorRT improves pose inference speed.
- TensorRT does not directly improve action classification accuracy.
- Action accuracy still depends on pose quality, tracking ID stability, sequence quality, and action model behavior.

---

## 9. Useful Commands

Check active action model:

```bash
cat runs/active_action_model_path.txt
```

Run headless profile benchmark:

```bash
source .trt-export-venv/bin/activate
python run_headless_profile_benchmark.py \
  --video "VideoTest/Human Fall Detection Sample.mp4" \
  --profiles fast,balanced,quality
```

Run classic CLI pipeline on a video:

```bash
python main.py --video data/video/video1.mp4 --out runs/action/demo --device cuda
```

Run tracking only:

```bash
python main.py --video data/video/video1.mp4 --mode track --out runs/track/demo
```

---

## 10. Repository Structure

| Path | Purpose |
|---|---|
| `pyqt_app.py` | Main PyQt6 desktop UI |
| `src/runtime_shared.py` | Shared runtime recognizer, tracker/action helpers, postprocess logic |
| `src/module_a_detect.py` | Detection-only module |
| `src/module_b_track.py` | Tracking module |
| `src/module_b_botsort.py` | BoT-SORT tracking implementation |
| `src/module_c_action.py` | CLI pose + action module |
| `train_professional_v3.py` | Bi-GRU action model training script |
| `train_extratrees_action.py` | ExtraTrees baseline training script |
| `export_pose_engine.py` | Optional TensorRT export script |
| `config/` | Tracker and runtime YAML configs |
| `runs/active_action_model_path.txt` | Selects active action model |
| `runs/train_bigru_prod5_best_v1/` | Active Bi-GRU model artifact and training outputs |
| `PROJECT_PRESENTATION_REPORT.md` | Detailed project report for presentation |
| `TEST_GUIDELINES.md` | Local testing instructions |

---

## 11. Submission ZIP Notes

The lightweight submission package should include:

- Source code.
- `requirements.txt`.
- `README.md`.
- `TEST_GUIDELINES.md`.
- `PROJECT_PRESENTATION_REPORT.md`.
- `yolov8n-pose.pt`.
- `runs/train_bigru_prod5_best_v1/final_safe_system.pth`.
- Training plots such as `confusion_matrix.png` and `training_curves.png` if available.
- Slide file, if included by the student.

The package should exclude:

- Virtual environments: `.venv/`, `venv/`, `.trt-export-venv/`.
- Raw datasets and train-ready arrays under `config/data/`.
- Runtime outputs under `runs/qt_outputs/`.
- Cache/build folders: `.pip-cache/`, `.tmp-build/`, `__pycache__/`.
- Large demo videos if a small submission is required.
- Optional generated artifacts such as `.onnx` or `.engine` if size is limited.

These excluded artifacts are reproducible or too large for submission. The final action model and source code are included.

---

## 12. Known Limitations

- Validation accuracy is high, but real videos can still fail because of pose missing, occlusion, fast scene cuts, and domain shift.
- Sitting, Standing, and Lying_Down can be confused when legs are occluded or the person is partially outside the frame.
- TensorRT improves FPS, but classification errors must be handled through better data, model training, and runtime state logic.
- Raw training datasets are not included in the lightweight submission ZIP because of size.

---

## 13. Short Project Description

This project builds an end-to-end human action recognition system using YOLOv8 pose estimation, multi-person tracking, and a temporal Bi-GRU action model. The system runs in a PyQt6 desktop application and supports optional TensorRT acceleration on NVIDIA GPUs. The current active model recognizes five actions: Fall, Standing, Walking, Sitting, and Lying_Down.
