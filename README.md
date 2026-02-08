Multi-Object Tracking & Real-Time Behavior Analysis System

1. Project Overview

This project implements a real-time multi-object human monitoring system designed to not only detect people in a video stream, but also maintain consistent identities (IDs) over time and provide a foundation for higher-level behavior analysis.

The system is built as a multi-stage computer vision pipeline, combining modern deep learning models with classical state estimation techniques.
In the current stage, the project focuses on robust multi-object tracking under resource-constrained environments (CPU-only systems), ensuring system stability and reproducibility.

Core objectives:

Detect humans in video streams using a state-of-the-art object detector.

Track multiple individuals simultaneously while maintaining stable IDs.

Handle partial occlusions and crossings between individuals.

Operate in near real-time on systems without dedicated GPU hardware.

2. System Architecture (High-Level Pipeline)

The system follows a modular pipeline design:

Object Detection
Human detection is performed using YOLOv8 (Ultralytics), configured to detect only the person class for efficiency.

Multi-Object Tracking
Detected bounding boxes are passed to ByteTrack, which integrates Kalman Filter-based motion prediction to maintain consistent identities across frames, even during short-term occlusions or missed detections.

Visualization & Monitoring
Each tracked individual is rendered with a bounding box and a unique ID in the output video stream, allowing real-time inspection of tracking stability.

Pose estimation and behavior classification are intentionally excluded in this phase to isolate and validate the tracking backbone.

3. Computational Constraints & Design Assumptions

This project is designed to run on laptop-class machines without GPU acceleration.
To prevent hardware overload and ensure stable execution, the following constraints and design choices are enforced:

Hardware Assumptions

CPU-only execution (no CUDA / GPU acceleration)

Consumer-grade laptop hardware

Limited thermal headroom

Enforced System Constraints

Input resolution: downscaled to 640×360 or 854×480

Model size: lightweight detector (yolov8n)

Target processing rate: 10–15 FPS

Maximum tracked objects: ≤ 5 persons

Video duration for demo/testing: 30–60 seconds

Performance Optimization Strategies

Frame skipping: detection is performed every N frames, with intermediate frames handled via Kalman Filter prediction.

Class filtering: only the person class is processed.

Confidence thresholding: low-confidence detections are discarded to reduce tracker load.

Minimal visualization overhead: only bounding boxes and IDs are rendered.

These constraints allow the system to remain responsive while avoiding sustained high CPU load or thermal stress.

4. Dataset & Input Videos

The system operates on publicly available video sources intended for academic and research use, including:

Stock videos featuring pedestrian movement in indoor or semi-controlled environments.

Public surveillance-style footage with fixed camera viewpoints.

Benchmark datasets commonly used in multi-object tracking research.

All input videos are preprocessed (resolution and duration) prior to inference to ensure compatibility with CPU-only execution.

5. Project Scope (Current Stage)
   Included:

Human detection

Multi-object tracking with stable identity assignment

Real-time visualization

Performance evaluation under resource constraints

Explicitly Excluded (Future Work):

Pose estimation

Action / behavior classification

Statistical behavior analytics

GPU acceleration

This staged development approach ensures a solid and verifiable tracking backbone before introducing higher-level semantic analysis.

6. Reproducibility & Stability

The project emphasizes:

Modular code structure

Explicit dependency management

Conservative runtime configuration

Hardware-aware optimization

These principles ensure that the system can be reproduced and demonstrated reliably across different CPU-only environments.
