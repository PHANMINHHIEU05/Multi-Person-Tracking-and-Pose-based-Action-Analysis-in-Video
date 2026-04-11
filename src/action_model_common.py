from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


LEGACY_ACTION_LABEL_MAP: Dict[int, str] = {
    0: "Fall",
    1: "Walking",
    2: "Sitting_Quickly",
    3: "Bending",
    4: "Lying_Down",
}

DEFAULT_ACTION_LABEL_MAP: Dict[int, str] = {
    0: "Fall",
    1: "Standing",
    2: "Walking",
    3: "Sitting_Quickly",
    4: "Bending",
    5: "Lying_Down",
}

TAXONOMY_REPAIR_ROUND3_LABEL_MAP: Dict[int, str] = {
    0: "Fall",
    1: "Standing",
    2: "Walking",
    3: "Sitting",
    4: "Sitting_Quickly",
    5: "Bending",
    6: "Lying_Down",
}

FIVE_ACTION_FOCUS_LABEL_MAP: Dict[int, str] = {
    0: "Fall",
    1: "Standing",
    2: "Walking",
    3: "Sitting",
    4: "Lying_Down",
}

ACTION_LABEL_COLORS_BY_NAME: Dict[str, tuple[int, int, int]] = {
    "Fall": (0, 0, 255),
    "Standing": (255, 215, 0),
    "Walking": (0, 200, 0),
    "Sitting": (180, 0, 255),
    "Sitting_Quickly": (255, 0, 200),
    "Bending": (255, 140, 0),
    "Lying_Down": (0, 200, 255),
}

EXTRATREES_FEATURE_SPEC_V1 = "mean,std,min,max,first,last,delta,q25,q75,abs_vel_mean,vel_std"
EXTRATREES_FEATURE_SPEC_V2 = "v2_temporal_tail_stats"
EXTRATREES_FEATURE_SPEC_V3_PHYSICS = "v3_physics_pose_stats"


def normalize_label_map(raw_map: Dict[int | str, str]) -> Dict[int, str]:
    return {int(k): str(v) for k, v in raw_map.items()}


def infer_label_map(label_ids: Iterable[int], label_names: Iterable[str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for label_id, label_name in zip(label_ids, label_names):
        mapping[int(label_id)] = str(label_name)
    return {int(k): mapping[k] for k in sorted(mapping)}


def build_label_colors(label_map: Dict[int, str]) -> Dict[int, tuple[int, int, int]]:
    return {int(label_id): get_action_color(label_name, int(label_id), label_map) for label_id, label_name in label_map.items()}


def get_action_color(
    label_name: Optional[str],
    label_id: Optional[int] = None,
    label_map: Optional[Dict[int, str]] = None,
) -> tuple[int, int, int]:
    if label_name:
        color = ACTION_LABEL_COLORS_BY_NAME.get(str(label_name))
        if color is not None:
            return color
    if label_id is not None and label_map is not None:
        mapped_name = label_map.get(int(label_id))
        if mapped_name:
            color = ACTION_LABEL_COLORS_BY_NAME.get(mapped_name)
            if color is not None:
                return color
    return (200, 200, 200)


def _build_extratrees_feature_vector_v1(seq69: np.ndarray) -> np.ndarray:
    q25 = np.quantile(seq69, 0.25, axis=0)
    q75 = np.quantile(seq69, 0.75, axis=0)
    vel = np.diff(seq69, axis=0)
    feat = np.concatenate(
        [
            seq69.mean(axis=0),
            seq69.std(axis=0),
            seq69.min(axis=0),
            seq69.max(axis=0),
            seq69[0],
            seq69[-1],
            (seq69[-1] - seq69[0]),
            q25,
            q75,
            np.mean(np.abs(vel), axis=0),
            np.std(vel, axis=0),
        ],
        axis=0,
    )
    return feat.astype(np.float32)


def _window_stats(seq69: np.ndarray, start: int, end: int) -> list[np.ndarray]:
    seg = seq69[start:end]
    if seg.shape[0] < 2:
        seg = np.repeat(seg[:1], 2, axis=0)
    vel = np.diff(seg, axis=0)
    return [
        seg.mean(axis=0),
        seg.std(axis=0),
        np.mean(np.abs(vel), axis=0),
    ]


def _build_extratrees_feature_vector_v2(seq69: np.ndarray) -> np.ndarray:
    base = _build_extratrees_feature_vector_v1(seq69)
    t_len = int(seq69.shape[0])
    half = max(1, t_len // 2)
    tail32_start = max(0, t_len - 32)
    tail16_start = max(0, t_len - 16)

    head16 = seq69[: min(16, t_len)]
    head32 = seq69[: min(32, t_len)]
    tail16 = seq69[tail16_start:]
    tail32 = seq69[tail32_start:]
    half_seq = seq69[half:]

    extras = [
        *_window_stats(half_seq, 0, half_seq.shape[0]),
        *_window_stats(tail32, 0, tail32.shape[0]),
        *_window_stats(tail16, 0, tail16.shape[0]),
        tail16.mean(axis=0) - head16.mean(axis=0),
        tail32.mean(axis=0) - head32.mean(axis=0),
    ]
    feat = np.concatenate([base, *extras], axis=0)
    return feat.astype(np.float32)


def _safe_pair_mean(xy: np.ndarray, left_idx: int, right_idx: int) -> np.ndarray:
    pts = xy[:, [left_idx, right_idx], :]
    valid = ~np.all(pts == 0.0, axis=-1)
    weights = valid.astype(np.float32)
    summed = (pts * weights[..., np.newaxis]).sum(axis=1)
    denom = np.clip(weights.sum(axis=1, keepdims=True), 1.0, None)
    return summed / denom


def _series_stats(series: np.ndarray) -> list[np.ndarray]:
    diffs = np.diff(series, axis=0) if series.shape[0] > 1 else np.zeros((1,), dtype=np.float32)
    return [
        series.mean(axis=0),
        series.std(axis=0),
        series.min(axis=0),
        series.max(axis=0),
        series[-1] - series[0],
        np.mean(np.abs(diffs), axis=0),
        np.std(diffs, axis=0),
    ]


def _build_extratrees_feature_vector_v3(seq69: np.ndarray) -> np.ndarray:
    base = _build_extratrees_feature_vector_v1(seq69)

    body = seq69[:, :68].reshape(seq69.shape[0], 17, 4).astype(np.float32, copy=False)
    xy = body[..., :2]
    joint_speed = body[..., 2]

    mid_shoulder = _safe_pair_mean(xy, 5, 6)
    mid_hip = _safe_pair_mean(xy, 11, 12)
    mid_ankle = _safe_pair_mean(xy, 15, 16)
    mid_knee = _safe_pair_mean(xy, 13, 14)

    torso_dx = mid_shoulder[:, 0] - mid_hip[:, 0]
    torso_dy = mid_shoulder[:, 1] - mid_hip[:, 1]
    torso_angle = np.arctan2(np.abs(torso_dx), np.abs(torso_dy) + 1e-6).astype(np.float32)

    hip_y = mid_hip[:, 1].astype(np.float32)
    shoulder_y = mid_shoulder[:, 1].astype(np.float32)
    ankle_y = mid_ankle[:, 1].astype(np.float32)
    knee_y = mid_knee[:, 1].astype(np.float32)

    torso_height = np.linalg.norm(mid_shoulder - mid_hip, axis=1).astype(np.float32)
    body_height = np.linalg.norm(mid_shoulder - mid_ankle, axis=1).astype(np.float32)
    lower_height = np.linalg.norm(mid_hip - mid_ankle, axis=1).astype(np.float32)

    shoulder_span = np.abs(xy[:, 5, 0] - xy[:, 6, 0]).astype(np.float32)
    ankle_span = np.abs(xy[:, 15, 0] - xy[:, 16, 0]).astype(np.float32)
    knee_span = np.abs(xy[:, 13, 0] - xy[:, 14, 0]).astype(np.float32)
    support_ratio = ankle_span / np.clip(shoulder_span, 1e-4, None)

    hip_vertical_clearance = (mid_ankle[:, 1] - mid_hip[:, 1]).astype(np.float32)
    nose_to_hip = (xy[:, 0, 1] - mid_hip[:, 1]).astype(np.float32)
    knee_to_hip = (knee_y - hip_y).astype(np.float32)

    mean_joint_speed = joint_speed.mean(axis=1).astype(np.float32)
    upper_speed = joint_speed[:, [0, 5, 6, 7, 8, 9, 10]].mean(axis=1).astype(np.float32)
    lower_speed = joint_speed[:, [11, 12, 13, 14, 15, 16]].mean(axis=1).astype(np.float32)
    speed_balance = (upper_speed - lower_speed).astype(np.float32)

    physics_series = [
        torso_angle,
        hip_y,
        shoulder_y,
        ankle_y,
        torso_height,
        body_height,
        lower_height,
        shoulder_span,
        knee_span,
        ankle_span,
        support_ratio,
        hip_vertical_clearance,
        nose_to_hip,
        knee_to_hip,
        mean_joint_speed,
        upper_speed,
        lower_speed,
        speed_balance,
        seq69[:, -1].astype(np.float32),
    ]

    extras = np.concatenate(
        [np.concatenate(_series_stats(series[:, np.newaxis]), axis=0) for series in physics_series],
        axis=0,
    )
    extras = np.nan_to_num(extras, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    feat = np.concatenate([base, extras], axis=0)
    return feat.astype(np.float32, copy=False)


def build_extratrees_feature_vector(
    seq69: np.ndarray,
    feature_spec: str = EXTRATREES_FEATURE_SPEC_V2,
) -> np.ndarray:
    spec = str(feature_spec or EXTRATREES_FEATURE_SPEC_V1)
    if spec == EXTRATREES_FEATURE_SPEC_V1:
        return _build_extratrees_feature_vector_v1(seq69)
    if spec == EXTRATREES_FEATURE_SPEC_V2:
        return _build_extratrees_feature_vector_v2(seq69)
    if spec == EXTRATREES_FEATURE_SPEC_V3_PHYSICS:
        return _build_extratrees_feature_vector_v3(seq69)
    raise ValueError(f"Unsupported ExtraTrees feature spec: {spec}")


def build_extratrees_feature_matrix(
    X: np.ndarray,
    feature_spec: str = EXTRATREES_FEATURE_SPEC_V2,
) -> np.ndarray:
    rows = [build_extratrees_feature_vector(seq69, feature_spec=feature_spec) for seq69 in X]
    return np.stack(rows, axis=0).astype(np.float32, copy=False)
