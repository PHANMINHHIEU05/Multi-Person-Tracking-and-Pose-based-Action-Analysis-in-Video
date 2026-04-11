from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.action_model_common import (
    DEFAULT_ACTION_LABEL_MAP,
    FIVE_ACTION_FOCUS_LABEL_MAP,
    TAXONOMY_REPAIR_ROUND3_LABEL_MAP,
)


OLD_TO_NEW_LABEL_ID = {
    0: 0,  # Fall
    1: 2,  # Walking
    2: 3,  # Sitting_Quickly
    3: 4,  # Bending
    4: 5,  # Lying_Down
}

ROUND2_TO_ROUND3_LABEL_ID = {
    0: 0,  # Fall
    1: 1,  # Standing
    2: 2,  # Walking
    3: 4,  # Sitting_Quickly
    4: 5,  # Bending
    5: 6,  # Lying_Down
}

ROUND2_TO_FIVE_ACTION_LABEL_ID = {
    0: 0,  # Fall
    1: 1,  # Standing
    2: 2,  # Walking
    3: 3,  # Sitting
    5: 4,  # Lying_Down
}

LOWER_BODY_KPTS = {
    "ankles": [15, 16],
    "knees_ankles": [13, 14, 15, 16],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repair action datasets for runtime model upgrades",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["repair_v1", "taxonomy_round3", "five_action_round4"], default="repair_v1")
    p.add_argument("--base_dir", default="data/train_ready_unified_clean_v2_fallboost")
    p.add_argument("--out_dir", default="data/train_ready_action_repair_v1")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--standing_sources", default="Multicam")
    p.add_argument("--standing_seed_aug_types", default="orig,flip,orig_mc_balance,flip_mc_balance")
    p.add_argument("--standing_max_mean_vel", type=float, default=0.0040)
    p.add_argument("--standing_max_hip_disp", type=float, default=0.0015)
    p.add_argument("--standing_max_aspect", type=float, default=0.16)
    p.add_argument("--standing_min_candidate_rows_per_action", type=int, default=4)

    p.add_argument("--sit_occlusion_aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sit_occlusion_mode", choices=sorted(LOWER_BODY_KPTS), default="ankles")
    p.add_argument("--sit_occlusion_seed_aug_types", default="orig,flip,orig_mc_balance,flip_mc_balance")
    p.add_argument("--sit_occlusion_max_copies", type=int, default=162)

    p.add_argument("--sit_external_dir", default="data/train_ready_action_repair_v4_sit_only")
    p.add_argument("--include_external_sitting", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sitting_static_seed_tags", default="orig")
    p.add_argument("--sitting_static_max_mean_vel", type=float, default=0.0070)
    p.add_argument("--sitting_static_max_hip_disp", type=float, default=0.0002)
    p.add_argument("--sitting_static_max_aspect", type=float, default=0.25)
    p.add_argument("--sitting_static_min_candidate_rows_per_action", type=int, default=4)
    p.add_argument("--sitting_occlusion_aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sitting_occlusion_mode", choices=sorted(LOWER_BODY_KPTS), default="ankles")
    p.add_argument("--sitting_occlusion_max_copies", type=int, default=128)

    return p.parse_args()


def compute_motion_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = X[:, :, :68].reshape(-1, X.shape[1], 17, 4)[..., :2]
    vel = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
    mean_vel = vel.mean(axis=(1, 2))
    hips = xy[:, :, [11, 12], :].mean(axis=2)
    hip_disp = np.linalg.norm(hips[:, -1] - hips[:, 0], axis=1)
    mean_aspect = X[:, :, -1].mean(axis=1)
    return mean_vel.astype(np.float32), hip_disp.astype(np.float32), mean_aspect.astype(np.float32)


def choose_standing_action_ids(
    X: np.ndarray,
    meta: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[set[str], dict[str, int]]:
    source_set = {part.strip() for part in str(args.standing_sources).split(",") if part.strip()}
    seed_aug_types = {part.strip() for part in str(args.standing_seed_aug_types).split(",") if part.strip()}

    walking_mask = meta["label_name"].eq("Walking").to_numpy()
    source_mask = meta["source"].isin(source_set).to_numpy()
    aug_mask = meta["aug_type"].isin(seed_aug_types).to_numpy()
    seed_mask = walking_mask & source_mask & aug_mask

    X_seed = X[seed_mask]
    meta_seed = meta.loc[seed_mask].reset_index(drop=True)
    mean_vel, hip_disp, mean_aspect = compute_motion_stats(X_seed)

    candidate_mask = (
        (mean_vel <= args.standing_max_mean_vel)
        & (hip_disp <= args.standing_max_hip_disp)
        & (mean_aspect <= args.standing_max_aspect)
    )
    candidate_meta = meta_seed.loc[candidate_mask].copy()
    counts = candidate_meta["action_id"].value_counts()
    selected_counts = counts[counts >= args.standing_min_candidate_rows_per_action]
    selected_ids = set(selected_counts.index.astype(str))
    return selected_ids, {str(k): int(v) for k, v in selected_counts.items()}


def choose_sitting_static_indices(
    X: np.ndarray,
    meta: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, int]]:
    allowed_tags = {part.strip() for part in str(args.sitting_static_seed_tags).split(",") if part.strip()}
    sitting_mask = meta["label_name"].eq("Sitting_Quickly").to_numpy()
    if "repair_tag" in meta.columns and allowed_tags:
        sitting_mask &= meta["repair_tag"].isin(allowed_tags).to_numpy()

    seed_indices = np.flatnonzero(sitting_mask)
    if seed_indices.size == 0:
        return [], {}

    mean_vel, hip_disp, mean_aspect = compute_motion_stats(X[seed_indices])
    candidate_mask = (
        (mean_vel <= args.sitting_static_max_mean_vel)
        & (hip_disp <= args.sitting_static_max_hip_disp)
        & (mean_aspect <= args.sitting_static_max_aspect)
    )
    candidate_indices = seed_indices[candidate_mask]
    if candidate_indices.size == 0:
        return [], {}

    candidate_meta = meta.loc[candidate_indices].copy()
    counts = candidate_meta["action_id"].astype(str).value_counts()
    selected_counts = counts[counts >= args.sitting_static_min_candidate_rows_per_action]
    keep_ids = set(selected_counts.index.astype(str))
    final_indices = [int(idx) for idx in candidate_indices if str(meta.loc[int(idx), "action_id"]) in keep_ids]
    return final_indices, {str(k): int(v) for k, v in selected_counts.items()}


def ensure_metadata_columns(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    if "original_index" not in meta.columns:
        meta["original_index"] = np.arange(len(meta), dtype=np.int64)
    if "repair_parent_index" not in meta.columns:
        meta["repair_parent_index"] = meta["original_index"]
    if "repair_tag" not in meta.columns:
        meta["repair_tag"] = "orig"
    return meta


def recompute_aspect_ratio(seq69: np.ndarray) -> np.ndarray:
    xy = seq69[:, :68].reshape(seq69.shape[0], 17, 4)[..., :2]
    ar = np.ones((seq69.shape[0],), dtype=np.float32)
    for t in range(seq69.shape[0]):
        frame_xy = xy[t]
        valid = ~np.all(frame_xy == 0, axis=1)
        if int(valid.sum()) < 2:
            continue
        valid_xy = frame_xy[valid]
        width = float(valid_xy[:, 0].max() - valid_xy[:, 0].min())
        height = float(valid_xy[:, 1].max() - valid_xy[:, 1].min())
        ar[t] = width / max(height, 1e-4)
    return ar


def apply_keypoint_occlusion(seq69: np.ndarray, keypoints: list[int]) -> np.ndarray:
    repaired = np.array(seq69, copy=True)
    body = repaired[:, :68].reshape(repaired.shape[0], 17, 4)
    body[:, keypoints, :] = 0.0
    repaired[:, :68] = body.reshape(repaired.shape[0], 68)
    repaired[:, -1] = recompute_aspect_ratio(repaired)
    return repaired.astype(np.float32, copy=False)


def load_external_sitting_rows(
    sit_external_dir: Path,
    label_map: dict[int, str],
) -> tuple[list[np.ndarray], list[int], list[dict]]:
    if not sit_external_dir.exists():
        return [], [], []

    X_ext = np.load(sit_external_dir / "X_train.npy").astype(np.float32)
    meta_ext = pd.read_csv(sit_external_dir / "metadata_train.csv").copy()
    if len(X_ext) != len(meta_ext):
        raise ValueError("External sitting dataset arrays and metadata length mismatch.")

    if "repair_tag" not in meta_ext.columns:
        return [], [], []

    sit_mask = meta_ext["repair_tag"].eq("external_unicomfacauca_ADL-SIT").to_numpy()
    indices = np.flatnonzero(sit_mask)
    if indices.size == 0:
        return [], [], []

    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    rows_meta: list[dict] = []
    for idx in indices:
        row = meta_ext.loc[int(idx)].to_dict()
        row["label_id"] = 3
        row["label_name"] = label_map[3]
        row["repair_tag"] = "external_unicomfacauca_ADL-SIT_sitting"
        rows_x.append(X_ext[int(idx)].astype(np.float32, copy=True))
        rows_y.append(3)
        rows_meta.append(row)
    return rows_x, rows_y, rows_meta


def build_external_data_manifest(base_meta: pd.DataFrame, *, mode: str) -> dict:
    ntu_count = int(base_meta["source"].eq("NTU_pseudo").sum()) if "source" in base_meta.columns else 0
    manifest = {
        "status": "prepared-but-not-ingested",
        "mode": mode,
        "preferred_sources": [
            {
                "name": "NTU RGB+D 60/120",
                "why": "Best match for standing, sit/stand transitions, and viewpoint diversity if local skeleton exports are available.",
                "local_ready": False,
            },
            {
                "name": "Additional chair-sit real video clips",
                "why": "Best direct supervision for lower-body-occluded sitting cases.",
                "local_ready": False,
            },
        ],
        "current_dataset_external_coverage": {
            "NTU_pseudo_windows_already_in_base_dataset": ntu_count,
        },
    }
    if mode == "taxonomy_round3":
        manifest["focus"] = [
            "stable Sitting taxonomy",
            "chair-sit domain support",
            "bend-vs-lie hard cases",
        ]
    if mode == "five_action_round4":
        manifest["focus"] = [
            "remove Bending from target label space",
            "unify all sitting variants into Sitting",
            "maximize accuracy for Fall / Standing / Walking / Sitting / Lying_Down",
        ]
    return manifest


def build_repair_v1_dataset(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_base = np.load(base_dir / "X_train.npy").astype(np.float32)
    y_base = np.load(base_dir / "y_train.npy").astype(np.int64)
    meta_base = pd.read_csv(base_dir / "metadata_train.csv").copy()

    if not (len(X_base) == len(y_base) == len(meta_base)):
        raise ValueError("Base dataset arrays and metadata length mismatch.")

    y_new = np.array([OLD_TO_NEW_LABEL_ID[int(label)] for label in y_base], dtype=np.int64)
    meta_new = ensure_metadata_columns(meta_base)
    meta_new["repair_tag"] = "orig"
    meta_new["label_id"] = y_new
    meta_new["label_name"] = meta_new["label_id"].map(DEFAULT_ACTION_LABEL_MAP)

    standing_action_ids, standing_seed_counts = choose_standing_action_ids(X_base, meta_base, args)
    standing_mask = meta_new["label_name"].eq("Walking") & meta_new["action_id"].isin(standing_action_ids)
    y_new[standing_mask.to_numpy()] = 1
    meta_new.loc[standing_mask, "label_id"] = 1
    meta_new.loc[standing_mask, "label_name"] = DEFAULT_ACTION_LABEL_MAP[1]
    meta_new.loc[standing_mask, "repair_tag"] = "relabel_standing_mined"

    X_aug_list: list[np.ndarray] = []
    y_aug_list: list[int] = []
    meta_aug_rows: list[dict] = []
    sit_aug_count = 0
    sit_seed_indices: list[int] = []

    if args.sit_occlusion_aug:
        seed_aug_types = {part.strip() for part in str(args.sit_occlusion_seed_aug_types).split(",") if part.strip()}
        sit_seed_mask = (
            meta_new["label_name"].eq("Sitting_Quickly")
            & meta_new["aug_type"].isin(seed_aug_types)
        )
        sit_seed_indices = meta_new.index[sit_seed_mask].to_list()
        rng.shuffle(sit_seed_indices)
        sit_seed_indices = sit_seed_indices[: args.sit_occlusion_max_copies]

        keypoints = LOWER_BODY_KPTS[args.sit_occlusion_mode]
        for idx in sit_seed_indices:
            seq_aug = apply_keypoint_occlusion(X_base[int(idx)], keypoints)
            row = meta_new.loc[int(idx)].to_dict()
            row["action_id"] = f"{row['action_id']}_aug_repair_sit_occl"
            row["aug_type"] = f"repair_occl_{args.sit_occlusion_mode}"
            row["repair_tag"] = f"aug_sit_occluded_{args.sit_occlusion_mode}"
            row["repair_parent_index"] = int(idx)
            X_aug_list.append(seq_aug)
            y_aug_list.append(int(row["label_id"]))
            meta_aug_rows.append(row)
        sit_aug_count = len(meta_aug_rows)

    if X_aug_list:
        X_out = np.concatenate([X_base, np.stack(X_aug_list, axis=0)], axis=0).astype(np.float32, copy=False)
        y_out = np.concatenate([y_new, np.array(y_aug_list, dtype=np.int64)], axis=0)
        meta_out = pd.concat([meta_new, pd.DataFrame(meta_aug_rows)], ignore_index=True)
    else:
        X_out = X_base
        y_out = y_new
        meta_out = meta_new

    np.save(out_dir / "X_train.npy", X_out)
    np.save(out_dir / "y_train.npy", y_out)
    meta_out.to_csv(out_dir / "metadata_train.csv", index=False)

    label_map_payload = {"label_map": DEFAULT_ACTION_LABEL_MAP}
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_payload, f, indent=2)

    summary = {
        "mode": args.mode,
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "label_map": DEFAULT_ACTION_LABEL_MAP,
        "standing_action_ids": sorted(standing_action_ids),
        "standing_seed_candidate_counts": standing_seed_counts,
        "standing_relabel_count": int(standing_mask.sum()),
        "sitting_occlusion_aug_count": int(sit_aug_count),
        "sitting_occlusion_seed_count": int(len(sit_seed_indices)),
        "class_distribution": {
            DEFAULT_ACTION_LABEL_MAP[int(label_id)]: int(count)
            for label_id, count in zip(*np.unique(y_out, return_counts=True))
        },
        "repair_tag_distribution": {
            str(tag): int(count) for tag, count in meta_out["repair_tag"].value_counts().items()
        },
        "source_distribution": {
            str(source): int(count) for source, count in meta_out["source"].value_counts().items()
        },
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "external_data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(build_external_data_manifest(meta_base, mode=args.mode), f, indent=2)

    return summary


def build_taxonomy_round3_dataset(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    sit_external_dir = Path(args.sit_external_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_base = np.load(base_dir / "X_train.npy").astype(np.float32)
    y_base = np.load(base_dir / "y_train.npy").astype(np.int64)
    meta_base = ensure_metadata_columns(pd.read_csv(base_dir / "metadata_train.csv").copy())

    if not (len(X_base) == len(y_base) == len(meta_base)):
        raise ValueError("Base dataset arrays and metadata length mismatch.")

    y_round3 = np.array([ROUND2_TO_ROUND3_LABEL_ID[int(label)] for label in y_base], dtype=np.int64)
    label_map = TAXONOMY_REPAIR_ROUND3_LABEL_MAP

    meta_round3 = meta_base.copy()
    meta_round3["label_id"] = y_round3
    meta_round3["label_name"] = meta_round3["label_id"].map(label_map)

    sitting_indices, sitting_seed_counts = choose_sitting_static_indices(X_base, meta_round3, args)
    if sitting_indices:
        y_round3[np.array(sitting_indices, dtype=np.int64)] = 3
        meta_round3.loc[sitting_indices, "label_id"] = 3
        meta_round3.loc[sitting_indices, "label_name"] = label_map[3]
        meta_round3.loc[sitting_indices, "repair_tag"] = "relabel_sitting_static_mined"

    X_extra_rows: list[np.ndarray] = []
    y_extra_rows: list[int] = []
    meta_extra_rows: list[dict] = []
    if args.include_external_sitting:
        x_ext, y_ext, meta_ext = load_external_sitting_rows(sit_external_dir, label_map)
        X_extra_rows.extend(x_ext)
        y_extra_rows.extend(y_ext)
        meta_extra_rows.extend(meta_ext)

    if X_extra_rows:
        X_combined = np.concatenate([X_base, np.stack(X_extra_rows, axis=0)], axis=0).astype(np.float32, copy=False)
        y_combined = np.concatenate([y_round3, np.array(y_extra_rows, dtype=np.int64)], axis=0)
        meta_combined = pd.concat([meta_round3, pd.DataFrame(meta_extra_rows)], ignore_index=True)
    else:
        X_combined = X_base
        y_combined = y_round3
        meta_combined = meta_round3

    sitting_aug_count = 0
    sitting_aug_indices: list[int] = []
    X_aug_list: list[np.ndarray] = []
    y_aug_list: list[int] = []
    meta_aug_rows: list[dict] = []

    if args.sitting_occlusion_aug:
        sitting_seed_mask = meta_combined["repair_tag"].isin(
            ["relabel_sitting_static_mined", "external_unicomfacauca_ADL-SIT_sitting"]
        )
        sitting_aug_indices = meta_combined.index[sitting_seed_mask].to_list()
        rng.shuffle(sitting_aug_indices)
        sitting_aug_indices = sitting_aug_indices[: args.sitting_occlusion_max_copies]
        keypoints = LOWER_BODY_KPTS[args.sitting_occlusion_mode]
        for idx in sitting_aug_indices:
            seq_aug = apply_keypoint_occlusion(X_combined[int(idx)], keypoints)
            row = meta_combined.loc[int(idx)].to_dict()
            row["action_id"] = f"{row['action_id']}_aug_tax3_sit_occl"
            row["aug_type"] = f"tax3_occl_{args.sitting_occlusion_mode}"
            row["repair_tag"] = f"aug_sitting_occluded_{args.sitting_occlusion_mode}"
            row["repair_parent_index"] = int(meta_combined.loc[int(idx), "repair_parent_index"])
            X_aug_list.append(seq_aug)
            y_aug_list.append(3)
            row["label_id"] = 3
            row["label_name"] = label_map[3]
            meta_aug_rows.append(row)
        sitting_aug_count = len(meta_aug_rows)

    if X_aug_list:
        X_out = np.concatenate([X_combined, np.stack(X_aug_list, axis=0)], axis=0).astype(np.float32, copy=False)
        y_out = np.concatenate([y_combined, np.array(y_aug_list, dtype=np.int64)], axis=0)
        meta_out = pd.concat([meta_combined, pd.DataFrame(meta_aug_rows)], ignore_index=True)
    else:
        X_out = X_combined
        y_out = y_combined
        meta_out = meta_combined

    np.save(out_dir / "X_train.npy", X_out)
    np.save(out_dir / "y_train.npy", y_out)
    meta_out.to_csv(out_dir / "metadata_train.csv", index=False)

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label_map": label_map}, f, indent=2)

    summary = {
        "mode": args.mode,
        "base_dir": str(base_dir),
        "sit_external_dir": str(sit_external_dir),
        "out_dir": str(out_dir),
        "label_map": label_map,
        "sitting_static_action_counts": sitting_seed_counts,
        "sitting_static_relabel_count": int(len(sitting_indices)),
        "external_sitting_count": int(len(y_extra_rows)),
        "sitting_occlusion_aug_count": int(sitting_aug_count),
        "sitting_occlusion_seed_count": int(len(sitting_aug_indices)),
        "class_distribution": {
            label_map[int(label_id)]: int(count)
            for label_id, count in zip(*np.unique(y_out, return_counts=True))
        },
        "repair_tag_distribution": {
            str(tag): int(count) for tag, count in meta_out["repair_tag"].value_counts().items()
        },
        "source_distribution": {
            str(source): int(count) for source, count in meta_out["source"].value_counts().items()
        },
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "external_data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(build_external_data_manifest(meta_base, mode=args.mode), f, indent=2)

    return summary


def build_five_action_round4_dataset(args: argparse.Namespace) -> dict[str, object]:
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    sit_external_dir = Path(args.sit_external_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_base = np.load(base_dir / "X_train.npy").astype(np.float32)
    y_base = np.load(base_dir / "y_train.npy").astype(np.int64)
    meta_base = ensure_metadata_columns(pd.read_csv(base_dir / "metadata_train.csv").copy())

    if not (len(X_base) == len(y_base) == len(meta_base)):
        raise ValueError("Base dataset arrays and metadata length mismatch.")

    keep_mask = np.array([int(label) in ROUND2_TO_FIVE_ACTION_LABEL_ID for label in y_base], dtype=bool)
    dropped_bending_count = int((~keep_mask).sum())

    X_keep = X_base[keep_mask].astype(np.float32, copy=False)
    y_keep = np.array([ROUND2_TO_FIVE_ACTION_LABEL_ID[int(label)] for label in y_base[keep_mask]], dtype=np.int64)
    meta_keep = meta_base.loc[keep_mask].reset_index(drop=True)
    label_map = FIVE_ACTION_FOCUS_LABEL_MAP

    meta_keep["label_id"] = y_keep
    meta_keep["label_name"] = meta_keep["label_id"].map(label_map)
    meta_keep["repair_tag"] = meta_keep["repair_tag"].replace(
        {
            "aug_sit_occluded_ankles": "aug_sitting_occluded_ankles",
            "aug_sit_occluded_knees_ankles": "aug_sitting_occluded_knees_ankles",
        }
    )

    X_extra_rows: list[np.ndarray] = []
    y_extra_rows: list[int] = []
    meta_extra_rows: list[dict] = []
    if args.include_external_sitting:
        x_ext, _, meta_ext = load_external_sitting_rows(sit_external_dir, label_map)
        X_extra_rows.extend(x_ext)
        y_extra_rows.extend([3] * len(x_ext))
        meta_extra_rows.extend(meta_ext)

    if X_extra_rows:
        X_out = np.concatenate([X_keep, np.stack(X_extra_rows, axis=0)], axis=0).astype(np.float32, copy=False)
        y_out = np.concatenate([y_keep, np.array(y_extra_rows, dtype=np.int64)], axis=0)
        meta_out = pd.concat([meta_keep, pd.DataFrame(meta_extra_rows)], ignore_index=True)
    else:
        X_out = X_keep
        y_out = y_keep
        meta_out = meta_keep

    np.save(out_dir / "X_train.npy", X_out)
    np.save(out_dir / "y_train.npy", y_out)
    meta_out.to_csv(out_dir / "metadata_train.csv", index=False)

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label_map": label_map}, f, indent=2)

    summary = {
        "mode": args.mode,
        "base_dir": str(base_dir),
        "sit_external_dir": str(sit_external_dir),
        "out_dir": str(out_dir),
        "label_map": label_map,
        "dropped_bending_count": dropped_bending_count,
        "external_sitting_count": int(len(y_extra_rows)),
        "class_distribution": {
            label_map[int(label_id)]: int(count)
            for label_id, count in zip(*np.unique(y_out, return_counts=True))
        },
        "repair_tag_distribution": {
            str(tag): int(count) for tag, count in meta_out["repair_tag"].value_counts().items()
        },
        "source_distribution": {
            str(source): int(count) for source, count in meta_out["source"].value_counts().items()
        },
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "external_data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(build_external_data_manifest(meta_base, mode=args.mode), f, indent=2)

    return summary


def main() -> None:
    args = parse_args()

    if args.mode == "taxonomy_round3":
        summary = build_taxonomy_round3_dataset(args)
        label_map = TAXONOMY_REPAIR_ROUND3_LABEL_MAP
        title = "ACTION DATASET TAXONOMY ROUND 3 COMPLETE"
    elif args.mode == "five_action_round4":
        summary = build_five_action_round4_dataset(args)
        label_map = FIVE_ACTION_FOCUS_LABEL_MAP
        title = "ACTION DATASET FIVE-ACTION ROUND 4 COMPLETE"
    else:
        summary = build_repair_v1_dataset(args)
        label_map = DEFAULT_ACTION_LABEL_MAP
        title = "ACTION DATASET REPAIR COMPLETE"

    out_dir = Path(args.out_dir)
    y_out = np.load(out_dir / "y_train.npy").astype(np.int64)

    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"Base dataset: {summary['base_dir']}")
    print(f"Output dataset: {summary['out_dir']}")
    if args.mode == "taxonomy_round3":
        print(f"Sitting relabeled: {summary['sitting_static_relabel_count']}")
        print(f"External Sitting added: {summary['external_sitting_count']}")
        print(f"Sitting occlusion augmentations: {summary['sitting_occlusion_aug_count']}")
    elif args.mode == "five_action_round4":
        print(f"Dropped Bending rows: {summary['dropped_bending_count']}")
        print(f"External Sitting added: {summary['external_sitting_count']}")
    else:
        print(f"Standing relabeled: {summary['standing_relabel_count']}")
        print(f"Sitting occlusion augmentations: {summary['sitting_occlusion_aug_count']}")
    print("Class distribution:")
    for label_id, count in zip(*np.unique(y_out, return_counts=True)):
        print(f"  {label_map[int(label_id)]:18s}: {int(count)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
