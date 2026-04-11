from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.action_model_common import DEFAULT_ACTION_LABEL_MAP


SIDE_OCCLUSION_KPTS = {
    "left": [1, 3, 5, 7, 9, 11, 13, 15],
    "right": [2, 4, 6, 8, 10, 12, 14, 16],
}
LOWER_BODY_KPTS = [11, 12, 13, 14, 15, 16]
UPPER_BODY_KPTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare a conservative round-2 hardcase action dataset for runtime/model follow-up",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_dir", default="data/train_ready_action_repair_v2_unicomfacauca")
    p.add_argument("--out_dir", default="data/train_ready_action_repair_v3_hardcases")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--walk_partial_body_copies", type=int, default=180)
    p.add_argument("--bending_boundary_copies", type=int, default=160)
    p.add_argument("--walk_seed_tags", default="orig,external_unicomfacauca_ADL-WALK")
    p.add_argument("--bending_seed_tags", default="orig")
    p.add_argument("--walk_min_mean_velocity", type=float, default=0.018)
    p.add_argument("--bending_min_aspect_ratio", type=float, default=0.45)
    p.add_argument("--bending_widen_factor", type=float, default=1.18)
    p.add_argument("--bending_upper_lift", type=float, default=1.08)
    return p.parse_args()


def load_label_map(base_dir: Path) -> dict[int, str]:
    label_map_path = base_dir / "label_map.json"
    if not label_map_path.exists():
        return DEFAULT_ACTION_LABEL_MAP
    with open(label_map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "label_map" in raw:
        raw = raw["label_map"]
    return {int(k): str(v) for k, v in raw.items()}


def unpack_xy(seq69: np.ndarray) -> np.ndarray:
    return np.array(seq69[:, :68].reshape(seq69.shape[0], 17, 4)[..., :2], copy=True, dtype=np.float32)


def recompute_aspect_ratio_from_xy(xy: np.ndarray) -> np.ndarray:
    aspect = np.ones((xy.shape[0], 1), dtype=np.float32)
    for t in range(xy.shape[0]):
        frame_xy = xy[t]
        valid = ~np.all(frame_xy == 0.0, axis=1)
        if int(valid.sum()) < 2:
            continue
        valid_xy = frame_xy[valid]
        width = float(valid_xy[:, 0].max() - valid_xy[:, 0].min())
        height = float(valid_xy[:, 1].max() - valid_xy[:, 1].min())
        aspect[t, 0] = width / max(height, 1e-4)
    return aspect


def repack_from_xy(xy: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(xy, dtype=np.float32)
    delta[1:] = xy[1:] - xy[:-1]
    vel = np.linalg.norm(delta, axis=-1)
    acc = np.zeros_like(vel, dtype=np.float32)
    acc[1:] = np.abs(vel[1:] - vel[:-1])
    packed = np.concatenate(
        [
            xy,
            vel[..., np.newaxis],
            acc[..., np.newaxis],
        ],
        axis=-1,
    ).reshape(xy.shape[0], 68)
    return np.concatenate([packed, recompute_aspect_ratio_from_xy(xy)], axis=-1).astype(np.float32, copy=False)


def mean_velocity(seq69: np.ndarray) -> float:
    xy = unpack_xy(seq69)
    vel = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    return float(vel.mean())


def mean_aspect_ratio(seq69: np.ndarray) -> float:
    return float(np.mean(seq69[:, -1]))


def apply_side_occlusion(seq69: np.ndarray, side: str) -> np.ndarray:
    xy = unpack_xy(seq69)
    xy[:, SIDE_OCCLUSION_KPTS[side], :] = 0.0
    return repack_from_xy(xy)


def apply_bending_boundary_variant(
    seq69: np.ndarray,
    *,
    widen_factor: float,
    upper_lift: float,
) -> np.ndarray:
    xy = unpack_xy(seq69)
    hip_center_x = xy[:, [11, 12], 0].mean(axis=1, keepdims=True)
    xy[:, LOWER_BODY_KPTS, 0] = hip_center_x + (xy[:, LOWER_BODY_KPTS, 0] - hip_center_x) * widen_factor
    xy[:, UPPER_BODY_KPTS, 1] = xy[:, UPPER_BODY_KPTS, 1] * upper_lift
    return repack_from_xy(xy)


def select_indices(
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    label_name: str,
    allowed_tags: set[str],
    score_fn,
    max_copies: int,
    min_score: float,
) -> list[int]:
    mask = meta["label_name"].eq(label_name)
    if "repair_tag" in meta.columns:
        mask &= meta["repair_tag"].isin(allowed_tags)
    indices = meta.index[mask].to_list()
    scored: list[tuple[float, int]] = []
    for idx in indices:
        score = float(score_fn(X[int(idx)]))
        if score >= min_score:
            scored.append((score, int(idx)))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[:max_copies]]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_base = np.load(base_dir / "X_train.npy").astype(np.float32)
    y_base = np.load(base_dir / "y_train.npy").astype(np.int64)
    meta_base = pd.read_csv(base_dir / "metadata_train.csv").copy()
    label_map = load_label_map(base_dir)

    if not (len(X_base) == len(y_base) == len(meta_base)):
        raise ValueError("Base dataset arrays and metadata length mismatch.")

    walk_tags = {part.strip() for part in str(args.walk_seed_tags).split(",") if part.strip()}
    bending_tags = {part.strip() for part in str(args.bending_seed_tags).split(",") if part.strip()}

    walk_indices = select_indices(
        X_base,
        meta_base,
        label_name="Walking",
        allowed_tags=walk_tags,
        score_fn=mean_velocity,
        max_copies=args.walk_partial_body_copies,
        min_score=args.walk_min_mean_velocity,
    )
    bending_indices = select_indices(
        X_base,
        meta_base,
        label_name="Bending",
        allowed_tags=bending_tags,
        score_fn=mean_aspect_ratio,
        max_copies=args.bending_boundary_copies,
        min_score=args.bending_min_aspect_ratio,
    )

    rng.shuffle(walk_indices)
    rng.shuffle(bending_indices)

    X_aug_list: list[np.ndarray] = []
    y_aug_list: list[int] = []
    meta_aug_rows: list[dict] = []

    for offset, idx in enumerate(walk_indices):
        side = "left" if offset % 2 == 0 else "right"
        seq_aug = apply_side_occlusion(X_base[int(idx)], side)
        row = meta_base.loc[int(idx)].to_dict()
        row["action_id"] = f"{row['action_id']}_aug_hard_walk_occ_{side}"
        row["aug_type"] = f"hard_walk_occ_{side}"
        row["repair_parent_index"] = int(idx)
        row["repair_tag"] = f"hard_walk_partial_body_{side}"
        row["label_id"] = int(y_base[int(idx)])
        row["label_name"] = label_map[int(y_base[int(idx)])]
        X_aug_list.append(seq_aug)
        y_aug_list.append(int(y_base[int(idx)]))
        meta_aug_rows.append(row)

    for idx in bending_indices:
        seq_aug = apply_bending_boundary_variant(
            X_base[int(idx)],
            widen_factor=float(args.bending_widen_factor),
            upper_lift=float(args.bending_upper_lift),
        )
        row = meta_base.loc[int(idx)].to_dict()
        row["action_id"] = f"{row['action_id']}_aug_hard_bending_boundary"
        row["aug_type"] = "hard_bending_boundary"
        row["repair_parent_index"] = int(idx)
        row["repair_tag"] = "hard_bending_boundary"
        row["label_id"] = int(y_base[int(idx)])
        row["label_name"] = label_map[int(y_base[int(idx)])]
        X_aug_list.append(seq_aug)
        y_aug_list.append(int(y_base[int(idx)]))
        meta_aug_rows.append(row)

    if X_aug_list:
        X_out = np.concatenate([X_base, np.stack(X_aug_list, axis=0)], axis=0).astype(np.float32, copy=False)
        y_out = np.concatenate([y_base, np.array(y_aug_list, dtype=np.int64)], axis=0)
        meta_out = pd.concat([meta_base, pd.DataFrame(meta_aug_rows)], ignore_index=True)
    else:
        X_out = X_base
        y_out = y_base
        meta_out = meta_base

    np.save(out_dir / "X_train.npy", X_out)
    np.save(out_dir / "y_train.npy", y_out)
    meta_out.to_csv(out_dir / "metadata_train.csv", index=False)

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label_map": label_map}, f, indent=2)

    summary = {
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "walk_partial_body_count": int(len(walk_indices)),
        "bending_boundary_count": int(len(bending_indices)),
        "walk_seed_tags": sorted(walk_tags),
        "bending_seed_tags": sorted(bending_tags),
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

    print("=" * 72)
    print("ACTION HARDCASE ROUND 2 PREP COMPLETE")
    print("=" * 72)
    print(f"Base dataset: {base_dir}")
    print(f"Output dataset: {out_dir}")
    print(f"Walking partial-body augmentations: {len(walk_indices)}")
    print(f"Bending boundary augmentations: {len(bending_indices)}")
    print("Class distribution:")
    for label_id, count in zip(*np.unique(y_out, return_counts=True)):
        print(f"  {label_map[int(label_id)]:18s}: {int(count)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
