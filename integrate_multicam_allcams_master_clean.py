from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_prepare_v3 import compute_features, sliding_window
from prepare_master_clean_dataset import attach_quality_columns, compute_quality_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Integrate additional Multicam all-camera windows into master-clean 5-action dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_dir", default="data/train_ready_action_master_clean_v1_full")
    p.add_argument("--multicam_meta_csv", default="data/processed_pose_multicam_allcams_v1/metadata_pose_final.csv")
    p.add_argument("--multicam_pose_root", default="data/processed_pose_multicam_allcams_v1")
    p.add_argument("--out_dir", default="data/train_ready_action_master_clean_v2_multicam_allcams")
    p.add_argument("--cams_to_add", default="2,3,4,5,6,7,8")
    p.add_argument(
        "--include_labels",
        default="fall,walking",
        help="Comma-separated Multicam labels to include: fall,walking",
    )
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--window_stride", type=int, default=64)
    p.add_argument("--min_presence_ratio", type=float, default=0.08)
    p.add_argument("--evaluate_grouped_cv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_n_estimators", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_cam_set(raw: str) -> set[int]:
    token = str(raw).strip().lower()
    if token == "all":
        return {1, 2, 3, 4, 5, 6, 7, 8}
    out: set[int] = set()
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            cam = int(part)
        except ValueError:
            continue
        if 1 <= cam <= 8:
            out.add(cam)
    return out


def parse_label_set(raw: str) -> set[str]:
    allowed = {"fall", "walking"}
    out: set[str] = set()
    for part in str(raw).split(","):
        token = part.strip().lower()
        if token in allowed:
            out.add(token)
    return out


def load_label_map(base_dir: Path) -> dict[int, str]:
    with open(base_dir / "label_map.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "label_map" in raw:
        raw = raw["label_map"]
    return {int(k): str(v) for k, v in raw.items()}


def build_name_to_id(label_map: dict[int, str]) -> dict[str, int]:
    return {str(v): int(k) for k, v in label_map.items()}


def resolve_npy_path(row: pd.Series, pose_root: Path) -> Path:
    npy_path = Path(str(row["npy_path"]))
    if npy_path.is_absolute():
        return npy_path
    return (pose_root / npy_path.name).resolve()


def quality_presence_from_raw_pose(seq_raw: np.ndarray) -> float:
    valid_joint = np.any(np.abs(seq_raw) > 1e-6, axis=-1)
    return float(valid_joint.mean()) if seq_raw.size > 0 else 0.0


def grouped_cv_macro_f1(
    X: np.ndarray,
    y: np.ndarray,
    action_ids: np.ndarray,
    *,
    n_estimators: int,
    seed: int,
) -> float | None:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.metrics import f1_score
        from sklearn.model_selection import GroupKFold
        from src.action_model_common import EXTRATREES_FEATURE_SPEC_V1, build_extratrees_feature_matrix
    except Exception:
        return None

    def base_action_id(v: str) -> str:
        s = str(v)
        return s.rsplit("_aug", 1)[0] if "_aug" in s else s

    groups = np.array([base_action_id(v) for v in action_ids], dtype=object)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        return None

    Xf = build_extratrees_feature_matrix(X, feature_spec=EXTRATREES_FEATURE_SPEC_V1)
    cv = GroupKFold(n_splits=5)
    f1s: list[float] = []
    for tr, te in cv.split(Xf, y, groups=groups):
        clf = ExtraTreesClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(Xf[tr], y[tr])
        pred = clf.predict(Xf[te])
        f1s.append(float(f1_score(y[te], pred, average="macro")))
    return float(np.mean(f1s))


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    multicam_meta_csv = Path(args.multicam_meta_csv)
    multicam_pose_root = Path(args.multicam_pose_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_base = np.load(base_dir / "X_train.npy").astype(np.float32)
    y_base = np.load(base_dir / "y_train.npy").astype(np.int64)
    meta_base = pd.read_csv(base_dir / "metadata_train.csv").copy()
    if not (len(X_base) == len(y_base) == len(meta_base)):
        raise ValueError("Base dataset length mismatch.")

    label_map = load_label_map(base_dir)
    label_name_to_id = build_name_to_id(label_map)
    if "Fall" not in label_name_to_id or "Walking" not in label_name_to_id:
        raise ValueError("Base label map is missing Fall or Walking.")

    m_mc = pd.read_csv(multicam_meta_csv).copy()
    if "source" in m_mc.columns:
        m_mc = m_mc[m_mc["source"].astype(str) == "Multicam"].copy()
    if m_mc.empty:
        raise RuntimeError("No Multicam rows found in metadata CSV.")

    if "cam" in m_mc.columns:
        m_mc["cam"] = m_mc["cam"].astype(int)
    else:
        m_mc["cam"] = (
            m_mc["action_id"]
            .astype(str)
            .str.extract(r"_cam(\d+)_", expand=False)
            .fillna("1")
            .astype(int)
        )

    cams_to_add = parse_cam_set(args.cams_to_add)
    if not cams_to_add:
        raise ValueError("No valid cams_to_add provided.")

    m_mc = m_mc[m_mc["cam"].isin(sorted(cams_to_add))].copy()
    if m_mc.empty:
        raise RuntimeError("No Multicam rows remain after cam filtering.")

    include_labels = parse_label_set(args.include_labels)
    if not include_labels:
        raise ValueError("No valid include_labels provided (use fall and/or walking).")

    X_ext_rows: list[np.ndarray] = []
    y_ext_rows: list[int] = []
    meta_ext_rows: list[dict[str, Any]] = []

    for row in m_mc.itertuples(index=False):
        row_dict = row._asdict()
        npy_path = resolve_npy_path(pd.Series(row_dict), multicam_pose_root)
        if not npy_path.exists():
            continue

        seq_raw = np.load(npy_path).astype(np.float32)
        if seq_raw.ndim != 3 or seq_raw.shape[1:] != (17, 2) or seq_raw.shape[0] == 0:
            continue

        presence_ratio = quality_presence_from_raw_pose(seq_raw)
        if presence_ratio < float(args.min_presence_ratio):
            continue

        src_label_name = str(row_dict.get("label_name", ""))
        src_label_norm = src_label_name.strip().lower()
        if src_label_norm == "fall":
            if "fall" not in include_labels:
                continue
            label_id = int(label_name_to_id["Fall"])
        elif src_label_norm == "walking":
            if "walking" not in include_labels:
                continue
            label_id = int(label_name_to_id["Walking"])
        else:
            continue

        feat = compute_features(seq_raw)
        windows = sliding_window(feat, seq_len=int(args.seq_len), stride=int(args.window_stride))
        if not windows:
            continue

        action_id = str(row_dict["action_id"])
        cam = int(row_dict["cam"])
        for wi, win in enumerate(windows):
            X_ext_rows.append(win.astype(np.float32, copy=False))
            y_ext_rows.append(label_id)
            meta_ext_rows.append(
                {
                    "source": "Multicam_AllCams",
                    "action_id": action_id,
                    "aug_type": f"external_mc_cam{cam}_w{wi}",
                    "label_id": label_id,
                    "label_name": label_map[label_id],
                    "original_index": -1,
                    "repair_parent_index": -1,
                    "repair_tag": "external_multicam_allcams",
                    "external_category": f"cam{cam}",
                    "external_video_path": str(row_dict.get("video_path", "")),
                }
            )

    if not X_ext_rows:
        raise RuntimeError("No additional Multicam windows were generated.")

    X_ext = np.stack(X_ext_rows, axis=0).astype(np.float32, copy=False)
    y_ext = np.array(y_ext_rows, dtype=np.int64)
    meta_ext = pd.DataFrame(meta_ext_rows)

    X_out = np.concatenate([X_base, X_ext], axis=0).astype(np.float32, copy=False)
    y_out = np.concatenate([y_base, y_ext], axis=0)

    for col in meta_base.columns:
        if col not in meta_ext.columns:
            meta_ext[col] = np.nan
    for col in meta_ext.columns:
        if col not in meta_base.columns:
            meta_base[col] = np.nan
    meta_out = pd.concat([meta_base, meta_ext[meta_base.columns]], ignore_index=True)

    quality_metrics = compute_quality_metrics(X_out)
    meta_out = attach_quality_columns(meta_out, quality_metrics)

    np.save(out_dir / "X_train.npy", X_out)
    np.save(out_dir / "y_train.npy", y_out)
    meta_out.to_csv(out_dir / "metadata_train.csv", index=False)
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label_map": label_map}, f, indent=2)

    eval_summary = {"grouped_cv_macro_f1_est": None}
    if args.evaluate_grouped_cv:
        eval_summary["grouped_cv_macro_f1_est"] = grouped_cv_macro_f1(
            X_out,
            y_out,
            meta_out["action_id"].astype(str).to_numpy(),
            n_estimators=int(args.eval_n_estimators),
            seed=int(args.seed),
        )

    summary = {
        "base_dir": str(base_dir),
        "multicam_meta_csv": str(multicam_meta_csv),
        "multicam_pose_root": str(multicam_pose_root),
        "out_dir": str(out_dir),
        "cams_to_add": sorted(cams_to_add),
        "include_labels": sorted(include_labels),
        "seq_len": int(args.seq_len),
        "window_stride": int(args.window_stride),
        "min_presence_ratio": float(args.min_presence_ratio),
        "added_windows": int(len(X_ext)),
        "added_class_distribution": {
            label_map[int(k)]: int(v) for k, v in zip(*np.unique(y_ext, return_counts=True))
        },
        "final_class_distribution": {
            label_map[int(k)]: int(v) for k, v in zip(*np.unique(y_out, return_counts=True))
        },
        "final_source_distribution": {
            str(k): int(v) for k, v in meta_out["source"].value_counts().items()
        },
        "evaluation": eval_summary,
    }

    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("MULTICAM ALL-CAMS INTEGRATION COMPLETE")
    print("=" * 72)
    print(f"Base dataset: {base_dir}")
    print(f"Added windows: {len(X_ext)}")
    print(f"Output dataset: {out_dir}")
    print(f"Added class distribution: {summary['added_class_distribution']}")
    print(f"Final class distribution: {summary['final_class_distribution']}")
    if args.evaluate_grouped_cv:
        print(f"Grouped CV macro-F1 estimate: {eval_summary['grouped_cv_macro_f1_est']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
