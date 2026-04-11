from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare full + strict clean datasets with quality diagnostics for 5-action training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_dir", default="data/train_ready_action_repair_v6_five_action_round4_hardcases")
    p.add_argument("--out_full_dir", default="data/train_ready_action_master_clean_v1_full")
    p.add_argument("--out_strict_dir", default="data/train_ready_action_master_clean_v1_strict")
    p.add_argument("--drop_unicom_standing_presence_lt", type=float, default=0.04)
    p.add_argument("--drop_unicom_sitting_presence_lt", type=float, default=0.15)
    p.add_argument("--evaluate_grouped_cv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_n_estimators", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_label_map(base_dir: Path, y: np.ndarray, meta: pd.DataFrame) -> dict[int, str]:
    label_map_path = base_dir / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "label_map" in raw:
            raw = raw["label_map"]
        return {int(k): str(v) for k, v in raw.items()}

    if {"label_id", "label_name"}.issubset(meta.columns):
        mapping = (
            meta[["label_id", "label_name"]]
            .drop_duplicates()
            .set_index("label_id")["label_name"]
            .to_dict()
        )
        return {int(k): str(v) for k, v in mapping.items()}

    return {int(label_id): f"Class_{int(label_id)}" for label_id in sorted(int(v) for v in np.unique(y))}


def compute_quality_metrics(X: np.ndarray) -> dict[str, np.ndarray]:
    body = X[:, :, :68].reshape(X.shape[0], X.shape[1], 17, 4).astype(np.float32, copy=False)
    xy = body[..., :2]
    speed = body[..., 2]

    valid_joint = np.any(np.abs(body) > 1e-6, axis=-1)
    presence_ratio = valid_joint.mean(axis=(1, 2)).astype(np.float32)

    mean_joint_speed = speed.mean(axis=(1, 2)).astype(np.float32)
    hips = xy[:, :, [11, 12], :].mean(axis=2)
    hip_travel = np.linalg.norm(hips[:, -1] - hips[:, 0], axis=1).astype(np.float32)
    aspect_mean = X[:, :, -1].mean(axis=1).astype(np.float32)

    return {
        "presence_ratio": presence_ratio,
        "mean_joint_speed": mean_joint_speed,
        "hip_travel": hip_travel,
        "aspect_mean": aspect_mean,
    }


def attach_quality_columns(meta: pd.DataFrame, metrics: dict[str, np.ndarray]) -> pd.DataFrame:
    out = meta.copy()
    for key, values in metrics.items():
        out[key] = values

    presence = out["presence_ratio"].to_numpy()
    severe_missing = presence < 0.04
    low_missing = (presence >= 0.04) & (presence < 0.15)
    medium_missing = (presence >= 0.15) & (presence < 0.40)

    tier = np.full(len(out), "good", dtype=object)
    tier[medium_missing] = "medium_missing"
    tier[low_missing] = "low_missing"
    tier[severe_missing] = "severe_missing"

    out["quality_tier"] = tier
    out["is_severe_missing"] = severe_missing
    out["is_low_missing"] = low_missing | severe_missing
    return out


def build_strict_keep_mask(meta_q: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, int]]:
    keep = np.ones(len(meta_q), dtype=bool)
    reasons: dict[str, np.ndarray] = {}

    reasons["drop_unicom_standing_low_presence"] = (
        (meta_q["source"].astype(str) == "Unicomfacauca")
        & (meta_q["label_name"].astype(str) == "Standing")
        & (meta_q["presence_ratio"].to_numpy() < float(args.drop_unicom_standing_presence_lt))
    )
    reasons["drop_unicom_sitting_low_presence"] = (
        (meta_q["source"].astype(str) == "Unicomfacauca")
        & (meta_q["label_name"].astype(str) == "Sitting")
        & (meta_q["presence_ratio"].to_numpy() < float(args.drop_unicom_sitting_presence_lt))
    )

    for mask in reasons.values():
        keep &= ~mask.to_numpy()

    reason_counts = {name: int(mask.sum()) for name, mask in reasons.items()}
    reason_counts["total_dropped"] = int((~keep).sum())
    return keep, reason_counts


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_dataset(
    out_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    meta_q: pd.DataFrame,
    keep_mask: np.ndarray,
    label_map: dict[int, str],
) -> dict[str, Any]:
    ensure_dir(out_dir)

    X_out = X[keep_mask].astype(np.float32, copy=False)
    y_out = y[keep_mask].astype(np.int64, copy=False)
    meta_out = meta_q.loc[keep_mask].reset_index(drop=True)

    np.save(out_dir / "X_train.npy", X_out)
    np.save(out_dir / "y_train.npy", y_out)
    meta_out.to_csv(out_dir / "metadata_train.csv", index=False)

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label_map": label_map}, f, indent=2)

    class_dist = {
        label_map.get(int(label_id), f"Class_{int(label_id)}"): int(count)
        for label_id, count in zip(*np.unique(y_out, return_counts=True))
    }
    source_dist = {
        str(source): int(count)
        for source, count in meta_out["source"].value_counts().items()
    } if "source" in meta_out.columns else {}

    quality_dist = {
        str(tier): int(count)
        for tier, count in meta_out["quality_tier"].value_counts().items()
    } if "quality_tier" in meta_out.columns else {}

    return {
        "samples": int(len(meta_out)),
        "class_distribution": class_dist,
        "source_distribution": source_dist,
        "quality_tier_distribution": quality_dist,
    }


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
    out_full_dir = Path(args.out_full_dir)
    out_strict_dir = Path(args.out_strict_dir)

    X = np.load(base_dir / "X_train.npy").astype(np.float32)
    y = np.load(base_dir / "y_train.npy").astype(np.int64)
    meta = pd.read_csv(base_dir / "metadata_train.csv")
    if not (len(X) == len(y) == len(meta)):
        raise ValueError("Input arrays and metadata length mismatch.")

    label_map = load_label_map(base_dir, y, meta)
    metrics = compute_quality_metrics(X)
    meta_q = attach_quality_columns(meta, metrics)

    full_keep = np.ones(len(meta_q), dtype=bool)
    strict_keep, strict_drop_reasons = build_strict_keep_mask(meta_q, args)

    full_summary = save_dataset(out_full_dir, X, y, meta_q, full_keep, label_map)
    strict_summary = save_dataset(out_strict_dir, X, y, meta_q, strict_keep, label_map)

    eval_summary: dict[str, float | None] = {
        "full_grouped_cv_macro_f1_est": None,
        "strict_grouped_cv_macro_f1_est": None,
    }
    if args.evaluate_grouped_cv:
        eval_summary["full_grouped_cv_macro_f1_est"] = grouped_cv_macro_f1(
            X,
            y,
            meta_q["action_id"].astype(str).to_numpy(),
            n_estimators=int(args.eval_n_estimators),
            seed=int(args.seed),
        )
        eval_summary["strict_grouped_cv_macro_f1_est"] = grouped_cv_macro_f1(
            X[strict_keep],
            y[strict_keep],
            meta_q.loc[strict_keep, "action_id"].astype(str).to_numpy(),
            n_estimators=int(args.eval_n_estimators),
            seed=int(args.seed),
        )

    report = {
        "base_dir": str(base_dir),
        "out_full_dir": str(out_full_dir),
        "out_strict_dir": str(out_strict_dir),
        "strict_drop_rules": {
            "drop_unicom_standing_presence_lt": float(args.drop_unicom_standing_presence_lt),
            "drop_unicom_sitting_presence_lt": float(args.drop_unicom_sitting_presence_lt),
        },
        "strict_drop_counts": strict_drop_reasons,
        "full": full_summary,
        "strict": strict_summary,
        "evaluation": eval_summary,
    }

    with open(out_full_dir / "dataset_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(out_strict_dir / "dataset_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("MASTER CLEAN DATASET PREP COMPLETE")
    print("=" * 72)
    print(f"Base dataset: {base_dir}")
    print(f"Full output : {out_full_dir} | samples={full_summary['samples']}")
    print(f"Strict output: {out_strict_dir} | samples={strict_summary['samples']}")
    print(f"Strict drop counts: {strict_drop_reasons}")
    if args.evaluate_grouped_cv:
        print(
            "Grouped CV macro-F1 estimate (v1spec): "
            f"full={eval_summary['full_grouped_cv_macro_f1_est']} | "
            f"strict={eval_summary['strict_grouped_cv_macro_f1_est']}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
