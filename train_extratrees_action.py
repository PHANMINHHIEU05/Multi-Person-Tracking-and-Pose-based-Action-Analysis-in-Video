"""
train_extratrees_action.py
==========================
Train a high-accuracy ExtraTrees classifier for 5-action recognition from
sequence features.

Notes:
- This model is optimized for fast, strong offline accuracy.
- It reports both:
  1) Stratified holdout (common training metric)
  2) Grouped CV by base action_id (harder, leakage-resistant estimate)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, train_test_split

from src.action_model_common import (
    DEFAULT_ACTION_LABEL_MAP,
    EXTRATREES_FEATURE_SPEC_V2,
    build_extratrees_feature_matrix,
    infer_label_map,
)

# sklearn 1.8 + joblib on py3.14 may emit this warning repeatedly.
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
    category=UserWarning,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ExtraTrees action classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", default="data/train_ready_action_repair_v1")
    p.add_argument("--out_dir", default="runs/train_extratrees_action_repair_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--n_estimators", type=int, default=600)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--group_cv_splits", type=int, default=5)
    p.add_argument("--feature_spec", default=EXTRATREES_FEATURE_SPEC_V2)
    return p.parse_args()


def base_action_id(s: str) -> str:
    s = str(s)
    return s.rsplit("_aug", 1)[0] if "_aug" in s else s


def make_model(args: argparse.Namespace) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )


def grouped_cv_score(
    Xf: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float]:
    cv = GroupKFold(n_splits=args.group_cv_splits)
    accs: list[float] = []
    f1s: list[float] = []
    for tr, te in cv.split(Xf, y, groups=groups):
        model = make_model(args)
        model.fit(Xf[tr], y[tr])
        pred = model.predict(Xf[te])
        accs.append(float(accuracy_score(y[te], pred)))
        f1s.append(float(f1_score(y[te], pred, average="macro")))
    return float(np.mean(accs)), float(np.mean(f1s))


def load_label_map(data_dir: Path, meta: pd.DataFrame, y: np.ndarray) -> dict[int, str]:
    label_map_path = data_dir / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "label_map" in raw:
            raw = raw["label_map"]
        return {int(k): str(v) for k, v in raw.items()}

    if {"label_id", "label_name"}.issubset(meta.columns):
        mapping = infer_label_map(meta["label_id"].astype(int), meta["label_name"].astype(str))
        if mapping:
            return mapping

    unique_labels = sorted(int(v) for v in np.unique(y))
    return {label_id: DEFAULT_ACTION_LABEL_MAP.get(label_id, f"Class_{label_id}") for label_id in unique_labels}


def build_hard_case_summary(
    meta_te: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict[int, str],
) -> dict[str, dict[str, float | int]]:
    if "repair_tag" not in meta_te.columns:
        return {}

    summary: dict[str, dict[str, float | int]] = {}
    for repair_tag, idx in meta_te.groupby("repair_tag").groups.items():
        idx_arr = np.array(list(idx), dtype=np.int64)
        if idx_arr.size == 0:
            continue
        true_subset = y_true[idx_arr]
        pred_subset = y_pred[idx_arr]
        summary[str(repair_tag)] = {
            "n_samples": int(idx_arr.size),
            "accuracy": float(accuracy_score(true_subset, pred_subset)),
            "macro_f1": float(f1_score(true_subset, pred_subset, average="macro")),
        }
        if idx_arr.size > 0:
            true_ids, true_counts = np.unique(true_subset, return_counts=True)
            pred_ids, pred_counts = np.unique(pred_subset, return_counts=True)
            summary[str(repair_tag)]["true_distribution"] = {
                label_map.get(int(k), f"Class_{int(k)}"): int(v)
                for k, v in zip(true_ids, true_counts)
            }
            summary[str(repair_tag)]["pred_distribution"] = {
                label_map.get(int(k), f"Class_{int(k)}"): int(v)
                for k, v in zip(pred_ids, pred_counts)
            }
    return summary


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / "X_train.npy").astype(np.float32)
    y = np.load(data_dir / "y_train.npy").astype(np.int64)
    meta = pd.read_csv(data_dir / "metadata_train.csv")
    groups = meta["action_id"].astype(str).map(base_action_id).to_numpy()
    label_map = load_label_map(data_dir, meta, y)
    label_ids = sorted(label_map)
    Xf = build_extratrees_feature_matrix(X, feature_spec=args.feature_spec)

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        Xf,
        y,
        np.arange(len(y)),
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=y,
    )

    model = make_model(args)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    acc = float(accuracy_score(y_te, pred))
    macro_f1 = float(f1_score(y_te, pred, average="macro"))
    cm = confusion_matrix(y_te, pred, labels=label_ids)
    report = classification_report(
        y_te,
        pred,
        labels=label_ids,
        target_names=[label_map[i] for i in label_ids],
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    g_acc, g_f1 = grouped_cv_score(Xf, y, groups, args)
    hard_case_summary = build_hard_case_summary(meta.iloc[idx_te].reset_index(drop=True), y_te, pred, label_map)

    artifact = {
        "model": model,
        "label_map": label_map,
        "feature_spec": args.feature_spec,
        "data_dir": str(data_dir),
    }
    joblib.dump(artifact, out_dir / "extratrees_model.joblib")

    summary = {
        "stratified_holdout_accuracy": acc,
        "stratified_holdout_macro_f1": macro_f1,
        "grouped_cv_accuracy_mean": g_acc,
        "grouped_cv_macro_f1_mean": g_f1,
        "n_samples": int(len(y)),
        "n_features": int(Xf.shape[1]),
        "feature_spec": args.feature_spec,
        "label_map": label_map,
        "class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "hard_case_holdout_summary": hard_case_summary,
    }
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("TRAIN EXTRATREES ACTION MODEL")
    print("=" * 72)
    print(f"Data: {data_dir}")
    print(f"Samples: {len(y)} | Features: {Xf.shape[1]}")
    print(f"Stratified holdout Acc: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Stratified holdout Macro-F1: {macro_f1:.4f}")
    print(f"Grouped CV Acc mean: {g_acc:.4f}")
    print(f"Grouped CV Macro-F1 mean: {g_f1:.4f}")
    print(f"Saved model: {out_dir / 'extratrees_model.joblib'}")
    print(f"Saved metrics: {out_dir / 'metrics_summary.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
