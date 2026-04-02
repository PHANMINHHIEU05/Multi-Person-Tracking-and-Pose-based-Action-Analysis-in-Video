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


LABEL_MAP = {
    0: "Fall",
    1: "Walking",
    2: "Sitting_Quickly",
    3: "Bending",
    4: "Lying_Down",
}

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
    p.add_argument("--data_dir", default="data/train_ready_unified_clean_v2_fallboost")
    p.add_argument("--out_dir", default="runs/train_extratrees_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--n_estimators", type=int, default=600)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--group_cv_splits", type=int, default=3)
    return p.parse_args()


def base_action_id(s: str) -> str:
    s = str(s)
    return s.rsplit("_aug", 1)[0] if "_aug" in s else s


def build_features(X: np.ndarray) -> np.ndarray:
    """
    X shape: (N, 128, 69)
    """
    q25 = np.quantile(X, 0.25, axis=1)
    q75 = np.quantile(X, 0.75, axis=1)
    vel = np.diff(X, axis=1)
    feat = np.concatenate(
        [
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            X[:, 0, :],
            X[:, -1, :],
            (X[:, -1, :] - X[:, 0, :]),
            q25,
            q75,
            np.mean(np.abs(vel), axis=1),
            np.std(vel, axis=1),
        ],
        axis=1,
    )
    return feat.astype(np.float32)


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


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / "X_train.npy").astype(np.float32)
    y = np.load(data_dir / "y_train.npy").astype(np.int64)
    meta = pd.read_csv(data_dir / "metadata_train.csv")
    groups = meta["action_id"].astype(str).map(base_action_id).to_numpy()

    Xf = build_features(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xf,
        y,
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=y,
    )

    model = make_model(args)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    acc = float(accuracy_score(y_te, pred))
    macro_f1 = float(f1_score(y_te, pred, average="macro"))
    cm = confusion_matrix(y_te, pred, labels=[0, 1, 2, 3, 4])
    report = classification_report(
        y_te,
        pred,
        labels=[0, 1, 2, 3, 4],
        target_names=[LABEL_MAP[i] for i in range(5)],
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    g_acc, g_f1 = grouped_cv_score(Xf, y, groups, args)

    artifact = {
        "model": model,
        "label_map": LABEL_MAP,
        "feature_spec": "mean,std,min,max,first,last,delta,q25,q75,abs_vel_mean,vel_std",
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
        "class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
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
