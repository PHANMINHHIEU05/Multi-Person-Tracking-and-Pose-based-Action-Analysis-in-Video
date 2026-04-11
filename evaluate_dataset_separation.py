from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.action_model_common import DEFAULT_ACTION_LABEL_MAP, infer_label_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick grouped-separation benchmark")
    p.add_argument(
        "--datasets",
        default="data/train_ready_unified_clean_v2_fallboost,data/train_ready_unified_clean_v7_ntu_walk_sit",
        help="Comma-separated dataset dirs",
    )
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--max_iter", type=int, default=700)
    return p.parse_args()


def base_action_id(v: str) -> str:
    s = str(v)
    return s.rsplit("_aug", 1)[0] if "_aug" in s else s


def build_features(X: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            X.mean(axis=1),
            X.std(axis=1),
            X[:, 0, :],
            X[:, -1, :],
            X.max(axis=1),
            X.min(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def load_label_map(dataset_dir: Path, meta: pd.DataFrame) -> dict[int, str]:
    label_map_path = dataset_dir / "label_map.json"
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
    unique_ids = sorted(int(v) for v in meta["label_id"].unique())
    return {label_id: DEFAULT_ACTION_LABEL_MAP.get(label_id, f"Class_{label_id}") for label_id in unique_ids}


def evaluate_one(dataset_dir: Path, n_splits: int, max_iter: int) -> dict:
    X = np.load(dataset_dir / "X_train.npy").astype(np.float32)
    y = np.load(dataset_dir / "y_train.npy").astype(np.int64)
    meta = pd.read_csv(dataset_dir / "metadata_train.csv")
    label_map = load_label_map(dataset_dir, meta)
    label_ids = sorted(label_map)
    groups = meta["action_id"].astype(str).map(base_action_id).to_numpy()
    feat = build_features(X)

    cv = GroupKFold(n_splits=n_splits)
    f1s: list[float] = []
    cm_total = np.zeros((len(label_ids), len(label_ids)), dtype=np.int64)

    for tr, te in cv.split(feat, y, groups=groups):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(feat[tr])
        Xte = scaler.transform(feat[te])

        clf = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            class_weight="balanced",
        )
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        f1s.append(float(f1_score(y[te], pred, average="macro")))
        cm_total += confusion_matrix(y[te], pred, labels=label_ids)

    recalls = {}
    for idx, label_id in enumerate(label_ids):
        row = cm_total[idx]
        recalls[label_map[label_id]] = float(row[idx] / max(int(row.sum()), 1))

    return {
        "n": int(len(y)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "recalls": recalls,
        "cm": cm_total,
        "label_map": label_map,
    }


def main() -> None:
    args = parse_args()
    ds_list = [Path(s.strip()) for s in args.datasets.split(",") if s.strip()]
    for ds in ds_list:
        print(f"\n=== {ds} ===")
        if not ds.exists():
            print("MISSING")
            continue
        r = evaluate_one(ds, args.n_splits, args.max_iter)
        print(f"n={r['n']}")
        print(f"macro_f1={r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}")
        print(f"recalls={ {k: round(v,3) for k,v in r['recalls'].items()} }")
        print("cm rows=true cols=pred")
        print(r["cm"])


if __name__ == "__main__":
    main()
