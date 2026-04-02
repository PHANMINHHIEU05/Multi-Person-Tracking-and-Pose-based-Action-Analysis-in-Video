from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


LABEL_NAMES = {
    0: "Fall",
    1: "Walking",
    2: "Sitting_Quickly",
    3: "Bending",
    4: "Lying_Down",
}


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


def evaluate_one(dataset_dir: Path, n_splits: int, max_iter: int) -> dict:
    X = np.load(dataset_dir / "X_train.npy").astype(np.float32)
    y = np.load(dataset_dir / "y_train.npy").astype(np.int64)
    meta = pd.read_csv(dataset_dir / "metadata_train.csv")
    groups = meta["action_id"].astype(str).map(base_action_id).to_numpy()
    feat = build_features(X)

    cv = GroupKFold(n_splits=n_splits)
    f1s: list[float] = []
    cm_total = np.zeros((5, 5), dtype=np.int64)

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
        cm_total += confusion_matrix(y[te], pred, labels=[0, 1, 2, 3, 4])

    recalls = {}
    for k in range(5):
        row = cm_total[k]
        recalls[LABEL_NAMES[k]] = float(row[k] / max(int(row.sum()), 1))

    return {
        "n": int(len(y)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "recalls": recalls,
        "cm": cm_total,
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
