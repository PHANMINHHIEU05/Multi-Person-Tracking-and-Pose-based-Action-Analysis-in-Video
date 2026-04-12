"""
train_timeline_hardcase_calibrator.py
====================================
Train a lightweight timeline-based calibrator that corrects confusing runtime
action labels (mainly Fall <-> Standing/Walking/Sitting) using fall-debug
timeline records + mined hardcase segments.

This does not replace the main pose-sequence model; it is a post-classifier
used only at runtime on ambiguous frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


TARGET_LABELS = ["Fall", "Walking", "Standing", "Sitting", "Lying_Down"]
TARGET_LABEL_TO_ID = {name: idx for idx, name in enumerate(TARGET_LABELS)}
ID_TO_TARGET_LABEL = {idx: name for name, idx in TARGET_LABEL_TO_ID.items()}
CURRENT_LABEL_BUCKETS = ["Fall", "Walking", "Standing", "Sitting", "Lying_Down", "Unknown"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train hardcase timeline calibrator for runtime label correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--timeline",
        action="append",
        default=[],
        help="Path to a timeline JSON file (repeatable).",
    )
    p.add_argument(
        "--timeline_glob",
        default="runs/qt_outputs/fall_debug_timeline_cuda_*.json",
        help="Glob for timeline JSON files.",
    )
    p.add_argument(
        "--hardcase_csv",
        default="runs/qt_outputs/hardcases_162_latest.csv",
        help="Hardcase CSV (output of mine_timeline_hardcases.py).",
    )
    p.add_argument(
        "--out_dir",
        default="runs/train_timeline_hardcase_calibrator",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_ratio", type=float, default=0.20)
    p.add_argument("--context_frames", type=int, default=2)
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--min_samples_leaf", type=int, default=2)
    p.add_argument("--hardcase_weight", type=float, default=3.0)
    p.add_argument("--fall_weight", type=float, default=1.35)
    p.add_argument("--min_calibrator_conf", type=float, default=0.62)
    p.add_argument("--activate", action="store_true", help="Write runs/active_timeline_calibrator_path.txt")
    return p.parse_args()


def normalize_label(label_raw: object) -> str:
    label = str(label_raw or "").strip()
    if label in {"", "?", "unknown", "Unknown", "Bending"}:
        return "Unknown"
    if label == "Sitting_Quickly":
        return "Sitting"
    return label


def resolve_timeline_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    for raw in args.timeline:
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            files.append(p)
    if args.timeline_glob:
        files.extend(sorted(Path().glob(args.timeline_glob)))
    dedup: list[Path] = []
    seen: set[Path] = set()
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            dedup.append(rp)
    return dedup


def load_timeline_records(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        recs = payload.get("records", [])
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            rows.append(
                {
                    "file_name": path.name,
                    "tid": int(rec.get("tid", -1)),
                    "frame": int(rec.get("frame", -1)),
                    "sec": float(rec.get("sec", 0.0)),
                    "label": normalize_label(rec.get("label", "?")),
                    "conf": float(rec.get("conf", 0.0)),
                    "fall_cue": int(bool(rec.get("fall_cue", False))),
                    "fall_vel": int(bool(rec.get("fall_vel", False))),
                    "fall_hold": int(rec.get("fall_hold", 0)),
                    "fall_recovery_votes": int(rec.get("fall_recovery_votes", 0)),
                    "down_vel": float(rec.get("down_vel", 0.0)),
                    "bbox_ar": float(rec.get("bbox_ar", 1.0)),
                }
            )
    if not rows:
        raise ValueError("No timeline records found.")
    df = pd.DataFrame(rows)
    df = df.sort_values(["file_name", "tid", "frame"]).reset_index(drop=True)
    return df


def load_hardcases(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Hardcase CSV not found: {path}")
    hc = pd.read_csv(path)
    needed = {"file_name", "tid", "start_frame", "end_frame", "suggested_label", "issue_type"}
    missing = sorted(needed - set(hc.columns))
    if missing:
        raise ValueError(f"Hardcase CSV missing columns: {missing}")
    hc = hc.copy()
    hc["file_name"] = hc["file_name"].astype(str)
    hc["tid"] = hc["tid"].astype(int)
    hc["start_frame"] = hc["start_frame"].astype(int)
    hc["end_frame"] = hc["end_frame"].astype(int)
    hc["suggested_label"] = hc["suggested_label"].map(normalize_label)
    hc["issue_type"] = hc["issue_type"].astype(str)
    return hc


def apply_hardcase_targets(df: pd.DataFrame, hc: pd.DataFrame, context_frames: int) -> pd.DataFrame:
    out = df.copy()
    out["target_label"] = out["label"].map(normalize_label)
    out["hardcase_flag"] = 0
    out["issue_type"] = "none"

    context = max(0, int(context_frames))
    for _, row in hc.iterrows():
        file_name = str(row["file_name"])
        tid = int(row["tid"])
        start_frame = int(row["start_frame"]) - context
        end_frame = int(row["end_frame"]) + context
        suggested = normalize_label(row["suggested_label"])
        issue_type = str(row["issue_type"])

        mask = (
            out["file_name"].eq(file_name)
            & out["tid"].eq(tid)
            & out["frame"].between(start_frame, end_frame)
        )
        out.loc[mask, "target_label"] = suggested
        out.loc[mask, "hardcase_flag"] = 1
        out.loc[mask, "issue_type"] = issue_type
    return out


def _rolling_mean(grouped: pd.core.groupby.generic.SeriesGroupBy, window: int) -> pd.Series:
    return grouped.rolling(window=window, min_periods=1).mean().reset_index(level=[0, 1], drop=True)


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feat = df.copy()
    key_cols = ["file_name", "tid"]
    grp = feat.groupby(key_cols, sort=False)

    feat["prev_frame"] = grp["frame"].shift(1)
    feat["frame_gap"] = (feat["frame"] - feat["prev_frame"]).fillna(1.0)
    feat["is_new_track"] = (feat["frame_gap"] > 2.0).astype(np.float32)

    for col in ["down_vel", "bbox_ar", "conf", "fall_cue", "fall_vel"]:
        feat[f"prev_{col}"] = grp[col].shift(1).fillna(feat[col])
        feat[f"delta_{col}"] = feat[col] - feat[f"prev_{col}"]

    feat["abs_down_vel"] = feat["down_vel"].abs()
    feat["abs_delta_down_vel"] = feat["delta_down_vel"].abs()
    feat["bbox_ar_inv"] = 1.0 / np.clip(feat["bbox_ar"], 1e-4, None)
    feat["fall_event"] = ((feat["fall_cue"] > 0) | (feat["fall_vel"] > 0)).astype(np.float32)

    for col in ["down_vel", "bbox_ar", "conf", "fall_event"]:
        feat[f"{col}_roll3"] = _rolling_mean(grp[col], 3)
        feat[f"{col}_roll5"] = _rolling_mean(grp[col], 5)

    feat["track_age"] = grp.cumcount().astype(np.float32) + 1.0
    feat["track_age_log"] = np.log1p(feat["track_age"])

    current_label_norm = feat["label"].map(normalize_label)
    for bucket in CURRENT_LABEL_BUCKETS:
        feat[f"cur_is_{bucket.lower()}"] = (current_label_norm == bucket).astype(np.float32)

    feature_cols = [
        "conf",
        "fall_cue",
        "fall_vel",
        "fall_hold",
        "fall_recovery_votes",
        "down_vel",
        "bbox_ar",
        "abs_down_vel",
        "delta_down_vel",
        "delta_bbox_ar",
        "delta_conf",
        "abs_delta_down_vel",
        "bbox_ar_inv",
        "fall_event",
        "frame_gap",
        "is_new_track",
        "track_age",
        "track_age_log",
        "down_vel_roll3",
        "down_vel_roll5",
        "bbox_ar_roll3",
        "bbox_ar_roll5",
        "conf_roll3",
        "conf_roll5",
        "fall_event_roll3",
        "fall_event_roll5",
        "cur_is_fall",
        "cur_is_walking",
        "cur_is_standing",
        "cur_is_sitting",
        "cur_is_lying_down",
        "cur_is_unknown",
    ]
    feat[feature_cols] = feat[feature_cols].astype(np.float32)
    return feat, feature_cols


def split_indices(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if unique_groups.shape[0] >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        tr, te = next(gss.split(X, y, groups=groups))
        return tr, te
    try:
        tr, te = train_test_split(
            np.arange(len(y)),
            test_size=test_ratio,
            random_state=seed,
            stratify=y,
        )
        return np.asarray(tr), np.asarray(te)
    except Exception:
        tr, te = train_test_split(
            np.arange(len(y)),
            test_size=test_ratio,
            random_state=seed,
            stratify=None,
        )
        return np.asarray(tr), np.asarray(te)


def safe_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_LABELS))),
        target_names=TARGET_LABELS,
        digits=4,
        zero_division=0,
        output_dict=True,
    )


def compute_distribution(series: pd.Series) -> dict[str, int]:
    values, counts = np.unique(series.to_numpy(), return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def main() -> None:
    args = parse_args()

    timeline_files = resolve_timeline_files(args)
    if not timeline_files:
        raise SystemExit("No timeline files found. Use --timeline or --timeline_glob.")

    hardcase_csv = Path(args.hardcase_csv).expanduser().resolve()
    hardcases = load_hardcases(hardcase_csv)
    records = load_timeline_records(timeline_files)
    with_targets = apply_hardcase_targets(records, hardcases, context_frames=args.context_frames)
    feat_df, feature_cols = build_feature_table(with_targets)

    train_df = feat_df[feat_df["target_label"].isin(TARGET_LABELS)].copy()
    if train_df.empty:
        raise SystemExit("No trainable rows after applying hardcase targets.")

    train_df["target_id"] = train_df["target_label"].map(TARGET_LABEL_TO_ID).astype(np.int64)
    X = train_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = train_df["target_id"].to_numpy(dtype=np.int64, copy=True)
    groups = train_df["file_name"].astype(str).to_numpy()

    sample_weight = np.ones(len(train_df), dtype=np.float32)
    sample_weight *= np.where(train_df["hardcase_flag"].to_numpy() > 0, float(args.hardcase_weight), 1.0)
    sample_weight *= np.where(train_df["target_label"].to_numpy() == "Fall", float(args.fall_weight), 1.0)
    sample_weight *= np.where(train_df["label"].to_numpy() == "Unknown", 1.20, 1.0)

    tr_idx, te_idx = split_indices(X, y, groups, test_ratio=float(args.test_ratio), seed=int(args.seed))

    model = ExtraTreesClassifier(
        n_estimators=int(args.n_estimators),
        min_samples_leaf=int(args.min_samples_leaf),
        class_weight="balanced_subsample",
        random_state=int(args.seed),
        n_jobs=-1,
    )
    model.fit(X[tr_idx], y[tr_idx], sample_weight=sample_weight[tr_idx])

    pred_te = model.predict(X[te_idx])
    acc = float(accuracy_score(y[te_idx], pred_te))
    macro_f1 = float(f1_score(y[te_idx], pred_te, average="macro"))

    baseline_pred = train_df.iloc[te_idx]["label"].map(lambda v: TARGET_LABEL_TO_ID.get(normalize_label(v), -1)).to_numpy()
    baseline_mask = baseline_pred >= 0
    baseline_acc = float(accuracy_score(y[te_idx][baseline_mask], baseline_pred[baseline_mask])) if baseline_mask.any() else 0.0

    test_hardcase_mask = train_df.iloc[te_idx]["hardcase_flag"].to_numpy() > 0
    if test_hardcase_mask.any():
        hc_true = y[te_idx][test_hardcase_mask]
        hc_pred = pred_te[test_hardcase_mask]
        hc_acc = float(accuracy_score(hc_true, hc_pred))
        hc_macro_f1 = float(f1_score(hc_true, hc_pred, average="macro"))
        hc_baseline = baseline_pred[test_hardcase_mask]
        hc_baseline_mask = hc_baseline >= 0
        hc_baseline_acc = (
            float(accuracy_score(hc_true[hc_baseline_mask], hc_baseline[hc_baseline_mask]))
            if hc_baseline_mask.any()
            else 0.0
        )
    else:
        hc_acc = 0.0
        hc_macro_f1 = 0.0
        hc_baseline_acc = 0.0

    cm = confusion_matrix(y[te_idx], pred_te, labels=list(range(len(TARGET_LABELS))))
    report = safe_classification_report(y[te_idx], pred_te)

    importance_pairs = sorted(
        zip(feature_cols, model.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = out_dir / "timeline_hardcase_calibrator.joblib"
    metrics_path = out_dir / "metrics_summary.json"
    pred_path = out_dir / "test_predictions.csv"

    artifact = {
        "model": model,
        "feature_names": feature_cols,
        "label_map": ID_TO_TARGET_LABEL,
        "label_to_id": TARGET_LABEL_TO_ID,
        "min_confidence": float(args.min_calibrator_conf),
        "hardcase_csv": str(hardcase_csv),
        "timeline_files": [str(p.resolve()) for p in timeline_files],
    }
    joblib.dump(artifact, artifact_path)

    pred_df = train_df.iloc[te_idx][
        ["file_name", "tid", "frame", "sec", "label", "target_label", "hardcase_flag", "issue_type"]
    ].copy()
    pred_df["pred_label"] = [ID_TO_TARGET_LABEL[int(v)] for v in pred_te]
    pred_df["pred_correct"] = pred_df["pred_label"].eq(pred_df["target_label"])
    pred_df.to_csv(pred_path, index=False)

    summary = {
        "timeline_files": [str(p.resolve()) for p in timeline_files],
        "hardcase_csv": str(hardcase_csv),
        "n_total_rows": int(len(feat_df)),
        "n_trainable_rows": int(len(train_df)),
        "n_hardcase_rows": int((train_df["hardcase_flag"] > 0).sum()),
        "target_distribution": compute_distribution(train_df["target_label"]),
        "holdout_accuracy": acc,
        "holdout_macro_f1": macro_f1,
        "baseline_holdout_accuracy_from_raw_label": baseline_acc,
        "hardcase_holdout_accuracy": hc_acc,
        "hardcase_holdout_macro_f1": hc_macro_f1,
        "hardcase_baseline_accuracy_from_raw_label": hc_baseline_acc,
        "hardcase_accuracy_gain_vs_baseline": hc_acc - hc_baseline_acc,
        "feature_count": int(len(feature_cols)),
        "top_feature_importance": [
            {"feature": name, "importance": float(score)}
            for name, score in importance_pairs[:20]
        ],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "label_map": ID_TO_TARGET_LABEL,
        "label_to_id": TARGET_LABEL_TO_ID,
        "min_calibrator_confidence": float(args.min_calibrator_conf),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.activate:
        active_path = Path("runs/active_timeline_calibrator_path.txt")
        active_path.write_text(str(artifact_path), encoding="utf-8")

    print("=" * 72)
    print("TRAIN TIMELINE HARDCASE CALIBRATOR")
    print("=" * 72)
    print(f"Timeline files: {len(timeline_files)}")
    print(f"Hardcase csv:   {hardcase_csv}")
    print(f"Trainable rows: {len(train_df)} (hardcase rows: {(train_df['hardcase_flag'] > 0).sum()})")
    print(f"Holdout Acc:    {acc:.4f}")
    print(f"Holdout MacroF1:{macro_f1:.4f}")
    print(f"Hardcase Acc:   {hc_acc:.4f} (baseline {hc_baseline_acc:.4f}, gain {hc_acc - hc_baseline_acc:+.4f})")
    print(f"Saved model:    {artifact_path}")
    print(f"Saved metrics:  {metrics_path}")
    print(f"Saved preds:    {pred_path}")
    if args.activate:
        print("Activated calibrator via runs/active_timeline_calibrator_path.txt")
    print("=" * 72)


if __name__ == "__main__":
    main()

