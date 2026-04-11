from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from data_prepare_v3 import compute_features, sliding_window
from extract_pose import normalize_keypoints, select_best_person
from prepare_master_clean_dataset import attach_quality_columns, compute_quality_metrics


SEGMENT_PATTERN = re.compile(r"([^;\[]+?)\[\s*([0-9]*\.?[0-9]+)\s*to\s*([0-9]*\.?[0-9]+)\s*\]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Integrate GMDCSA24 fall-transition segments into master-clean 5-action dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_dir", default="data/train_ready_action_master_clean_v2b_multicam_walkonly")
    p.add_argument(
        "--source_root",
        default="data/external/gmdcsa24/extracted/ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76",
    )
    p.add_argument("--out_dir", default="data/train_ready_action_master_clean_v3_multicam_gmdcsa24")
    p.add_argument("--weights", default="yolov8n-pose.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--frame_step", type=int, default=3)
    p.add_argument("--csv_types", default="fall", help="Comma-separated: fall,adl")
    p.add_argument("--include_labels", default="fall,walking,sitting,standing")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--window_stride", type=int, default=64)
    p.add_argument("--min_presence_ratio", type=float, default=0.12)
    p.add_argument("--min_segment_duration_sec", type=float, default=0.8)
    p.add_argument("--max_segments", type=int, default=0)
    p.add_argument("--evaluate_grouped_cv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_n_estimators", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_set(raw: str, allowed: set[str]) -> set[str]:
    out: set[str] = set()
    for part in str(raw).split(","):
        token = part.strip().lower()
        if token in allowed:
            out.add(token)
    return out


def normalize_segment_label(raw_label: str) -> str | None:
    s = str(raw_label).strip().lower()
    if "fall" in s:
        return "Fall"
    if "walk" in s:
        return "Walking"
    if "sit" in s:
        return "Sitting"
    if "stand" in s:
        return "Standing"
    if "sleep" in s or "lying" in s or "lay" in s:
        return "Lying_Down"
    return None


def parse_class_segments(classes_cell: Any) -> list[tuple[str, float, float]]:
    if not isinstance(classes_cell, str):
        return []
    out: list[tuple[str, float, float]] = []
    for match in SEGMENT_PATTERN.finditer(classes_cell):
        label_raw = match.group(1).strip()
        start_sec = float(match.group(2))
        end_sec = float(match.group(3))
        if end_sec <= start_sec:
            continue
        norm = normalize_segment_label(label_raw)
        if norm is None:
            continue
        out.append((norm, start_sec, end_sec))
    return out


def resolve_subject_dirs(source_root: Path) -> list[Path]:
    return sorted([p for p in source_root.glob("Subject *") if p.is_dir()])


def collect_segment_specs(
    source_root: Path,
    csv_types: set[str],
    include_labels: set[str],
    min_duration_sec: float,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for subj_dir in resolve_subject_dirs(source_root):
        subj_name = subj_dir.name
        if "fall" in csv_types:
            csv_path = subj_dir / "Fall.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for row in df.itertuples(index=False):
                    file_name = str(getattr(row, "File_Name", "") or getattr(row, "File Name", "")).strip()
                    if not file_name:
                        file_name = str(getattr(row, "_0", "")).strip()
                    if not file_name:
                        continue
                    video_path = subj_dir / "Fall" / file_name
                    if not video_path.exists():
                        continue
                    classes = getattr(row, "_5", None)
                    if classes is None and hasattr(row, "_6"):
                        classes = getattr(row, "_6")
                    if classes is None:
                        # Fallback by column name with trailing spaces.
                        row_dict = row._asdict()
                        classes = row_dict.get(" Classes") or row_dict.get("Classes")
                    for label_name, start_sec, end_sec in parse_class_segments(classes):
                        if label_name.lower() not in include_labels:
                            continue
                        if (end_sec - start_sec) < min_duration_sec:
                            continue
                        specs.append(
                            {
                                "subject": subj_name,
                                "source_csv": "Fall.csv",
                                "video_path": str(video_path),
                                "video_stem": Path(file_name).stem,
                                "label_name": label_name,
                                "start_sec": float(start_sec),
                                "end_sec": float(end_sec),
                            }
                        )
        if "adl" in csv_types:
            csv_path = subj_dir / "ADL.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for row in df.itertuples(index=False):
                    file_name = str(getattr(row, "File_Name", "") or getattr(row, "File Name", "")).strip()
                    if not file_name:
                        file_name = str(getattr(row, "_0", "")).strip()
                    if not file_name:
                        continue
                    video_path = subj_dir / "ADL" / file_name
                    if not video_path.exists():
                        continue
                    classes = getattr(row, "_5", None)
                    if classes is None and hasattr(row, "_6"):
                        classes = getattr(row, "_6")
                    if classes is None:
                        row_dict = row._asdict()
                        classes = row_dict.get(" Classes") or row_dict.get("Classes")
                    for label_name, start_sec, end_sec in parse_class_segments(classes):
                        if label_name.lower() not in include_labels:
                            continue
                        if (end_sec - start_sec) < min_duration_sec:
                            continue
                        specs.append(
                            {
                                "subject": subj_name,
                                "source_csv": "ADL.csv",
                                "video_path": str(video_path),
                                "video_stem": Path(file_name).stem,
                                "label_name": label_name,
                                "start_sec": float(start_sec),
                                "end_sec": float(end_sec),
                            }
                        )
    return specs


def load_label_map(base_dir: Path) -> dict[int, str]:
    with open(base_dir / "label_map.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "label_map" in raw:
        raw = raw["label_map"]
    return {int(k): str(v) for k, v in raw.items()}


def build_name_to_id(label_map: dict[int, str]) -> dict[str, int]:
    return {str(v): int(k) for k, v in label_map.items()}


def extract_segment_pose(
    model: YOLO,
    video_path: Path,
    start_sec: float,
    end_sec: float,
    *,
    conf: float,
    imgsz: int,
    frame_step: int,
) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((0, 17, 2), dtype=np.float32), 0.0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = max(start_frame + 1, int(round(end_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    step = max(1, int(frame_step))
    frames: list[np.ndarray] = []
    prev = np.zeros((17, 2), dtype=np.float32)
    frame_id = start_frame

    while frame_id <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_id - start_frame) % step != 0:
            frame_id += 1
            continue

        h, w = frame.shape[:2]
        results = model(frame, conf=conf, imgsz=imgsz, classes=[0], verbose=False)
        person = select_best_person(results[0]) if results else None
        if person is not None:
            xy, _ = person
            if np.any(xy > 0):
                prev = normalize_keypoints(xy, w, h)
        frames.append(prev.copy())
        frame_id += 1

    cap.release()
    if not frames:
        return np.zeros((0, 17, 2), dtype=np.float32), fps
    seq = np.array(frames, dtype=np.float32)
    return seq, fps


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
    if len(np.unique(groups)) < 5:
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
    source_root = Path(args.source_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_types = parse_set(args.csv_types, {"fall", "adl"})
    include_labels = parse_set(args.include_labels, {"fall", "walking", "sitting", "standing", "lying_down"})
    if not csv_types:
        raise ValueError("No valid csv_types provided.")
    if not include_labels:
        raise ValueError("No valid include_labels provided.")

    X_base = np.load(base_dir / "X_train.npy").astype(np.float32)
    y_base = np.load(base_dir / "y_train.npy").astype(np.int64)
    meta_base = pd.read_csv(base_dir / "metadata_train.csv").copy()
    if not (len(X_base) == len(y_base) == len(meta_base)):
        raise ValueError("Base dataset length mismatch.")

    label_map = load_label_map(base_dir)
    name_to_id = build_name_to_id(label_map)

    specs = collect_segment_specs(
        source_root=source_root,
        csv_types=csv_types,
        include_labels=include_labels,
        min_duration_sec=float(args.min_segment_duration_sec),
    )
    if args.max_segments > 0:
        specs = specs[: int(args.max_segments)]
    if not specs:
        raise RuntimeError("No valid GMDCSA24 segment specs found after filtering.")

    model = YOLO(args.weights)
    if args.device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                model.to("cuda")
        except Exception:
            pass

    X_ext_rows: list[np.ndarray] = []
    y_ext_rows: list[int] = []
    meta_ext_rows: list[dict[str, Any]] = []

    skipped_missing_label = 0
    skipped_low_presence = 0
    skipped_empty_seq = 0

    for seg_idx, spec in enumerate(specs):
        label_name = spec["label_name"]
        if label_name not in name_to_id:
            skipped_missing_label += 1
            continue
        label_id = int(name_to_id[label_name])

        seq_raw, fps = extract_segment_pose(
            model=model,
            video_path=Path(spec["video_path"]),
            start_sec=float(spec["start_sec"]),
            end_sec=float(spec["end_sec"]),
            conf=float(args.conf),
            imgsz=int(args.imgsz),
            frame_step=int(args.frame_step),
        )
        if seq_raw.shape[0] == 0:
            skipped_empty_seq += 1
            continue

        presence_ratio = float(np.any(np.abs(seq_raw) > 1e-6, axis=-1).mean())
        if presence_ratio < float(args.min_presence_ratio):
            skipped_low_presence += 1
            continue

        feat = compute_features(seq_raw)
        windows = sliding_window(feat, seq_len=int(args.seq_len), stride=int(args.window_stride))
        if not windows:
            skipped_empty_seq += 1
            continue

        action_id = (
            f"gmdcsa24_{spec['subject'].replace(' ', '')}_{spec['video_stem']}_"
            f"{label_name.lower()}_{seg_idx}"
        )

        for wi, win in enumerate(windows):
            X_ext_rows.append(win.astype(np.float32, copy=False))
            y_ext_rows.append(label_id)
            meta_ext_rows.append(
                {
                    "source": "GMDCSA24",
                    "action_id": action_id,
                    "aug_type": f"external_gmdcsa24_fs{int(args.frame_step)}_w{wi}",
                    "label_id": label_id,
                    "label_name": label_name,
                    "original_index": -1,
                    "repair_parent_index": -1,
                    "repair_tag": f"external_gmdcsa24_{label_name.lower()}",
                    "external_category": spec["subject"],
                    "external_video_path": spec["video_path"],
                    "gmdcsa24_source_csv": spec["source_csv"],
                    "gmdcsa24_start_sec": float(spec["start_sec"]),
                    "gmdcsa24_end_sec": float(spec["end_sec"]),
                    "gmdcsa24_fps": float(fps),
                    "gmdcsa24_presence_ratio": presence_ratio,
                }
            )

    if not X_ext_rows:
        raise RuntimeError("No GMDCSA24 windows were generated after quality filtering.")

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
        "source_root": str(source_root),
        "out_dir": str(out_dir),
        "csv_types": sorted(csv_types),
        "include_labels": sorted(include_labels),
        "frame_step": int(args.frame_step),
        "imgsz": int(args.imgsz),
        "min_presence_ratio": float(args.min_presence_ratio),
        "min_segment_duration_sec": float(args.min_segment_duration_sec),
        "max_segments": int(args.max_segments),
        "collected_segment_specs": int(len(specs)),
        "added_windows": int(len(X_ext)),
        "skipped_missing_label": int(skipped_missing_label),
        "skipped_low_presence": int(skipped_low_presence),
        "skipped_empty_seq": int(skipped_empty_seq),
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
    print("GMDCSA24 FALL-TRANSITION INTEGRATION COMPLETE")
    print("=" * 72)
    print(f"Base dataset: {base_dir}")
    print(f"Collected segment specs: {len(specs)}")
    print(f"Added windows: {len(X_ext)}")
    print(f"Output dataset: {out_dir}")
    print(f"Added class distribution: {summary['added_class_distribution']}")
    print(f"Final class distribution: {summary['final_class_distribution']}")
    if args.evaluate_grouped_cv:
        print(f"Grouped CV macro-F1 estimate: {eval_summary['grouped_cv_macro_f1_est']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
