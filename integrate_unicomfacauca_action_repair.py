from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from data_prepare_v3 import compute_features, sliding_window
from extract_pose import normalize_keypoints, select_best_person
from src.action_model_common import DEFAULT_ACTION_LABEL_MAP, TAXONOMY_REPAIR_ROUND3_LABEL_MAP


SAFE_CATEGORY_TO_LABEL_ID = {
    "standing": 1,
    "ADL-WALK": 2,
    "ADL-LAY": 5,
}

EXTENDED_CATEGORY_TO_LABEL_ID = {
    **SAFE_CATEGORY_TO_LABEL_ID,
    "ADL-SIT": 3,
    "ADL-GRASP": 4,
}

TAXONOMY_ROUND3_CATEGORY_TO_LABEL_ID = {
    "standing": 1,
    "ADL-WALK": 2,
    "ADL-SIT": 3,
    "ADL-LAY": 6,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Integrate selected Unicomfacauca videos into the repaired 6-class action dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source_root",
        default="data/external/unicomfacauca/extracted/Fall-Detection-Dataset-Modification",
    )
    p.add_argument("--base_dir", default="data/train_ready_action_repair_v1")
    p.add_argument("--pose_out_dir", default="data/processed_pose_unicomfacauca_v1")
    p.add_argument("--out_dir", default="data/train_ready_action_repair_v2_unicomfacauca")
    p.add_argument("--weights", default="yolov8n-pose.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--frame_step", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--window_stride", type=int, default=64)
    p.add_argument("--mapping_profile", choices=["safe", "extended", "taxonomy_round3"], default="safe")
    p.add_argument("--categories", default="standing,ADL-WALK,ADL-LAY")
    p.add_argument("--limit_per_category", type=int, default=0)
    p.add_argument("--skip_pose_extract", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def resolve_category_map(profile: str) -> dict[str, int]:
    if profile == "taxonomy_round3":
        return TAXONOMY_ROUND3_CATEGORY_TO_LABEL_ID
    if profile == "extended":
        return EXTENDED_CATEGORY_TO_LABEL_ID
    return SAFE_CATEGORY_TO_LABEL_ID


def resolve_label_map(profile: str) -> dict[int, str]:
    if profile == "taxonomy_round3":
        return TAXONOMY_REPAIR_ROUND3_LABEL_MAP
    return DEFAULT_ACTION_LABEL_MAP


def load_model(weights_path: str, device: str) -> YOLO:
    model = YOLO(weights_path)
    target_device = device
    if target_device == "cuda" and not torch.cuda.is_available():
        target_device = "cpu"
    if target_device == "cuda":
        model.to("cuda")
    return model


def iter_selected_videos(source_root: Path, categories: list[str], limit_per_category: int) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    for category in categories:
        cat_dir = source_root / category
        if not cat_dir.exists():
            continue
        videos = sorted(cat_dir.glob("*.avi"))
        if limit_per_category > 0:
            videos = videos[:limit_per_category]
        for video in videos:
            selected.append((category, video))
    return selected


def extract_pose_sequence(
    model: YOLO,
    video_path: Path,
    conf: float,
    imgsz: int,
    frame_step: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    kept = 0
    missing = 0
    prev = np.zeros((17, 2), dtype=np.float32)
    all_kpts: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(frame_step, 1) != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        result = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        person = select_best_person(result)
        if person is not None:
            xy, _ = person
            if np.any(xy > 0):
                prev = normalize_keypoints(xy, w, h)
            else:
                missing += 1
        else:
            missing += 1
        all_kpts.append(prev.copy())
        kept += 1
        frame_idx += 1

    cap.release()
    seq = np.array(all_kpts, dtype=np.float32) if all_kpts else np.zeros((0, 17, 2), dtype=np.float32)
    stats = {
        "total_frames": total_frames,
        "kept_frames": kept,
        "missing_frames": missing,
        "missing_ratio": float(missing / max(kept, 1)),
    }
    return seq, stats


def build_external_windows(
    pose_meta: pd.DataFrame,
    base_dir: Path,
    out_dir: Path,
    seq_len: int,
    window_stride: int,
    label_map: dict[int, str],
) -> dict[str, object]:
    X_base = np.load(base_dir / "X_train.npy").astype(np.float32)
    y_base = np.load(base_dir / "y_train.npy").astype(np.int64)
    meta_base = pd.read_csv(base_dir / "metadata_train.csv").copy()

    ext_X: list[np.ndarray] = []
    ext_y: list[int] = []
    ext_rows: list[dict[str, object]] = []

    for row in pose_meta.itertuples(index=False):
        seq_raw = np.load(str(row.npy_path)).astype(np.float32)
        if seq_raw.ndim != 3 or seq_raw.shape[1:] != (17, 2) or seq_raw.shape[0] == 0:
            continue
        feat = compute_features(seq_raw)
        windows = sliding_window(feat, seq_len=seq_len, stride=window_stride)
        if not windows:
            continue
        for wi, win in enumerate(windows):
            ext_X.append(win.astype(np.float32, copy=False))
            ext_y.append(int(row.label_id))
            ext_rows.append(
                {
                    "source": "Unicomfacauca",
                    "action_id": str(row.action_id),
                    "aug_type": f"ext_uc_fs{int(row.frame_step)}_w{wi}",
                    "label_id": int(row.label_id),
                    "label_name": label_map[int(row.label_id)],
                    "original_index": -1,
                    "repair_parent_index": -1,
                    "repair_tag": f"external_unicomfacauca_{row.source_category}",
                    "external_category": str(row.source_category),
                    "external_video_path": str(row.video_path),
                }
            )

    if ext_X:
        X_out = np.concatenate([X_base, np.stack(ext_X, axis=0)], axis=0).astype(np.float32, copy=False)
        y_out = np.concatenate([y_base, np.array(ext_y, dtype=np.int64)], axis=0)
        meta_out = pd.concat([meta_base, pd.DataFrame(ext_rows)], ignore_index=True)
    else:
        X_out = X_base
        y_out = y_base
        meta_out = meta_base

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", X_out)
    np.save(out_dir / "y_train.npy", y_out)
    meta_out.to_csv(out_dir / "metadata_train.csv", index=False)

    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label_map": label_map}, f, indent=2)

    summary = {
        "base_dir": str(base_dir),
        "pose_source_rows": int(len(pose_meta)),
        "external_window_count": int(len(ext_rows)),
        "external_category_clip_counts": pose_meta["source_category"].value_counts().to_dict() if len(pose_meta) else {},
        "external_label_counts": (
            pd.Series(ext_y).map(label_map).value_counts().to_dict()
            if ext_y
            else {}
        ),
        "final_class_distribution": {
            label_map[int(label_id)]: int(count)
            for label_id, count in zip(*np.unique(y_out, return_counts=True))
        },
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()

    source_root = Path(args.source_root)
    base_dir = Path(args.base_dir)
    pose_out_dir = Path(args.pose_out_dir)
    out_dir = Path(args.out_dir)
    pose_out_dir.mkdir(parents=True, exist_ok=True)

    category_map = resolve_category_map(args.mapping_profile)
    label_map = resolve_label_map(args.mapping_profile)
    categories = [part.strip() for part in str(args.categories).split(",") if part.strip()]
    unknown_categories = [category for category in categories if category not in category_map]
    if unknown_categories:
        raise ValueError(
            f"Categories {unknown_categories} are not available for mapping profile '{args.mapping_profile}'. "
            f"Allowed categories: {sorted(category_map)}"
        )
    selected_videos = iter_selected_videos(source_root, categories, args.limit_per_category)
    if not selected_videos:
        raise RuntimeError(f"No videos found for categories: {categories}")

    pose_meta_rows: list[dict[str, object]] = []
    model: YOLO | None = None

    if not args.skip_pose_extract:
        model = load_model(args.weights, args.device)

    t0 = time.time()
    for idx, (category, video_path) in enumerate(selected_videos, start=1):
        label_id = category_map[category]
        action_id = f"uc_{category}_{video_path.stem}"
        npy_path = pose_out_dir / f"{action_id}.npy"

        if npy_path.exists():
            seq = np.load(npy_path).astype(np.float32)
            stats = {
                "total_frames": int(seq.shape[0] * args.frame_step),
                "kept_frames": int(seq.shape[0]),
                "missing_frames": 0,
                "missing_ratio": 0.0,
            }
        else:
            if model is None:
                raise RuntimeError("Pose extraction skipped but cached pose file is missing: " + str(npy_path))
            seq, stats = extract_pose_sequence(
                model=model,
                video_path=video_path,
                conf=args.conf,
                imgsz=args.imgsz,
                frame_step=args.frame_step,
            )
            np.save(npy_path, seq.astype(np.float32, copy=False))

        pose_meta_rows.append(
            {
                "source": "Unicomfacauca",
                "source_category": category,
                "video_path": str(video_path),
                "action_id": action_id,
                "label_id": label_id,
                "label_name": label_map[label_id],
                "npy_path": str(npy_path),
                "frame_step": int(args.frame_step),
                **stats,
            }
        )
        if idx % 20 == 0 or idx == len(selected_videos):
            elapsed = time.time() - t0
            print(
                f"[{idx}/{len(selected_videos)}] {category} -> {npy_path.name} | "
                f"elapsed={elapsed:.1f}s"
            )

    pose_meta = pd.DataFrame(pose_meta_rows)
    pose_meta.to_csv(pose_out_dir / "metadata_unicomfacauca_pose.csv", index=False)

    summary = build_external_windows(
        pose_meta=pose_meta,
        base_dir=base_dir,
        out_dir=out_dir,
        seq_len=args.seq_len,
        window_stride=args.window_stride,
        label_map=label_map,
    )

    print("=" * 72)
    print("UNICOMFACAUCA INTEGRATION COMPLETE")
    print("=" * 72)
    print(f"Selected videos: {len(selected_videos)}")
    print(f"Pose cache dir: {pose_out_dir}")
    print(f"Output dataset: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
