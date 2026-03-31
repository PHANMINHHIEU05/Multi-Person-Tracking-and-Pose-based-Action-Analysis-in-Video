"""
integrate_ntu_mapped.py
=======================
Integrate NTU samples into an existing train-ready dataset using a stable
raw-label -> 5-class mapping discovered from clip-level consensus.

This is designed to add more real NTU diversity for confusion-prone classes
without relying on per-window pseudo labels as the primary supervision signal.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from data_prepare_v3 import compute_features, sliding_window
from prepare_retrain_data import (
    infer_windows,
    load_pseudo_model,
    normalize_pose_sequence_01,
    ntu25_to_coco17_xy,
)


LABEL_NAME_MAP = {
    0: "Fall",
    1: "Walking",
    2: "Sitting_Quickly",
    3: "Bending",
    4: "Lying_Down",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build NTU raw-mapped real-data enhanced train dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_dir", default="data/train_ready_unified_clean_v2_fallboost")
    p.add_argument("--out_dir", default="data/train_ready_unified_clean_v4_ntu_mapped")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--ntu_train_csv", default="data/ntu_10_actions_filtered/cross_subject_train.csv")
    p.add_argument("--ntu_test_csv", default="data/ntu_10_actions_filtered/cross_subject_test.csv")
    p.add_argument("--ntu_checkpoint", default="runs/train_v3/final_safe_system.pth")
    p.add_argument("--ntu_chunk_size", type=int, default=64)

    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--ntu_min_valid_frames", type=int, default=48)

    # Mapping discovery controls
    p.add_argument(
        "--map_max_rows_per_raw",
        type=int,
        default=220,
        help="Max clips per raw label used for mapping discovery (0 = no cap)",
    )
    p.add_argument("--map_conf_threshold", type=float, default=0.80)
    p.add_argument("--map_min_high_conf_windows", type=int, default=2)
    p.add_argument("--map_min_support", type=int, default=80)
    p.add_argument("--map_purity_threshold", type=float, default=0.68)

    # Add stage controls
    p.add_argument(
        "--target_labels",
        default="1,2,3",
        help="Target 5-class labels to enrich from NTU mapping",
    )
    p.add_argument("--add_conf_threshold", type=float, default=0.75)
    p.add_argument("--add_min_high_conf_windows", type=int, default=2)
    p.add_argument("--add_consistency_threshold", type=float, default=0.55)
    p.add_argument("--max_windows_per_clip", type=int, default=2)
    p.add_argument("--max_per_target_label", type=int, default=380)
    return p.parse_args()


def parse_label_set(raw: str) -> set[int]:
    labels: set[int] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val not in LABEL_NAME_MAP:
            raise ValueError(f"Invalid label id in target_labels: {val}")
        labels.add(val)
    return labels


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1]).astype(np.float32)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def preprocess_ntu_row(
    row: np.ndarray,
    seq_len: int,
    stride: int,
    min_valid_frames: int,
) -> tuple[int, np.ndarray] | None:
    if row.shape[0] != 22501:
        return None
    raw_label = int(row[0])
    seq25 = row[1:].reshape(300, 25, 3)
    seq17 = ntu25_to_coco17_xy(seq25)

    frame_valid = np.any(np.any(np.abs(seq17) > 1e-6, axis=-1), axis=-1)
    if int(frame_valid.sum()) < min_valid_frames:
        return None
    end_idx = int(np.where(frame_valid)[0][-1]) + 1
    seq17 = seq17[:end_idx]

    seq17 = normalize_pose_sequence_01(seq17)
    if seq17 is None:
        return None

    feat = compute_features(seq17)
    wins = sliding_window(feat, seq_len, stride)
    if not wins:
        return None
    wins_np = np.stack(wins, axis=0).astype(np.float32)
    return raw_label, wins_np


def clip_consensus(
    labels: np.ndarray,
    conf: np.ndarray,
    conf_threshold: float,
    min_high_conf_windows: int,
) -> tuple[int, float, float, np.ndarray] | None:
    hi = conf >= conf_threshold
    if int(hi.sum()) < min_high_conf_windows:
        return None

    labels_hi = labels[hi]
    maj_label = int(np.bincount(labels_hi, minlength=5).argmax())
    maj_mask = hi & (labels == maj_label)
    consistency = float(maj_mask.sum() / max(int(hi.sum()), 1))
    maj_conf = float(conf[maj_mask].mean()) if int(maj_mask.sum()) > 0 else 0.0
    return maj_label, consistency, maj_conf, maj_mask


def discover_raw_mapping(args: argparse.Namespace) -> dict[int, int]:
    ntu_paths = [Path(args.ntu_train_csv), Path(args.ntu_test_csv)]
    ntu_paths = [p for p in ntu_paths if p.exists()]
    if not ntu_paths:
        return {}

    model, feat_mean, feat_std = load_pseudo_model(Path(args.ntu_checkpoint))
    raw_scores: dict[int, np.ndarray] = defaultdict(lambda: np.zeros((5,), dtype=np.float64))
    raw_support: dict[int, int] = defaultdict(int)
    raw_seen: dict[int, int] = defaultdict(int)

    print("\n[Mapping] Discovering stable NTU raw-label mapping ...")
    for csv_path in ntu_paths:
        print(f"  scanning: {csv_path}")
        for chunk in pd.read_csv(csv_path, header=None, chunksize=args.ntu_chunk_size, dtype=np.float32):
            arr = chunk.to_numpy(dtype=np.float32)
            for i in range(arr.shape[0]):
                row = arr[i]
                prep = preprocess_ntu_row(
                    row=row,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    min_valid_frames=args.ntu_min_valid_frames,
                )
                if prep is None:
                    continue
                raw_label, wins_np = prep
                if args.map_max_rows_per_raw > 0 and raw_seen[raw_label] >= args.map_max_rows_per_raw:
                    continue
                raw_seen[raw_label] += 1

                labels, conf = infer_windows(model, wins_np, feat_mean, feat_std)
                cc = clip_consensus(
                    labels=labels,
                    conf=conf,
                    conf_threshold=args.map_conf_threshold,
                    min_high_conf_windows=args.map_min_high_conf_windows,
                )
                if cc is None:
                    continue
                maj, consistency, maj_conf, _ = cc
                score = float(consistency * maj_conf)
                raw_scores[raw_label][maj] += score
                raw_support[raw_label] += 1

    mapping: dict[int, int] = {}
    print("\n[Mapping] Summary by raw label:")
    for raw_label in sorted(raw_scores.keys()):
        score_vec = raw_scores[raw_label]
        total = float(score_vec.sum())
        top_cls = int(np.argmax(score_vec))
        top_score = float(score_vec[top_cls])
        purity = (top_score / total) if total > 0 else 0.0
        support = int(raw_support[raw_label])
        ok = support >= args.map_min_support and purity >= args.map_purity_threshold
        status = "ACCEPT" if ok else "REJECT"
        print(
            f"  raw={raw_label}: support={support}, top={top_cls}({LABEL_NAME_MAP[top_cls]}), "
            f"purity={purity:.3f} -> {status}"
        )
        if ok:
            mapping[raw_label] = top_cls

    print(f"[Mapping] Accepted raw labels: {mapping}")
    return mapping


def add_mapped_ntu_samples(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    args: argparse.Namespace,
    mapping: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if not mapping:
        print("No accepted mapping. Skip NTU mapped integration.")
        return X, y, meta

    target_labels = parse_label_set(args.target_labels)
    model, feat_mean, feat_std = load_pseudo_model(Path(args.ntu_checkpoint))
    ntu_paths = [Path(args.ntu_train_csv), Path(args.ntu_test_csv)]
    ntu_paths = [p for p in ntu_paths if p.exists()]

    per_label = {i: 0 for i in range(5)}
    added_X: list[np.ndarray] = []
    added_y: list[int] = []
    added_meta: list[dict] = []
    global_row = 0

    print("\n[Integrate] Adding NTU mapped samples ...")
    print(f"  target_labels={sorted(target_labels)} max_per_target_label={args.max_per_target_label}")

    for csv_path in ntu_paths:
        print(f"  scanning: {csv_path}")
        for chunk in pd.read_csv(csv_path, header=None, chunksize=args.ntu_chunk_size, dtype=np.float32):
            arr = chunk.to_numpy(dtype=np.float32)
            for i in range(arr.shape[0]):
                row = arr[i]
                global_row += 1
                prep = preprocess_ntu_row(
                    row=row,
                    seq_len=args.seq_len,
                    stride=args.stride,
                    min_valid_frames=args.ntu_min_valid_frames,
                )
                if prep is None:
                    continue
                raw_label, wins_np = prep
                if raw_label not in mapping:
                    continue
                mapped = mapping[raw_label]
                if mapped not in target_labels:
                    continue
                if per_label[mapped] >= args.max_per_target_label:
                    continue

                labels, conf = infer_windows(model, wins_np, feat_mean, feat_std)
                cc = clip_consensus(
                    labels=labels,
                    conf=conf,
                    conf_threshold=args.add_conf_threshold,
                    min_high_conf_windows=args.add_min_high_conf_windows,
                )
                if cc is None:
                    continue
                maj, consistency, _, _ = cc
                if maj != mapped or consistency < args.add_consistency_threshold:
                    continue

                idx_keep = np.where((conf >= args.add_conf_threshold) & (labels == mapped))[0]
                if len(idx_keep) == 0:
                    continue
                idx_keep = idx_keep[np.argsort(conf[idx_keep])[::-1]]
                idx_keep = idx_keep[: args.max_windows_per_clip]

                action_id = f"ntu_mapped_{csv_path.stem}_r{global_row}"
                for wi in idx_keep:
                    if per_label[mapped] >= args.max_per_target_label:
                        break
                    added_X.append(wins_np[wi])
                    added_y.append(mapped)
                    added_meta.append(
                        {
                            "source": "NTU_mapped_raw",
                            "action_id": action_id,
                            "aug_type": f"ntu_raw{raw_label}_map{mapped}_conf{conf[wi]:.3f}",
                            "label_id": mapped,
                            "label_name": LABEL_NAME_MAP[mapped],
                            "ntu_raw_label": raw_label,
                        }
                    )
                    per_label[mapped] += 1

    if not added_X:
        print("[Integrate] No NTU mapped samples passed quality filters.")
        return X, y, meta

    add_X = np.stack(added_X, axis=0).astype(np.float32)
    add_y = np.array(added_y, dtype=np.int64)
    add_meta = pd.DataFrame(added_meta)

    X_all = np.concatenate([X, add_X], axis=0)
    y_all = np.concatenate([y, add_y], axis=0)
    meta_all = pd.concat([meta, add_meta], ignore_index=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(meta_all))
    X_all = X_all[perm]
    y_all = y_all[perm]
    meta_all = meta_all.iloc[perm].reset_index(drop=True)

    print(f"[Integrate] NTU mapped samples added: {len(add_y)}")
    print(f"[Integrate] per-label added: {per_label}")
    return X_all, y_all, meta_all


def print_distribution(prefix: str, y: np.ndarray, meta: pd.DataFrame) -> None:
    unique, counts = np.unique(y, return_counts=True)
    label_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    src_dist = meta["source"].value_counts().to_dict() if "source" in meta.columns else {}
    print(f"{prefix} samples: {len(y)}")
    print(f"{prefix} label dist: {label_dist}")
    print(f"{prefix} source dist: {src_dist}")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    ckpt = Path(args.ntu_checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir not found: {base_dir}")
    if not ckpt.exists():
        raise FileNotFoundError(f"ntu_checkpoint not found: {ckpt}")

    X = np.load(base_dir / "X_train.npy").astype(np.float32)
    y = np.load(base_dir / "y_train.npy").astype(np.int64)
    meta = pd.read_csv(base_dir / "metadata_train.csv")

    print("=" * 72)
    print("INTEGRATE NTU RAW-MAPPED REAL DATA")
    print("=" * 72)
    print_distribution("BASE", y, meta)

    mapping = discover_raw_mapping(args)
    X, y, meta = add_mapped_ntu_samples(X, y, meta, args, mapping)
    print_distribution("FINAL", y, meta)

    np.save(out_dir / "X_train.npy", X.astype(np.float32))
    np.save(out_dir / "y_train.npy", y.astype(np.int64))
    meta.to_csv(out_dir / "metadata_train.csv", index=False)
    mean, std = fit_scaler(X)
    np.save(out_dir / "feat_mean.npy", mean)
    np.save(out_dir / "feat_std.npy", std)

    print("\nSaved files:")
    print(f"  - {out_dir / 'X_train.npy'}")
    print(f"  - {out_dir / 'y_train.npy'}")
    print(f"  - {out_dir / 'metadata_train.csv'}")
    print(f"  - {out_dir / 'feat_mean.npy'}")
    print(f"  - {out_dir / 'feat_std.npy'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
