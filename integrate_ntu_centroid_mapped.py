"""
integrate_ntu_centroid_mapped.py
================================
Integrate real NTU samples by discovering a stable raw-label -> 5-class mapping
with a centroid-based action signature (domain-robust, no per-window pseudo label
as primary supervision).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from data_prepare_v3 import compute_features, sliding_window
from prepare_retrain_data import normalize_pose_sequence_01, ntu25_to_coco17_xy


LABEL_NAME_MAP = {
    0: "Fall",
    1: "Walking",
    2: "Sitting_Quickly",
    3: "Bending",
    4: "Lying_Down",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Integrate NTU with centroid-mapped raw labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_dir", default="data/train_ready_unified_clean_v2_fallboost")
    p.add_argument("--out_dir", default="data/train_ready_unified_clean_v5_ntu_centroid_mapped")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--ntu_train_csv", default="data/ntu_10_actions_filtered/cross_subject_train.csv")
    p.add_argument("--ntu_test_csv", default="data/ntu_10_actions_filtered/cross_subject_test.csv")
    p.add_argument("--ntu_chunk_size", type=int, default=64)

    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--ntu_min_valid_frames", type=int, default=48)

    # Mapping discovery
    p.add_argument(
        "--map_max_rows_per_raw",
        type=int,
        default=260,
        help="Max clips per raw label used for mapping discovery (0 = all)",
    )
    p.add_argument("--map_min_support", type=int, default=120)
    p.add_argument("--map_majority_threshold", type=float, default=0.62)
    p.add_argument("--map_margin_threshold", type=float, default=0.04)

    # Integration quality filters
    p.add_argument("--target_labels", default="1,2,3")
    p.add_argument(
        "--restrict_mapping_to_target_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, discover mapping only within target_labels space",
    )
    p.add_argument("--add_clip_purity_threshold", type=float, default=0.55)
    p.add_argument("--add_window_margin_threshold", type=float, default=0.03)
    p.add_argument("--max_windows_per_clip", type=int, default=2)
    p.add_argument("--max_per_target_label", type=int, default=420)
    return p.parse_args()


def parse_label_set(raw: str) -> set[int]:
    out: set[int] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val not in LABEL_NAME_MAP:
            raise ValueError(f"Invalid label in target_labels: {val}")
        out.add(val)
    return out


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1]).astype(np.float32)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def signature(seq: np.ndarray) -> np.ndarray:
    """
    Sequence action signature from pose feature sequence (T, 69).
    """
    m = seq.mean(axis=0)
    s = seq.std(axis=0)
    d = seq[-1] - seq[0]
    return np.concatenate([m, s, d], axis=0).astype(np.float32)


def build_base_centroids(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sig = np.stack([signature(seq) for seq in X], axis=0).astype(np.float32)
    sig_mean = sig.mean(axis=0)
    sig_std = sig.std(axis=0)
    sig_std = np.where(sig_std < 1e-8, 1.0, sig_std)
    z = (sig - sig_mean) / sig_std

    centroids = np.zeros((5, z.shape[1]), dtype=np.float32)
    for k in range(5):
        idx = np.where(y == k)[0]
        if len(idx) > 0:
            centroids[k] = z[idx].mean(axis=0)
    return centroids, sig_mean.astype(np.float32), sig_std.astype(np.float32)


def classify_windows(
    wins_np: np.ndarray,
    centroids: np.ndarray,
    sig_mean: np.ndarray,
    sig_std: np.ndarray,
    candidate_labels: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sig = np.stack([signature(w) for w in wins_np], axis=0).astype(np.float32)
    z = (sig - sig_mean) / sig_std
    # Distances to 5 centroids
    diff = z[:, None, :] - centroids[None, :, :]
    dists_full = np.linalg.norm(diff, axis=2)  # (n_win, 5)

    if candidate_labels:
        cand = np.array(sorted(candidate_labels), dtype=np.int64)
        dists = dists_full[:, cand]
        best_idx = np.argmin(dists, axis=1)
        best = cand[best_idx]
    else:
        dists = dists_full
        best = np.argmin(dists, axis=1).astype(np.int64)

    # Margin between 2nd-best and best (higher = better separation)
    if dists.shape[1] >= 2:
        part = np.partition(dists, kth=1, axis=1)
        best_d = part[:, 0]
        second_d = part[:, 1]
    else:
        best_d = dists[:, 0]
        second_d = dists[:, 0]
    margin = (second_d - best_d).astype(np.float32)
    return best, margin


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


def clip_vote(
    win_labels: np.ndarray,
    win_margin: np.ndarray,
) -> tuple[int, float, float]:
    maj = int(np.bincount(win_labels, minlength=5).argmax())
    mask = win_labels == maj
    purity = float(mask.mean())
    mean_margin = float(win_margin[mask].mean()) if mask.any() else 0.0
    return maj, purity, mean_margin


def discover_mapping(
    args: argparse.Namespace,
    centroids: np.ndarray,
    sig_mean: np.ndarray,
    sig_std: np.ndarray,
    candidate_labels: set[int] | None,
) -> dict[int, int]:
    ntu_paths = [Path(args.ntu_train_csv), Path(args.ntu_test_csv)]
    ntu_paths = [p for p in ntu_paths if p.exists()]
    if not ntu_paths:
        return {}

    raw_seen: dict[int, int] = defaultdict(int)
    raw_vote: dict[int, np.ndarray] = defaultdict(lambda: np.zeros((5,), dtype=np.int64))
    raw_margin_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros((5,), dtype=np.float64))
    raw_support: dict[int, int] = defaultdict(int)

    print("\n[Mapping] Discovering raw->5class mapping by centroid votes ...")
    for csv_path in ntu_paths:
        print(f"  scanning: {csv_path}")
        for chunk in pd.read_csv(csv_path, header=None, chunksize=args.ntu_chunk_size, dtype=np.float32):
            arr = chunk.to_numpy(dtype=np.float32)
            for row in arr:
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

                wlabel, wmargin = classify_windows(
                    wins_np,
                    centroids,
                    sig_mean,
                    sig_std,
                    candidate_labels=candidate_labels,
                )
                maj, _, mean_margin = clip_vote(wlabel, wmargin)
                raw_vote[raw_label][maj] += 1
                raw_margin_sum[raw_label][maj] += mean_margin
                raw_support[raw_label] += 1

    mapping: dict[int, int] = {}
    print("\n[Mapping] Summary:")
    for raw_label in sorted(raw_support.keys()):
        support = int(raw_support[raw_label])
        vote = raw_vote[raw_label]
        top = int(np.argmax(vote))
        top_count = int(vote[top])
        ratio = float(top_count / max(support, 1))
        mean_margin = float(raw_margin_sum[raw_label][top] / max(top_count, 1))
        ok = (
            support >= args.map_min_support
            and ratio >= args.map_majority_threshold
            and mean_margin >= args.map_margin_threshold
        )
        status = "ACCEPT" if ok else "REJECT"
        print(
            f"  raw={raw_label}: support={support}, top={top}({LABEL_NAME_MAP[top]}), "
            f"ratio={ratio:.3f}, mean_margin={mean_margin:.4f} -> {status}"
        )
        if ok:
            mapping[raw_label] = top
    print(f"[Mapping] Accepted raw labels: {mapping}")
    return mapping


def add_mapped_samples(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    args: argparse.Namespace,
    centroids: np.ndarray,
    sig_mean: np.ndarray,
    sig_std: np.ndarray,
    mapping: dict[int, int],
    candidate_labels: set[int] | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if not mapping:
        print("[Integrate] No mapping accepted. Skip adding NTU samples.")
        return X, y, meta

    target_labels = parse_label_set(args.target_labels)
    ntu_paths = [Path(args.ntu_train_csv), Path(args.ntu_test_csv)]
    ntu_paths = [p for p in ntu_paths if p.exists()]

    per_label = {i: 0 for i in range(5)}
    added_X: list[np.ndarray] = []
    added_y: list[int] = []
    added_meta: list[dict] = []
    global_row = 0

    print("\n[Integrate] Adding mapped NTU windows ...")
    print(f"  target_labels={sorted(target_labels)}")

    for csv_path in ntu_paths:
        print(f"  scanning: {csv_path}")
        for chunk in pd.read_csv(csv_path, header=None, chunksize=args.ntu_chunk_size, dtype=np.float32):
            arr = chunk.to_numpy(dtype=np.float32)
            for row in arr:
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

                wlabel, wmargin = classify_windows(
                    wins_np,
                    centroids,
                    sig_mean,
                    sig_std,
                    candidate_labels=candidate_labels,
                )
                maj, purity, _ = clip_vote(wlabel, wmargin)
                if maj != mapped or purity < args.add_clip_purity_threshold:
                    continue

                idx_keep = np.where(
                    (wlabel == mapped) & (wmargin >= args.add_window_margin_threshold)
                )[0]
                if len(idx_keep) == 0:
                    continue
                idx_keep = idx_keep[np.argsort(wmargin[idx_keep])[::-1]]
                idx_keep = idx_keep[: args.max_windows_per_clip]

                action_id = f"ntu_centroid_{csv_path.stem}_r{global_row}"
                for wi in idx_keep:
                    if per_label[mapped] >= args.max_per_target_label:
                        break
                    added_X.append(wins_np[wi])
                    added_y.append(mapped)
                    added_meta.append(
                        {
                            "source": "NTU_mapped_centroid",
                            "action_id": action_id,
                            "aug_type": f"ntu_raw{raw_label}_map{mapped}_margin{wmargin[wi]:.4f}",
                            "label_id": mapped,
                            "label_name": LABEL_NAME_MAP[mapped],
                            "ntu_raw_label": raw_label,
                        }
                    )
                    per_label[mapped] += 1

    if not added_X:
        print("[Integrate] No NTU windows passed mapped quality filters.")
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

    print(f"[Integrate] Added NTU mapped windows: {len(add_y)}")
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
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(base_dir / "X_train.npy").astype(np.float32)
    y = np.load(base_dir / "y_train.npy").astype(np.int64)
    meta = pd.read_csv(base_dir / "metadata_train.csv")

    print("=" * 72)
    print("INTEGRATE NTU VIA CENTROID MAPPING")
    print("=" * 72)
    print_distribution("BASE", y, meta)

    centroids, sig_mean, sig_std = build_base_centroids(X, y)
    target_labels = parse_label_set(args.target_labels)
    candidate_labels = target_labels if args.restrict_mapping_to_target_labels else None
    mapping = discover_mapping(args, centroids, sig_mean, sig_std, candidate_labels)
    X, y, meta = add_mapped_samples(
        X=X,
        y=y,
        meta=meta,
        args=args,
        centroids=centroids,
        sig_mean=sig_mean,
        sig_std=sig_std,
        mapping=mapping,
        candidate_labels=candidate_labels,
    )
    print_distribution("FINAL", y, meta)

    np.save(out_dir / "X_train.npy", X.astype(np.float32))
    np.save(out_dir / "y_train.npy", y.astype(np.int64))
    meta.to_csv(out_dir / "metadata_train.csv", index=False)
    m, s = fit_scaler(X)
    np.save(out_dir / "feat_mean.npy", m)
    np.save(out_dir / "feat_std.npy", s)

    print("\nSaved files:")
    print(f"  - {out_dir / 'X_train.npy'}")
    print(f"  - {out_dir / 'y_train.npy'}")
    print(f"  - {out_dir / 'metadata_train.csv'}")
    print(f"  - {out_dir / 'feat_mean.npy'}")
    print(f"  - {out_dir / 'feat_std.npy'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
