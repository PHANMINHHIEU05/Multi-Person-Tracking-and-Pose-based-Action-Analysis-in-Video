"""
prepare_retrain_data.py
=======================
Prepare a cleaner retraining dataset for 5-class action recognition.

What this script does:
1) Build raw 5-class windows from UR_Fall + Multicam using data_prepare_v3.py
2) Detect low-quality Multicam pose segments (all-zero or heavy missing keypoints)
3) Filter bad segments from the generated training windows
4) Save cleaned X/y/metadata and StandardScaler statistics (feat_mean/std)

Usage:
  python prepare_retrain_data.py
  python prepare_retrain_data.py --out_dir data/train_ready_retrain_v1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare cleaned retraining dataset for 5-class action model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pose_dir", default="data/processed_pose")
    p.add_argument("--multicam_csv", default="data/processed_pose/metadata_pose_final.csv")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="data/train_ready_retrain_v1")
    p.add_argument("--tmp_dir", default="data/train_ready_retrain_v1_raw")
    p.add_argument("--max_zero_frame_ratio", type=float, default=0.95)
    p.add_argument("--max_missing_kpt_ratio", type=float, default=0.50)
    p.add_argument(
        "--target_multicam_ratio",
        type=float,
        default=0.25,
        help=(
            "Target ratio of Multicam samples after balancing (0..0.9). "
            "Set 0 to disable angle balancing."
        ),
    )
    p.add_argument(
        "--include_ntu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Integrate NTU by high-confidence pseudo-labeling",
    )
    p.add_argument("--ntu_train_csv", default="data/ntu_10_actions_filtered/cross_subject_train.csv")
    p.add_argument("--ntu_test_csv", default="data/ntu_10_actions_filtered/cross_subject_test.csv")
    p.add_argument("--ntu_checkpoint", default="runs/train_v3/final_safe_system.pth")
    p.add_argument("--ntu_chunk_size", type=int, default=64)
    p.add_argument("--ntu_stride", type=int, default=32)
    p.add_argument("--ntu_min_valid_frames", type=int, default=48)
    p.add_argument("--ntu_conf_threshold", type=float, default=0.90)
    p.add_argument("--ntu_consistency_threshold", type=float, default=0.80)
    p.add_argument("--ntu_clip_conf_threshold", type=float, default=0.92)
    p.add_argument("--ntu_min_high_conf_windows", type=int, default=2)
    p.add_argument(
        "--ntu_max_per_label",
        type=int,
        default=300,
        help="Max NTU pseudo-labeled samples to add per class (0..4)",
    )
    p.add_argument(
        "--include_sisfall",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Placeholder flag (SisFall sensor modality is not merged into this pose pipeline yet)",
    )
    p.add_argument(
        "--keep_tmp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep intermediate raw dataset directory",
    )
    return p.parse_args()


def _zero_and_missing_ratio(arr: np.ndarray) -> tuple[float, float]:
    frame_is_zero = np.all(arr == 0, axis=(1, 2))
    zero_ratio = float(frame_is_zero.mean())
    kpt_is_zero = np.all(arr == 0, axis=-1)
    missing_ratio = float(kpt_is_zero.sum() / max(arr.shape[0] * arr.shape[1], 1))
    return zero_ratio, missing_ratio


def collect_bad_multicam_action_ids(
    pose_dir: Path,
    metadata_csv: Path,
    max_zero_frame_ratio: float,
    max_missing_kpt_ratio: float,
) -> set[str]:
    if not metadata_csv.exists():
        return set()

    df = pd.read_csv(metadata_csv)
    if "source" not in df.columns:
        return set()

    mc = df[df["source"] == "Multicam"].copy()
    bad: set[str] = set()
    for _, row in mc.iterrows():
        action_id = str(row["action_id"])
        npy_path = Path(str(row["npy_path"]))
        if not npy_path.is_absolute():
            npy_path = (Path.cwd() / npy_path).resolve()
        if not npy_path.exists():
            # fall back by name convention
            npy_path = pose_dir / f"multicam_{action_id}.npy"
        if not npy_path.exists():
            bad.add(action_id)
            continue

        arr = np.load(npy_path).astype(np.float32)
        if arr.ndim != 3 or arr.shape[1] != 17 or arr.shape[2] != 2:
            bad.add(action_id)
            continue

        zero_ratio, missing_ratio = _zero_and_missing_ratio(arr)
        if zero_ratio >= max_zero_frame_ratio or missing_ratio >= max_missing_kpt_ratio:
            bad.add(action_id)

    return bad


def run_data_prepare_v3(args: argparse.Namespace, tmp_dir: Path) -> None:
    cmd = [
        sys.executable,
        "data_prepare_v3.py",
        "--pose_dir",
        args.pose_dir,
        "--multicam_csv",
        args.multicam_csv,
        "--out_dir",
        str(tmp_dir),
        "--seq_len",
        str(args.seq_len),
        "--stride",
        str(args.stride),
        "--seed",
        str(args.seed),
        "--no-horizontal_only",
    ]
    subprocess.run(cmd, check=True)


def fit_and_save_scaler(X: np.ndarray, out_dir: Path) -> None:
    flat = X.reshape(-1, X.shape[-1]).astype(np.float32)
    feat_mean = flat.mean(axis=0)
    feat_std = flat.std(axis=0)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    np.save(out_dir / "feat_mean.npy", feat_mean.astype(np.float32))
    np.save(out_dir / "feat_std.npy", feat_std.astype(np.float32))


def print_distribution(prefix: str, y: np.ndarray, meta: pd.DataFrame) -> None:
    u, c = np.unique(y, return_counts=True)
    label_dist = {int(k): int(v) for k, v in zip(u, c)}
    src_dist = meta["source"].value_counts().to_dict() if "source" in meta.columns else {}
    print(f"{prefix} samples: {len(y)}")
    print(f"{prefix} label dist: {label_dist}")
    print(f"{prefix} source dist: {src_dist}")


def rebalance_multicam_ratio(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    target_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if target_ratio <= 0.0:
        return X, y, meta
    target_ratio = max(0.0, min(target_ratio, 0.9))
    if "source" not in meta.columns:
        return X, y, meta

    src = meta["source"].astype(str).to_numpy()
    idx_mc = np.where(src == "Multicam")[0]
    if len(idx_mc) == 0:
        return X, y, meta

    n_total = len(meta)
    n_mc = len(idx_mc)
    current_ratio = n_mc / max(n_total, 1)
    if current_ratio >= target_ratio:
        return X, y, meta

    # Solve for x in: (n_mc + x) / (n_total + x) = target_ratio
    needed = int(np.ceil((target_ratio * n_total - n_mc) / (1.0 - target_ratio)))
    if needed <= 0:
        return X, y, meta

    rng = np.random.default_rng(seed)
    chosen = rng.choice(idx_mc, size=needed, replace=True)

    X_extra = X[chosen]
    y_extra = y[chosen]
    meta_extra = meta.iloc[chosen].copy()
    if "aug_type" in meta_extra.columns:
        meta_extra["aug_type"] = meta_extra["aug_type"].astype(str) + "_mc_balance"

    X_all = np.concatenate([X, X_extra], axis=0)
    y_all = np.concatenate([y, y_extra], axis=0)
    meta_all = pd.concat([meta, meta_extra], ignore_index=True)

    # Shuffle after balancing
    perm = rng.permutation(len(meta_all))
    X_all = X_all[perm]
    y_all = y_all[perm]
    meta_all = meta_all.iloc[perm].reset_index(drop=True)

    new_ratio = float((meta_all["source"] == "Multicam").mean())
    print(
        f"Applied Multicam balancing: +{needed} samples "
        f"(ratio {current_ratio:.3f} -> {new_ratio:.3f})"
    )
    return X_all, y_all, meta_all


def ntu25_to_coco17_xy(seq25: np.ndarray) -> np.ndarray:
    """
    Convert NTU 25-joint skeleton (x,y,z) to COCO-like 17 joints (x,y).
    seq25: (T, 25, 3)
    return: (T, 17, 2)
    """
    out = np.zeros((seq25.shape[0], 17, 2), dtype=np.float32)

    # NTU indices are 1-based in docs, converted here to 0-based
    idx = {
        "head": 3,
        "l_shoulder": 4,
        "l_elbow": 5,
        "l_wrist": 6,
        "r_shoulder": 8,
        "r_elbow": 9,
        "r_wrist": 10,
        "l_hip": 12,
        "l_knee": 13,
        "l_ankle": 14,
        "r_hip": 16,
        "r_knee": 17,
        "r_ankle": 18,
    }

    # Core mapping
    out[:, 0, :] = seq25[:, idx["head"], :2]        # nose <- head
    out[:, 5, :] = seq25[:, idx["l_shoulder"], :2]
    out[:, 6, :] = seq25[:, idx["r_shoulder"], :2]
    out[:, 7, :] = seq25[:, idx["l_elbow"], :2]
    out[:, 8, :] = seq25[:, idx["r_elbow"], :2]
    out[:, 9, :] = seq25[:, idx["l_wrist"], :2]
    out[:, 10, :] = seq25[:, idx["r_wrist"], :2]
    out[:, 11, :] = seq25[:, idx["l_hip"], :2]
    out[:, 12, :] = seq25[:, idx["r_hip"], :2]
    out[:, 13, :] = seq25[:, idx["l_knee"], :2]
    out[:, 14, :] = seq25[:, idx["r_knee"], :2]
    out[:, 15, :] = seq25[:, idx["l_ankle"], :2]
    out[:, 16, :] = seq25[:, idx["r_ankle"], :2]

    # Eye/Ear are unavailable in NTU-25; replicate head for topology compatibility
    out[:, 1, :] = out[:, 0, :]
    out[:, 2, :] = out[:, 0, :]
    out[:, 3, :] = out[:, 0, :]
    out[:, 4, :] = out[:, 0, :]
    return out


def normalize_pose_sequence_01(seq17: np.ndarray) -> np.ndarray | None:
    """
    Normalize sequence coordinates to [0,1] while preserving missing joints as zeros.
    """
    valid_joint_mask = np.any(np.abs(seq17) > 1e-6, axis=-1)  # (T,17)
    if valid_joint_mask.sum() < 10:
        return None

    xs = seq17[..., 0][valid_joint_mask]
    ys = seq17[..., 1][valid_joint_mask]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    rx = xmax - xmin
    ry = ymax - ymin
    if rx < 1e-6 or ry < 1e-6:
        return None

    out = seq17.copy()
    out[..., 0] = (out[..., 0] - xmin) / rx
    out[..., 1] = (out[..., 1] - ymin) / ry
    out = np.clip(out, 0.0, 1.0)
    out[~valid_joint_mask] = 0.0
    return out.astype(np.float32)


def load_pseudo_model(checkpoint_path: Path):
    from train_professional_v3 import ActionRecognitionModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ActionRecognitionModel(
        input_dim=69,
        hidden_dim=128,
        num_layers=3,
        num_classes=5,
        num_heads=8,
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    feat_mean = ckpt.get("feat_mean")
    feat_std = ckpt.get("feat_std")
    return model, feat_mean, feat_std


@torch.no_grad()
def infer_windows(model, windows: np.ndarray, feat_mean, feat_std) -> tuple[np.ndarray, np.ndarray]:
    Xw = windows.astype(np.float32)
    if feat_mean is not None and feat_std is not None:
        Xw = (Xw - feat_mean) / feat_std
    logits, _ = model(torch.from_numpy(Xw))
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    labels = probs.argmax(axis=1).astype(np.int64)
    conf = probs[np.arange(len(labels)), labels].astype(np.float32)
    return labels, conf


def add_ntu_pseudo_samples(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if not args.include_ntu:
        return X, y, meta

    ntu_paths = [Path(args.ntu_train_csv), Path(args.ntu_test_csv)]
    ntu_paths = [p for p in ntu_paths if p.exists()]
    if not ntu_paths:
        print("NTU integration skipped: NTU CSV files not found.")
        return X, y, meta

    ckpt_path = Path(args.ntu_checkpoint)
    if not ckpt_path.exists():
        print(f"NTU integration skipped: checkpoint not found at {ckpt_path}")
        return X, y, meta

    from data_prepare_v3 import compute_features, sliding_window

    model, feat_mean, feat_std = load_pseudo_model(ckpt_path)
    per_label = {i: 0 for i in range(5)}
    added_X: list[np.ndarray] = []
    added_y: list[int] = []
    added_meta: list[dict] = []

    def reached_cap() -> bool:
        return all(per_label[i] >= args.ntu_max_per_label for i in range(5))

    global_row = 0
    for csv_path in ntu_paths:
        if reached_cap():
            break
        print(f"Scanning NTU file: {csv_path}")
        for chunk in pd.read_csv(csv_path, header=None, chunksize=args.ntu_chunk_size, dtype=np.float32):
            arr = chunk.to_numpy(dtype=np.float32)
            for i in range(arr.shape[0]):
                if reached_cap():
                    break
                row = arr[i]
                global_row += 1
                if row.shape[0] != 22501:
                    continue

                raw_label = int(row[0])
                seq25 = row[1:].reshape(300, 25, 3)
                seq17 = ntu25_to_coco17_xy(seq25)

                frame_valid = np.any(np.any(np.abs(seq17) > 1e-6, axis=-1), axis=-1)
                if int(frame_valid.sum()) < args.ntu_min_valid_frames:
                    continue
                end_idx = int(np.where(frame_valid)[0][-1]) + 1
                seq17 = seq17[:end_idx]

                seq17 = normalize_pose_sequence_01(seq17)
                if seq17 is None:
                    continue

                feat = compute_features(seq17)
                wins = sliding_window(feat, args.seq_len, args.ntu_stride)
                if not wins:
                    continue
                wins_np = np.stack(wins, axis=0).astype(np.float32)

                labels, conf = infer_windows(model, wins_np, feat_mean, feat_std)
                hi = conf >= args.ntu_conf_threshold
                if int(hi.sum()) < args.ntu_min_high_conf_windows:
                    continue

                labels_hi = labels[hi]
                maj_label = int(np.bincount(labels_hi, minlength=5).argmax())
                maj_mask = hi & (labels == maj_label)
                consistency = float(maj_mask.sum() / max(int(hi.sum()), 1))
                maj_conf = float(conf[maj_mask].mean()) if int(maj_mask.sum()) > 0 else 0.0

                if consistency < args.ntu_consistency_threshold or maj_conf < args.ntu_clip_conf_threshold:
                    continue

                idx_keep = np.where(maj_mask)[0]
                # Keep highest-confidence windows first for cleaner pseudo labels
                idx_keep = idx_keep[np.argsort(conf[idx_keep])[::-1]]
                action_id = f"ntu_{csv_path.stem}_r{global_row}"
                for wi in idx_keep:
                    if per_label[maj_label] >= args.ntu_max_per_label:
                        break
                    added_X.append(wins_np[wi])
                    added_y.append(maj_label)
                    added_meta.append(
                        {
                            "source": "NTU_pseudo",
                            "action_id": action_id,
                            "aug_type": f"ntu_raw{raw_label}_conf{conf[wi]:.3f}",
                            "label_id": maj_label,
                            "label_name": "",
                        }
                    )
                    per_label[maj_label] += 1
            if reached_cap():
                break

    if not added_X:
        print("NTU integration done: no pseudo-labeled windows passed strict filters.")
        return X, y, meta

    label_name_map = {
        0: "Fall",
        1: "Walking",
        2: "Sitting_Quickly",
        3: "Bending",
        4: "Lying_Down",
    }
    add_X = np.stack(added_X, axis=0).astype(np.float32)
    add_y = np.array(added_y, dtype=np.int64)
    add_meta = pd.DataFrame(added_meta)
    add_meta["label_name"] = add_meta["label_id"].map(label_name_map)

    X_all = np.concatenate([X, add_X], axis=0)
    y_all = np.concatenate([y, add_y], axis=0)
    meta_all = pd.concat([meta, add_meta], ignore_index=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(meta_all))
    X_all = X_all[perm]
    y_all = y_all[perm]
    meta_all = meta_all.iloc[perm].reset_index(drop=True)

    print(f"NTU pseudo samples added: {len(add_y)}")
    print(f"NTU per-label added: {per_label}")
    return X_all, y_all, meta_all


def main() -> None:
    args = parse_args()
    pose_dir = Path(args.pose_dir)
    metadata_csv = Path(args.multicam_csv)
    out_dir = Path(args.out_dir)
    tmp_dir = Path(args.tmp_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PREPARE RETRAIN DATASET (5 classes)")
    print("=" * 72)
    if args.include_sisfall:
        print("NOTE: SisFall integration is not implemented in this pose-only pipeline (sensor modality mismatch).")

    bad_action_ids = collect_bad_multicam_action_ids(
        pose_dir=pose_dir,
        metadata_csv=metadata_csv,
        max_zero_frame_ratio=args.max_zero_frame_ratio,
        max_missing_kpt_ratio=args.max_missing_kpt_ratio,
    )
    print(f"Detected bad Multicam segments: {len(bad_action_ids)}")
    if bad_action_ids:
        print("Bad action_ids:")
        for aid in sorted(bad_action_ids):
            print(f"  - {aid}")

    print("\n[Step 1/3] Building raw train windows via data_prepare_v3.py ...")
    run_data_prepare_v3(args, tmp_dir)

    X = np.load(tmp_dir / "X_train.npy")
    y = np.load(tmp_dir / "y_train.npy")
    meta = pd.read_csv(tmp_dir / "metadata_train.csv")
    print_distribution("RAW", y, meta)

    print("\n[Step 2/3] Filtering low-quality Multicam segments ...")
    if bad_action_ids:
        drop_mask = (meta["source"] == "Multicam") & (meta["action_id"].astype(str).isin(bad_action_ids))
        keep_mask = ~drop_mask.to_numpy()
        X = X[keep_mask]
        y = y[keep_mask]
        meta = meta.loc[keep_mask].reset_index(drop=True)
        print(f"Dropped samples: {int(drop_mask.sum())}")
    else:
        print("No bad segments detected; no samples dropped.")
    print_distribution("CLEAN", y, meta)

    print("\n[Step 3/4] Angle balancing (UR_Fall + Multicam) ...")
    X, y, meta = rebalance_multicam_ratio(
        X=X,
        y=y,
        meta=meta,
        target_ratio=args.target_multicam_ratio,
        seed=args.seed,
    )
    print_distribution("BALANCED", y, meta)

    print("\n[Step 4/4] Integrating NTU (strict pseudo-label filters) + saving ...")
    X, y, meta = add_ntu_pseudo_samples(
        X=X,
        y=y,
        meta=meta,
        args=args,
    )
    print_distribution("FINAL", y, meta)

    np.save(out_dir / "X_train.npy", X.astype(np.float32))
    np.save(out_dir / "y_train.npy", y.astype(np.int64))
    meta.to_csv(out_dir / "metadata_train.csv", index=False)
    fit_and_save_scaler(X, out_dir)

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nSaved files:")
    print(f"  - {out_dir / 'X_train.npy'}")
    print(f"  - {out_dir / 'y_train.npy'}")
    print(f"  - {out_dir / 'metadata_train.csv'}")
    print(f"  - {out_dir / 'feat_mean.npy'}")
    print(f"  - {out_dir / 'feat_std.npy'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
