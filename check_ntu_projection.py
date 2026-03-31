from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from data_prepare_v3 import compute_features, sliding_window
from prepare_retrain_data import (
    infer_windows,
    load_pseudo_model,
    normalize_pose_sequence_01,
)


IDX = {
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


def map17(seq25: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
    out = np.zeros((seq25.shape[0], 17, 2), dtype=np.float32)
    out[:, 0, :] = seq25[:, IDX["head"], dims]
    out[:, 5, :] = seq25[:, IDX["l_shoulder"], dims]
    out[:, 6, :] = seq25[:, IDX["r_shoulder"], dims]
    out[:, 7, :] = seq25[:, IDX["l_elbow"], dims]
    out[:, 8, :] = seq25[:, IDX["r_elbow"], dims]
    out[:, 9, :] = seq25[:, IDX["l_wrist"], dims]
    out[:, 10, :] = seq25[:, IDX["r_wrist"], dims]
    out[:, 11, :] = seq25[:, IDX["l_hip"], dims]
    out[:, 12, :] = seq25[:, IDX["r_hip"], dims]
    out[:, 13, :] = seq25[:, IDX["l_knee"], dims]
    out[:, 14, :] = seq25[:, IDX["r_knee"], dims]
    out[:, 15, :] = seq25[:, IDX["l_ankle"], dims]
    out[:, 16, :] = seq25[:, IDX["r_ankle"], dims]
    out[:, 1, :] = out[:, 0, :]
    out[:, 2, :] = out[:, 0, :]
    out[:, 3, :] = out[:, 0, :]
    out[:, 4, :] = out[:, 0, :]
    return out


def normalize_root(seq17: np.ndarray) -> np.ndarray | None:
    out = seq17.copy()
    valid = np.any(np.abs(out) > 1e-6, axis=-1)
    if int(valid.sum()) < 10:
        return None

    hips_valid = valid[:, 11] & valid[:, 12]
    if hips_valid.any():
        root = (out[:, 11, :] + out[:, 12, :]) / 2.0
    else:
        root = out[:, 0, :]
    out = out - root[:, None, :]

    xs = out[..., 0][valid]
    ys = out[..., 1][valid]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    rx, ry = xmax - xmin, ymax - ymin
    if rx < 1e-6 or ry < 1e-6:
        return None

    out[..., 0] = (out[..., 0] - xmin) / rx
    out[..., 1] = (out[..., 1] - ymin) / ry
    out = np.clip(out, 0.0, 1.0)
    out[~valid] = 0.0
    return out.astype(np.float32)


def eval_projection(
    model,
    feat_mean,
    feat_std,
    dims_name: str,
    dims: tuple[int, int],
    use_root_norm: bool,
    max_rows: int = 60,
) -> None:
    labels_counter: Counter[int] = Counter()
    conf_sum: Counter[int] = Counter()
    n_rows = 0

    paths = [
        Path("data/ntu_10_actions_filtered/cross_subject_train.csv"),
        Path("data/ntu_10_actions_filtered/cross_subject_test.csv"),
    ]
    for csv_path in paths:
        for chunk in pd.read_csv(csv_path, header=None, chunksize=64, dtype=np.float32):
            arr = chunk.to_numpy(dtype=np.float32)
            for row in arr:
                if row.shape[0] != 22501:
                    continue
                seq25 = row[1:].reshape(300, 25, 3)
                seq17 = map17(seq25, dims)
                frame_valid = np.any(np.any(np.abs(seq17) > 1e-6, axis=-1), axis=-1)
                if int(frame_valid.sum()) < 48:
                    continue
                seq17 = seq17[: int(np.where(frame_valid)[0][-1]) + 1]
                if use_root_norm:
                    seq17 = normalize_root(seq17)
                else:
                    seq17 = normalize_pose_sequence_01(seq17)
                if seq17 is None:
                    continue
                feat = compute_features(seq17)
                wins = sliding_window(feat, 128, 32)
                if not wins:
                    continue
                wins_np = np.stack(wins, axis=0).astype(np.float32)
                pred, conf = infer_windows(model, wins_np, feat_mean, feat_std)
                maj = int(np.bincount(pred, minlength=5).argmax())
                labels_counter[maj] += 1
                conf_sum[maj] += float(conf[pred == maj].mean()) if np.any(pred == maj) else 0.0
                n_rows += 1
                if n_rows >= max_rows:
                    break
            if n_rows >= max_rows:
                break
        if n_rows >= max_rows:
            break

    avg_conf = {
        k: (conf_sum[k] / labels_counter[k])
        for k in labels_counter
        if labels_counter[k] > 0
    }
    norm_name = "root_norm" if use_root_norm else "minmax_norm"
    print(f"\n{dims_name} {norm_name} rows={n_rows}", flush=True)
    print(f"clip maj dist={dict(sorted(labels_counter.items()))}", flush=True)
    print(
        f"avg conf by class={{ {', '.join(f'{k}: {avg_conf[k]:.3f}' for k in sorted(avg_conf))} }}",
        flush=True,
    )


def main() -> None:
    ckpt = Path("runs/train_v3/final_safe_system.pth")
    model, feat_mean, feat_std = load_pseudo_model(ckpt)

    for dims_name, dims in [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]:
        eval_projection(model, feat_mean, feat_std, dims_name, dims, use_root_norm=False, max_rows=60)
        eval_projection(model, feat_mean, feat_std, dims_name, dims, use_root_norm=True, max_rows=60)


if __name__ == "__main__":
    main()
