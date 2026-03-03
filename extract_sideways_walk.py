"""
extract_sideways_walk.py
────────────────────────────────────────────────────────────────────────────
Trích xuất chuỗi keypoint (128 frame) từ video của người đi bộ ngang camera,
apply đúng compute_features() pipeline, lưu thành .npy để merge vào
data/train_ready_horizontal/.

Usage:
    python extract_sideways_walk.py --video data/video/video1.mp4 \
        --out data/train_ready_horizontal/sideways_walk.npy \
        --vis  # optional: hiện overlay
"""

import argparse, sys, cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '.')
from data_prepare_v3 import compute_features, interpolate_zeros

SEQ_LEN   = 128   # frames per sample
STRIDE    = 32    # step size between samples (overlap for more data)
MIN_TRACK = 140   # minimum track length to extract one sample
CONF_KP   = 0.3   # min keypoint confidence

# ──────────────────────────────────────────────────────────────────────────
def is_upright(kpts_norm: np.ndarray) -> bool:
    """Kiểm tra xem người đang đứng thẳng (không gục, không ngã).
    kpts_norm: (17,2) normalized to [0,1].
    Returns True nếu:
      - nose_y < hip_y  (đầu trên hông)
      - hip_y < ankle_y (hông trên mắt cá)
      - aspect_ratio < 0.5 (người đứng thẳng, không nằm)
    """
    nose_y   = kpts_norm[0,  1]
    hip_y    = (kpts_norm[11,1] + kpts_norm[12,1]) / 2.0
    ankle_y  = (kpts_norm[15,1] + kpts_norm[16,1]) / 2.0

    # Lọc zero keypoints
    valid = np.any(kpts_norm != 0, axis=1)
    if valid.sum() < 8:
        return False

    vk = kpts_norm[valid]
    w = vk[:,0].max() - vk[:,0].min()
    h = max(vk[:,1].max() - vk[:,1].min(), 1e-4)
    ar = w / h

    return (nose_y < hip_y < ankle_y) and (ar < 0.55)


def extract_sequences(video_path: str, vis: bool = False) -> list[np.ndarray]:
    """
    Chạy YOLOv8-pose + BotSort trên video, thu thập raw keypoints theo track.
    Trả về list các (128,17,2) float32 array.
    """
    from ultralytics import YOLO
    model = YOLO('yolov8n-pose.pt')

    cap = cv2.VideoCapture(video_path)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Video: {video_path}  {W}×{H}  {fps:.1f} fps")

    if vis:
        cv2.namedWindow("extract", cv2.WINDOW_NORMAL)

    # track_id → deque of (17,2)
    track_buf: dict[int, list[np.ndarray]] = defaultdict(list)
    track_upright: dict[int, int] = defaultdict(int)   # count upright frames

    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1

        results = model.track(
            frame, persist=True, conf=0.25, classes=[0],
            tracker="config/botsort.yaml", verbose=False
        )
        r = results[0]

        if r.boxes.id is None:
            continue

        ids  = r.boxes.id.int().cpu().numpy()
        kpts = r.keypoints

        for i, tid in enumerate(ids):
            if kpts is None or i >= len(kpts.xy):
                continue
            kp = kpts.xy[i].cpu().numpy().copy()
            kp[:, 0] /= W
            kp[:, 1] /= H
            kp = np.clip(kp, 0.0, 1.0)

            track_buf[tid].append(kp)
            if is_upright(kp):
                track_upright[tid] += 1

        if vis:
            vis_frame = r.plot()
            cv2.imshow("extract", vis_frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    if vis:
        cv2.destroyAllWindows()

    print(f"[INFO] Processed {fi} frames, {len(track_buf)} unique tracks")

    # ── Extract sequences from stable tracks ─────────────────────────────
    seqs = []
    for tid, buf in track_buf.items():
        n = len(buf)
        upright_ratio = track_upright[tid] / max(n, 1)
        if n < MIN_TRACK or upright_ratio < 0.7:
            continue

        # Sliding window with STRIDE
        for start in range(0, n - SEQ_LEN + 1, STRIDE):
            window = buf[start : start + SEQ_LEN]
            arr = np.array(window, dtype=np.float32)   # (128,17,2)

            # Double-check majority of frames are upright
            ok = sum(is_upright(arr[t]) for t in range(0, SEQ_LEN, 8))
            if ok < SEQ_LEN // 8 * 0.7:
                continue

            seqs.append(arr)

    print(f"[INFO] Extracted {len(seqs)} raw walking sequences (128 frames each)")
    return seqs


# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True)
    parser.add_argument("--out",    default="data/train_ready_horizontal/sideways_walk.npy")
    parser.add_argument("--vis",    action="store_true")
    parser.add_argument("--n_aug",  type=int, default=2,
                        help="số augmentation variants mỗi sequence")
    args = parser.parse_args()

    raw_seqs = extract_sequences(args.video, vis=args.vis)
    if not raw_seqs:
        print("[ERROR] Không có sequence nào được trích xuất!"); sys.exit(1)

    rng = np.random.default_rng(42)
    all_features = []

    for seq in raw_seqs:
        # orig
        all_features.append(compute_features(seq))
        # flip
        all_features.append(compute_features(seq, do_flip=True))
        # small tilt (vì camera thực có thể hơi nghiêng)
        for tilt in [rng.uniform(2,8), rng.uniform(-8,-2)]:
            all_features.append(compute_features(seq, tilt_angle=tilt))
        # scale
        for sc in [rng.uniform(0.8,0.95), rng.uniform(1.05,1.2)]:
            all_features.append(compute_features(seq, scale_factor=sc))
        # extra aug
        for _ in range(args.n_aug):
            tilt = rng.uniform(-6,6)  if rng.random()>0.4 else 0.0
            sc   = rng.uniform(0.8,1.2) if rng.random()>0.4 else 1.0
            ns   = rng.uniform(0.001,0.005) if rng.random()>0.4 else 0.0
            all_features.append(compute_features(seq, tilt_angle=tilt,
                                                  scale_factor=sc,
                                                  do_flip=rng.random()>0.5,
                                                  noise_sigma=ns))

    X_new = np.array(all_features, dtype=np.float32)  # (N, 128, 69)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, X_new)
    print(f"[INFO] Saved {X_new.shape} → {out_path}")
    print(f"  ar stats: mean={X_new[:,:,-1].mean():.3f}  min={X_new[:,:,-1].min():.3f}  max={X_new[:,:,-1].max():.3f}")
    print(f"  vel stats (feat 35): mean={X_new[:,:,35].mean():.4f} std={X_new[:,:,35].std():.4f}")


if __name__ == "__main__":
    main()
