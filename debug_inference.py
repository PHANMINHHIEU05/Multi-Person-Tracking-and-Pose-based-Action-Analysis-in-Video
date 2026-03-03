"""Debug script: so sánh feature distribution giữa training data và video1.mp4"""
import sys; sys.path.insert(0,'.')
import cv2, numpy as np
from ultralytics import YOLO
from scipy.ndimage import uniform_filter1d

model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture('data/video/video1.mp4')
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {W}x{H}")

# Lấy 128+ frames liên tiếp từ frame 50
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
buf = []
for fi in range(180):
    ret, frame = cap.read()
    if not ret: break
    res = model(frame, conf=0.25, classes=[0], verbose=False)
    r = res[0]
    if r.keypoints is not None and len(r.keypoints.xy) > 0:
        kp = r.keypoints.xy[0].cpu().numpy().copy()
        kp[:,0] /= W; kp[:,1] /= H
        buf.append(kp)

cap.release()
print(f"Buffer: {len(buf)} frames")

if len(buf) < 128:
    print("Not enough frames!"); sys.exit()

seq = np.array(buf[-128:], dtype=np.float32)  # (128,17,2)

# aspect ratio
xs = seq[:,:,0]; ys = seq[:,:,1]
w = np.max(xs,axis=1) - np.min(xs,axis=1)
h = np.max(ys,axis=1) - np.min(ys,axis=1)
ar = (w / np.maximum(h,1e-6))[:,np.newaxis]
print(f"\nAspect ratio: mean={ar.mean():.3f} min={ar.min():.3f} max={ar.max():.3f}")

# Hip positions before normalize
hip_raw = (seq[:,11,:] + seq[:,12,:]) / 2.0
print(f"Hip Y raw: mean={hip_raw[:,1].mean():.3f} min={hip_raw[:,1].min():.3f} max={hip_raw[:,1].max():.3f}")

# MA smoothing
seq_flat = seq.reshape(128,-1)
seq_flat = uniform_filter1d(seq_flat, size=5, axis=0)
seq = seq_flat.reshape(128,17,2)

# Hip-centered normalize
hip_mid = (seq[:,11,:] + seq[:,12,:]) / 2.0
nose = seq[:,0,:]; ankle = (seq[:,15,:]+seq[:,16,:])/2.0
skel_h = np.linalg.norm(nose - ankle, axis=1, keepdims=True)
print(f"\nSkeleton height (raw norm): mean={skel_h.mean():.3f} min={skel_h.min():.3f} max={skel_h.max():.3f}")
skel_h = np.where(skel_h < 0.05, 0.3, skel_h)
seq = seq - hip_mid[:,np.newaxis,:]
seq = seq / skel_h[:,np.newaxis,:]
print(f"After hip-center: range [{seq.min():.3f}, {seq.max():.3f}]")

# Velocity / Acceleration
seq_f = seq.reshape(128,-1)
vel = np.diff(seq_f, n=1, axis=0, prepend=seq_f[:1])
acc = np.diff(vel,   n=1, axis=0, prepend=vel[:1])
feat = np.concatenate([seq_f, vel, acc, ar], axis=1)

print(f"\nFeatures: {feat.shape}")
print(f"  xy  stats: mean={feat[:,:34].mean():.4f} std={feat[:,:34].std():.4f}")
print(f"  vel stats: mean={feat[:,34:68].mean():.4f} std={feat[:,34:68].std():.4f}  max={np.abs(feat[:,34:68]).max():.4f}")
print(f"  acc stats: mean={feat[:,68:].mean():.4f} std={feat[:,68:].std():.4f}  max={np.abs(feat[:,68:]).max():.4f}")

# === Training data ===
print("\n=== Training data ===")
X = np.load('data/train_ready_horizontal/X_train.npy')
y = np.load('data/train_ready_horizontal/y_train.npy')
print(f"Shape: {X.shape}  label dist: {dict(zip(*np.unique(y,return_counts=True)))}")
for cls in np.unique(y):
    idx = y == cls
    print(f"  Class {cls}: vel_std={X[idx,:,34:68].std():.4f} ar_mean={X[idx,:,-1].mean():.3f}")
print(f"Overall vel max={np.abs(X[:,:,34:68]).max():.4f}")
print(f"Overall acc max={np.abs(X[:,:,68:]).max():.4f}")

# So sánh vel distribution chi tiết
print("\n=== Velocity comparison ===")
print(f"Video1 vel std: {feat[:,34:68].std():.4f}")
for cls in np.unique(y):
    idx = y==cls
    print(f"  Train class {cls} vel std: {X[idx,:,34:68].std():.4f}")
