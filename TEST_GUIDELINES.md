cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .trt-export-venv/bin/activate
python pyqt_app.py

# Fallback runtime without TensorRT

cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .venv/bin/activate
python pyqt_app.py

# Optional TensorRT export on an NVIDIA RTX machine

# Uses the dedicated Python 3.12 TensorRT export environment created for this project.

cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .trt-export-venv/bin/activate
python export_pose_engine.py

# Rebuild the clean 5-class action dataset

cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .venv/bin/activate
python repair_action_dataset.py \
  --mode five_action_round4 \
  --base_dir data/train_ready_action_repair_v2_unicomfacauca \
  --sit_external_dir data/train_ready_action_repair_v4_sit_only \
  --out_dir data/train_ready_action_repair_v6_five_action_round4

# Build the focused hardcase variant for partial-body walking without reintroducing Bending

cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .venv/bin/activate
python prepare_action_hardcases_round2.py \
  --base_dir data/train_ready_action_repair_v6_five_action_round4 \
  --out_dir data/train_ready_action_repair_v6_five_action_round4_hardcases \
  --walk_partial_body_copies 120 \
  --bending_boundary_copies 0

# Retrain the current active ExtraTrees action model

cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .venv/bin/activate
python train_extratrees_action.py \
  --data_dir data/train_ready_action_repair_v6_five_action_round4_hardcases \
  --out_dir runs/train_extratrees_action_repair_v6_five_action_round4_hardcases_v1spec \
  --feature_spec mean,std,min,max,first,last,delta,q25,q75,abs_vel_mean,vel_std

# Active runtime note

# The current runtime pointer now stays on:
# runs/train_extratrees_action_repair_v2_unicomfacauca_v1spec/extratrees_model.joblib
# internal labels:
# Fall | Standing | Walking | Sitting_Quickly | Bending | Lying_Down
# UI display labels:
# Fall | Standing | Walking | Sitting | Lying_Down
# because the internal Bending class is currently used as an ambiguity buffer
# for occluded or transitional poses in real-video runtime tests.
