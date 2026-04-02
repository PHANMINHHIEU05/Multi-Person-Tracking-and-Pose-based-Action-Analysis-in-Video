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
