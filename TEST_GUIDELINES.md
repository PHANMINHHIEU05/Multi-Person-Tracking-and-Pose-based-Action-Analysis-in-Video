# TEST GUIDELINES

## 1) Chạy app PyQt6

### Runtime chính (khuyên dùng)
```bash
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .trt-export-venv/bin/activate
python pyqt_app.py
```

### Fallback runtime (không dùng env TensorRT)
```bash
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .venv/bin/activate
python pyqt_app.py
```

## 2) Profile hiện có trong UI

- `Fast Mode`
- `RTX 3050 Balanced`
- `Quality Mode`

Ghi chú: runtime hiện đã rollback về logic legacy (không dùng clean 5-class toggle).

## 3) Headless benchmark (không mở UI)

```bash
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .trt-export-venv/bin/activate
python run_headless_profile_benchmark.py \
  --video "VideoTest/Human Fall Detection Sample.mp4" \
  --profiles fast,balanced,quality
```

Output tổng hợp sẽ lưu ở: `runs/qt_outputs/profile_benchmark_headless_*.json`.

## 4) Kiểm tra model action đang active

```bash
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
cat runs/active_action_model_path.txt
```

Đổi model active:
```bash
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
echo "/absolute/path/to/your_action_model.pth" > runs/active_action_model_path.txt
```

## 5) Optional: export TensorRT pose engine

```bash
cd /home/hiubeo/Documents/code/Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video
source .trt-export-venv/bin/activate
python export_pose_engine.py
```
