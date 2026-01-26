# Multi-Person-Tracking-and-Pose-based-Action-Analysis-in-Video

Giai đoạn 1: Detect & Track (Nhìn và Nhớ)

    Công nghệ: YOLOv8 (Detector) + ByteTrack (Tracker).

    Thuật toán: Kalman Filter (Dự đoán vị trí) & Hungarian Algorithm (Khớp ID).

    Kết quả: Mỗi người trong video được bao quanh bởi một Bounding Box và có một con số ID duy nhất không đổi suốt video.

Giai đoạn 2: Pose Extraction (Trích xuất khung xương)

    Công nghệ: MediaPipe Pose.

    Cách làm: Với mỗi ID người, hệ thống crop ảnh của riêng người đó và chạy model MediaPipe để lấy ra 33 điểm khớp xương (Keypoints) dưới dạng tọa độ (x,y,z).

    Kết quả: Một bộ khung xương kỹ thuật số mô phỏng tư thế của từng ID.

Giai đoạn 3: Action Classification (Hiểu hành vi)

    Công nghệ: Machine Learning (Random Forest hoặc SVM).

    Thuật toán: Feature Engineering (Tính góc khớp xương, vận tốc di chuyển, tỉ lệ chiều cao cơ thể).

    Kết quả: Chuyển đổi tọa độ khớp xương thành nhãn hành động: "Standing", "Walking", "Sitting".

Giai đoạn 4: Data Analytics (Thống kê & Dashboard)

    Công nghệ: Python (Pandas), OpenCV (Overlay).

    Kết quả: Vẽ nhãn hành động đè lên video thời gian thực và xuất file báo cáo cuối cùng (ví dụ: ID_01 đã ngồi 70% thời gian).
