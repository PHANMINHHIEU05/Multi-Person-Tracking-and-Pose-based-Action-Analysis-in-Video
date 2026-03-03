import kagglehub
import os

# Cấu hình tài khoản của đại ca
os.environ["KAGGLE_USERNAME"] = "hiubeo05"
os.environ["KAGGLE_KEY"] = "06d6fa628e8775d77c72709529a11aaf"

datasets = {
    "UR_Fall": "shahliza27/ur-fall-detection-dataset",
    "Multicam": "soumicksarker/multiple-cameras-fall-dataset"
}

for folder, slug in datasets.items():
    print(f"--- Đang kéo bộ {folder} ---")
    try:
        path = kagglehub.dataset_download(slug)
        print(f"[OK] Đã tải {folder} về: {path}")
        
        # Tạo liên kết mềm ra thư mục dự án cho dễ dùng
        os.system(f"mkdir -p ./data/{folder}")
        os.system(f"ln -s {path}/* ./data/{folder}/")
    except Exception as e:
        print(f"[LỖI] Không tải được {folder}: {e}")

print("\n--- TẤT CẢ ĐÃ SẴN SÀNG ĐỂ TRÍCH XUẤT POSE! ---")