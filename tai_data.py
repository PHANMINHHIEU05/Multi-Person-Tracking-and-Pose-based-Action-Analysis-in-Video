import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract():
    # 1. Khởi tạo API (Đảm bảo file kaggle.json đã ở đúng chỗ)
    api = KaggleApi()
    api.authenticate()

    # 2. Tên bộ dữ liệu 10 actions cực nhẹ
    dataset = "oussemakirmani/ntu-rgb-d-10-actions"
    zip_name = "ntu-rgb-d-10-actions.zip"
    output_dir = "ntu_10_actions_filtered"

    print(f"🚀 Đang kéo bộ 10 actions (~324MB) về máy...")
    api.dataset_download_files(dataset, path=".", unzip=False)

    # 3. Mã hành động mục tiêu (A001: Walk, A008: Sit, A043: Fall)
    # Vì bộ này chỉ có 10 actions, ta cứ giải nén ra xem có những gì
    print("📦 Đang giải nén...")
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_name, 'r') as z:
        z.extractall(output_dir)

    # 4. Dọn dẹp
    if os.path.exists(zip_name):
        os.remove(zip_name)
        
    print(f"\n🔥 Xong! Đại ca kiểm tra thư mục: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    download_and_extract()