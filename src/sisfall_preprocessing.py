"""
=============================================================================
SisFall Dataset Preprocessing Pipeline
=============================================================================
Mô tả: Script tiền xử lý dữ liệu SisFall cho bài toán Multi-class Action
        Recognition (Phát hiện ngã và phân loại hành động).

Tác giả: Auto-generated preprocessing pipeline
Ngày tạo: 2026-02-28

Bộ dữ liệu: SisFall - A Fall and Movement Dataset
    - 23 người trẻ (SA01-SA23) và 15 người già (SE01-SE15)
    - 19 hoạt động hàng ngày (D01-D19) và 15 kiểu ngã (F01-F15)
    - Cảm biến: ADXL345 (13-bit, ±16g), MMA8451Q, ITG3200

Pipeline bao gồm:
    1. Quét và gán nhãn 6 lớp (Fall, Walking, Stairs, ...)
    2. Đọc và chuyển đổi tín hiệu thô sang đơn vị g
    3. Cắt nhiễu đầu/cuối (100 dòng)
    4. Chia cửa sổ trượt (Sliding Window) với overlap 50%
    5. Xuất metadata, X_train.npy, y_train.npy
=============================================================================
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from pathlib import Path

# =============================================================================
# PHẦN 1: CẤU HÌNH (CONFIGURATION)
# =============================================================================

# --- Đường dẫn ---
BASE_DIR = Path(__file__).resolve().parent.parent          # Thư mục gốc dự án
DATA_DIR = BASE_DIR / "data" / "SisFall"                   # Thư mục chứa dataset
OUTPUT_DIR = BASE_DIR / "data" / "processed"                # Thư mục xuất kết quả

# --- Tham số cảm biến ADXL345 ---
ADXL345_RESOLUTION_BITS = 13   # Độ phân giải 13-bit
ADXL345_RANGE_G = 16           # Phạm vi đo ±16g
# Công thức: g = (RawData / 2^13) * 2 * 16 = RawData * 32 / 2^13
ADXL345_SCALE_FACTOR = (2 * ADXL345_RANGE_G) / (2 ** ADXL345_RESOLUTION_BITS)

# --- Tham số tiền xử lý ---
TRIM_HEAD = 100                # Số dòng cắt bỏ ở đầu file (loại nhiễu khởi tạo)
TRIM_TAIL = 100                # Số dòng cắt bỏ ở cuối file (loại nhiễu kết thúc)
WINDOW_SIZE = 256              # Kích thước cửa sổ (~3 giây ở tần số 200Hz)
OVERLAP_RATIO = 0.5            # Tỷ lệ chồng lấp giữa các cửa sổ
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))  # = 128 mẫu

# --- Bảng ánh xạ 6 nhãn hành động ---
# Mã trong dataset: D01-D19 (ADL) và F01-F15 (Fall)
LABEL_MAP: Dict[str, Tuple[str, int]] = {
    # Walking: Đi bộ
    "D01": ("Walking", 0),
    "D02": ("Walking", 0),
    # Stairs: Lên/xuống cầu thang
    "D03": ("Stairs", 1),
    "D04": ("Stairs", 1),
    # Sitting_Standing: Ngồi/đứng
    "D05": ("Sitting_Standing", 2),
    "D06": ("Sitting_Standing", 2),
    "D07": ("Sitting_Standing", 2),
    "D08": ("Sitting_Standing", 2),
    # Bending: Cúi người
    "D14": ("Bending", 3),
    # Jogging_Jumping: Chạy/nhảy
    "D15": ("Jogging_Jumping", 4),
    "D16": ("Jogging_Jumping", 4),
    "D17": ("Jogging_Jumping", 4),
}
# Tất cả mã F01-F15 đều là Fall (label_id = 5)
for i in range(1, 16):
    LABEL_MAP[f"F{i:02d}"] = ("Fall", 5)

# Bảng tên nhãn theo thứ tự label_id
LABEL_NAMES = {
    0: "Walking",
    1: "Stairs",
    2: "Sitting_Standing",
    3: "Bending",
    4: "Jogging_Jumping",
    5: "Fall",
}


# =============================================================================
# PHẦN 2: HÀM ĐỌC VÀ CHUYỂN ĐỔI TÍN HIỆU (DATA I/O & CONVERSION)
# =============================================================================

def read_raw_file(file_path: str) -> Optional[np.ndarray]:
    """
    Đọc file tín hiệu thô SisFall (.txt).

    Định dạng mỗi dòng: "val1,val2,...,val9;\r\n"
    - 9 cột, phân tách bằng dấu phẩy, kết thúc bằng dấu chấm phẩy.
    - 3 cột đầu: ADXL345 (gia tốc X, Y, Z) — cảm biến chính ta sử dụng.

    Args:
        file_path: Đường dẫn tuyệt đối tới file .txt

    Returns:
        np.ndarray shape (N, 3) chứa dữ liệu thô 3 trục, hoặc None nếu lỗi.
    """
    try:
        rows = []
        with open(file_path, "r") as f:
            for line in f:
                # Loại bỏ dấu chấm phẩy cuối dòng và khoảng trắng thừa
                line = line.strip().rstrip(";").strip()
                if not line:
                    continue
                # Tách các giá trị theo dấu phẩy, chỉ lấy 3 cột đầu (ADXL345)
                parts = line.split(",")
                if len(parts) >= 3:
                    rows.append([int(parts[0]), int(parts[1]), int(parts[2])])

        if len(rows) == 0:
            print(f"  [CẢNH BÁO] File rỗng: {file_path}")
            return None

        return np.array(rows, dtype=np.float64)

    except Exception as e:
        print(f"  [LỖI] Không thể đọc file {file_path}: {e}")
        return None


def convert_to_g(raw_data: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi dữ liệu thô ADXL345 sang đơn vị gia tốc trọng trường (g).

    Công thức: Acceleration(g) = RawData × (2 × Range) / (2^Resolution)
                                = RawData × (2 × 16) / (2^13)
                                = RawData × 32 / 8192
                                ≈ RawData × 0.00390625

    Args:
        raw_data: Mảng NumPy shape (N, 3) chứa dữ liệu thô (int).

    Returns:
        Mảng NumPy shape (N, 3) chứa dữ liệu đã chuyển đổi sang đơn vị g.
    """
    return raw_data * ADXL345_SCALE_FACTOR


# =============================================================================
# PHẦN 3: LÀM SẠCH DỮ LIỆU (DATA CLEANING)
# =============================================================================

def trim_signal(data: np.ndarray,
                head: int = TRIM_HEAD,
                tail: int = TRIM_TAIL) -> Optional[np.ndarray]:
    """
    Cắt bỏ các mẫu nhiễu ở đầu và cuối tín hiệu.

    Khi bắt đầu và kết thúc ghi dữ liệu, cảm biến thường tạo ra nhiễu
    do quá trình khởi tạo/tắt. Ta cắt bỏ `head` dòng đầu và `tail` dòng cuối.

    Args:
        data: Mảng NumPy shape (N, 3).
        head: Số dòng cắt bỏ ở đầu (mặc định: 100).
        tail: Số dòng cắt bỏ ở cuối (mặc định: 100).

    Returns:
        Mảng NumPy đã cắt, hoặc None nếu dữ liệu quá ngắn.
    """
    total_trim = head + tail
    if data.shape[0] <= total_trim:
        print(f"  [CẢNH BÁO] Dữ liệu quá ngắn ({data.shape[0]} dòng), "
              f"không đủ để cắt {total_trim} dòng → Bỏ qua file này.")
        return None

    return data[head: -tail] if tail > 0 else data[head:]


# =============================================================================
# PHẦN 4: KỸ THUẬT WINDOWING (SLIDING WINDOW)
# =============================================================================

def sliding_window(data: np.ndarray,
                   window_size: int = WINDOW_SIZE,
                   step_size: int = STEP_SIZE) -> np.ndarray:
    """
    Chia tín hiệu thành các cửa sổ trượt (Sliding Windows).

    Mục đích: Tạo nhiều mẫu huấn luyện từ một chuỗi tín hiệu dài.
    Mỗi cửa sổ có kích thước cố định (256 mẫu ≈ 3 giây).
    Các cửa sổ chồng lấp 50% để tăng số lượng mẫu.

    Args:
        data: Mảng NumPy shape (M, 3) — tín hiệu đã làm sạch.
        window_size: Số mẫu trong mỗi cửa sổ (mặc định: 256).
        step_size: Bước nhảy giữa các cửa sổ (mặc định: 128 = 50% overlap).

    Returns:
        Mảng NumPy shape (num_windows, window_size, 3).
        Nếu tín hiệu ngắn hơn window_size → trả về mảng rỗng shape (0, ws, 3).
    """
    num_samples = data.shape[0]
    windows = []

    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(data[start:end])

    if len(windows) == 0:
        return np.empty((0, window_size, 3))

    return np.array(windows)


# =============================================================================
# PHẦN 5: QUÉT DỮ LIỆU VÀ GÁN NHÃN (DATA SCANNING & LABELING)
# =============================================================================

def parse_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """
    Phân tích tên file để trích xuất thông tin.

    Tên file có dạng: ActionCode_SubjectCode_TrialCode.txt
    Ví dụ: D01_SA01_R01.txt → action='D01', subject='SA01', trial='R01'
            F05_SE03_R02.txt → action='F05', subject='SE03', trial='R02'

    Args:
        filename: Tên file (không bao gồm đường dẫn thư mục).

    Returns:
        Tuple (action_code, subject_code, trial_code) hoặc None nếu không hợp lệ.
    """
    match = re.match(r'^([A-Z]\d{2})_([A-Z]{2}\d{2})_(R\d{2})\.txt$', filename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def scan_dataset(data_dir: Path) -> pd.DataFrame:
    """
    Quét toàn bộ thư mục dataset, lọc file hợp lệ và gán nhãn 6 lớp.

    Logic gán nhãn:
        - F01-F15 → Fall (label_id=5)
        - D01,D02 → Walking (0); D03,D04 → Stairs (1)
        - D05-D08 → Sitting_Standing (2); D14 → Bending (3)
        - D15-D17 → Jogging_Jumping (4)
        - Các mã D khác (D09-D13, D18-D19) → Bỏ qua

    Args:
        data_dir: Đường dẫn tới thư mục gốc SisFall.

    Returns:
        DataFrame với các cột: file_path, action_code, subject_code,
        trial_code, label_name, label_id.
    """
    records = []
    skipped_codes = set()

    # Duyệt qua tất cả thư mục con (SA01-SA23, SE01-SE15)
    for subject_dir in sorted(data_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        # Duyệt qua tất cả file .txt trong thư mục subject
        for txt_file in sorted(subject_dir.glob("*.txt")):
            parsed = parse_filename(txt_file.name)
            if parsed is None:
                continue

            action_code, subject_code, trial_code = parsed

            # Kiểm tra xem mã hành động có nằm trong bảng ánh xạ không
            if action_code in LABEL_MAP:
                label_name, label_id = LABEL_MAP[action_code]
                records.append({
                    "file_path": str(txt_file),
                    "action_code": action_code,
                    "subject_code": subject_code,
                    "trial_code": trial_code,
                    "label_name": label_name,
                    "label_id": label_id,
                })
            else:
                skipped_codes.add(action_code)

    # Thông báo các mã bị bỏ qua
    if skipped_codes:
        print(f"[INFO] Các mã hành động bị bỏ qua (không thuộc 6 lớp): "
              f"{sorted(skipped_codes)}")

    df = pd.DataFrame(records)
    print(f"[INFO] Tổng số file hợp lệ: {len(df)}")
    return df


# =============================================================================
# PHẦN 6: TRỰC QUAN HÓA TÍN HIỆU (SIGNAL VISUALIZATION)
# =============================================================================

def plot_signal(file_path: str,
                title: Optional[str] = None,
                save_path: Optional[str] = None) -> None:
    """
    Vẽ biểu đồ 3 trục gia tốc (X, Y, Z) sau khi chuyển đổi sang đơn vị g.

    Hàm này dùng để kiểm tra trực quan chất lượng dữ liệu:
    - Tín hiệu ADL bình thường sẽ dao động nhẹ quanh ±1g.
    - Tín hiệu Fall sẽ có đỉnh đột biến (spike) lên đến ±8g hoặc hơn.

    Args:
        file_path: Đường dẫn tới file .txt.
        title: Tiêu đề biểu đồ (nếu None, sẽ dùng tên file).
        save_path: Nếu cung cấp, lưu biểu đồ vào file thay vì hiển thị.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[LỖI] Cần cài đặt matplotlib: pip install matplotlib")
        return

    # Đọc và chuyển đổi
    raw = read_raw_file(file_path)
    if raw is None:
        return

    data_g = convert_to_g(raw)

    # Tạo trục thời gian (giây), tần số lấy mẫu ADXL345 = 200Hz
    time_axis = np.arange(data_g.shape[0]) / 200.0

    # Vẽ biểu đồ
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axis_labels = ["Trục X (g)", "Trục Y (g)", "Trục Z (g)"]
    colors = ["#2196F3", "#4CAF50", "#FF5722"]  # Xanh dương, Xanh lá, Đỏ cam

    for i, (ax, label, color) in enumerate(zip(axes, axis_labels, colors)):
        ax.plot(time_axis, data_g[:, i], color=color, linewidth=0.6, alpha=0.85)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    axes[-1].set_xlabel("Thời gian (giây)", fontsize=11)

    # Tiêu đề
    if title is None:
        title = f"Tín hiệu gia tốc ADXL345 — {Path(file_path).name}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Đã lưu biểu đồ: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# =============================================================================
# PHẦN 7: PIPELINE XỬ LÝ CHÍNH (MAIN PROCESSING PIPELINE)
# =============================================================================

def process_single_file(file_path: str) -> Optional[Tuple[np.ndarray, int]]:
    """
    Xử lý một file dữ liệu: Đọc → Chuyển đổi → Cắt nhiễu → Windowing.

    Đây là hàm xử lý cốt lõi cho mỗi file, được gọi lặp lại bởi pipeline.

    Args:
        file_path: Đường dẫn file .txt.

    Returns:
        Tuple (windows, num_windows) hoặc None nếu file bị bỏ qua.
        windows: np.ndarray shape (num_windows, WINDOW_SIZE, 3)
    """
    # Bước 1: Đọc dữ liệu thô
    raw_data = read_raw_file(file_path)
    if raw_data is None:
        return None

    # Bước 2: Chuyển đổi sang đơn vị g
    data_g = convert_to_g(raw_data)

    # Bước 3: Cắt nhiễu đầu/cuối
    data_trimmed = trim_signal(data_g)
    if data_trimmed is None:
        return None

    # Bước 4: Chia cửa sổ trượt
    windows = sliding_window(data_trimmed)
    if windows.shape[0] == 0:
        print(f"  [CẢNH BÁO] Không đủ dữ liệu để tạo cửa sổ: {file_path} "
              f"(còn {data_trimmed.shape[0]} mẫu sau cắt, cần ≥ {WINDOW_SIZE})")
        return None

    return windows


def run_pipeline(data_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None) -> None:
    """
    Chạy toàn bộ Pipeline tiền xử lý dữ liệu SisFall.

    Luồng xử lý:
        1. Quét dataset → tạo metadata
        2. Với mỗi file: đọc → chuyển đổi → cắt → windowing
        3. Ghép tất cả windows và nhãn
        4. Lưu X_train.npy, y_train.npy, metadata_processed.csv
        5. In thống kê phân bố nhãn

    Args:
        data_dir: Thư mục chứa SisFall dataset (mặc định: DATA_DIR).
        output_dir: Thư mục xuất kết quả (mặc định: OUTPUT_DIR).
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print("=" * 70)
    print("  SisFall Preprocessing Pipeline — Multi-class Action Recognition")
    print("=" * 70)

    # --- Bước 1: Quét và gán nhãn ---
    print("\n[BƯỚC 1/5] Quét dataset và gán nhãn 6 lớp...")
    metadata_df = scan_dataset(data_dir)

    if metadata_df.empty:
        print("[LỖI] Không tìm thấy file nào! Kiểm tra lại đường dẫn dataset.")
        return

    # --- Bước 2: Tạo thư mục output ---
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Thư mục đầu ra: {output_dir}")

    # --- Bước 3: Xử lý từng file ---
    print(f"\n[BƯỚC 2/5] Xử lý tín hiệu ({len(metadata_df)} files)...")
    print(f"  Cấu hình: TRIM={TRIM_HEAD}/{TRIM_TAIL} | "
          f"WINDOW={WINDOW_SIZE} | STEP={STEP_SIZE} (overlap {OVERLAP_RATIO*100:.0f}%)")

    all_windows: List[np.ndarray] = []
    all_labels: List[int] = []
    metadata_records: List[Dict] = []

    processed_count = 0
    skipped_count = 0

    for idx, row in metadata_df.iterrows():
        file_path = row["file_path"]
        label_id = row["label_id"]

        result = process_single_file(file_path)

        if result is None:
            skipped_count += 1
            continue

        windows = result
        num_windows = windows.shape[0]

        # Lưu kết quả
        all_windows.append(windows)
        all_labels.extend([label_id] * num_windows)

        metadata_records.append({
            "subject_id": row["subject_code"],
            "action_code": row["action_code"],
            "trial_code": row["trial_code"],
            "label_id": label_id,
            "label_name": row["label_name"],
            "num_windows": num_windows,
            "file_path": file_path,
        })

        processed_count += 1

        # Hiển thị tiến trình mỗi 200 file
        if processed_count % 200 == 0:
            print(f"  → Đã xử lý: {processed_count}/{len(metadata_df)} files...")

    print(f"  → Hoàn tất: {processed_count} files xử lý thành công, "
          f"{skipped_count} files bị bỏ qua.")

    if len(all_windows) == 0:
        print("[LỖI] Không có dữ liệu nào sau khi xử lý!")
        return

    # --- Bước 4: Ghép và lưu dữ liệu ---
    print("\n[BƯỚC 3/5] Ghép dữ liệu và lưu file NumPy...")

    X_train = np.concatenate(all_windows, axis=0)  # (total_windows, 256, 3)
    y_train = np.array(all_labels, dtype=np.int64)  # (total_windows,)

    print(f"  X_train shape: {X_train.shape}  (samples, timesteps, channels)")
    print(f"  y_train shape: {y_train.shape}  (samples,)")
    print(f"  Kích thước bộ nhớ: X={X_train.nbytes / 1e6:.1f} MB, "
          f"y={y_train.nbytes / 1e3:.1f} KB")

    x_path = output_dir / "X_train.npy"
    y_path = output_dir / "y_train.npy"
    np.save(x_path, X_train)
    np.save(y_path, y_train)
    print(f"  → Đã lưu: {x_path}")
    print(f"  → Đã lưu: {y_path}")

    # --- Bước 5: Lưu metadata ---
    print("\n[BƯỚC 4/5] Lưu metadata_processed.csv...")

    meta_df = pd.DataFrame(metadata_records)
    meta_path = output_dir / "metadata_processed.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"  → Đã lưu: {meta_path}")

    # --- Bước 6: Thống kê phân bố nhãn ---
    print("\n[BƯỚC 5/5] Thống kê phân bố nhãn:")
    print("-" * 55)
    print(f"  {'Label ID':<10} {'Tên nhãn':<20} {'Số mẫu':<10} {'Tỷ lệ (%)':<10}")
    print("-" * 55)

    total_samples = len(y_train)
    for label_id in sorted(LABEL_NAMES.keys()):
        count = int(np.sum(y_train == label_id))
        ratio = count / total_samples * 100
        name = LABEL_NAMES[label_id]
        print(f"  {label_id:<10} {name:<20} {count:<10} {ratio:<10.2f}")

    print("-" * 55)
    print(f"  {'TỔNG':<10} {'':<20} {total_samples:<10} {'100.00':<10}")
    print("-" * 55)

    # Cảnh báo mất cân bằng
    counts = [int(np.sum(y_train == lid)) for lid in LABEL_NAMES]
    max_count, min_count = max(counts), min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    if imbalance_ratio > 5:
        print(f"\n  ⚠ CẢNH BÁO: Dữ liệu mất cân bằng nghiêm trọng! "
              f"(tỷ lệ max/min = {imbalance_ratio:.1f}x)")
        print(f"  → Gợi ý: Sử dụng class_weight, oversampling (SMOTE), "
              f"hoặc focal loss khi huấn luyện.")
    elif imbalance_ratio > 2:
        print(f"\n  ⚠ LƯU Ý: Dữ liệu hơi mất cân bằng "
              f"(tỷ lệ max/min = {imbalance_ratio:.1f}x)")

    print("\n" + "=" * 70)
    print("  Pipeline hoàn tất! Dữ liệu sẵn sàng cho huấn luyện model.")
    print("=" * 70)


# =============================================================================
# PHẦN 8: TIỆN ÍCH BỔ SUNG (UTILITIES)
# =============================================================================

def load_processed_data(output_dir: Optional[Path] = None
                        ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Tải dữ liệu đã tiền xử lý từ thư mục output.

    Hàm tiện ích để sử dụng trong notebook hoặc script huấn luyện.

    Usage:
        from src.sisfall_preprocessing import load_processed_data
        X, y, meta = load_processed_data()

    Returns:
        Tuple (X_train, y_train, metadata_df)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    X = np.load(output_dir / "X_train.npy")
    y = np.load(output_dir / "y_train.npy")
    meta = pd.read_csv(output_dir / "metadata_processed.csv")

    print(f"[INFO] Đã tải dữ liệu: X={X.shape}, y={y.shape}, "
          f"metadata={meta.shape[0]} records")
    return X, y, meta


def get_label_name(label_id: int) -> str:
    """Trả về tên nhãn từ label_id."""
    return LABEL_NAMES.get(label_id, f"Unknown({label_id})")


# =============================================================================
# PHẦN 9: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Chạy trực tiếp script:
        python -m src.sisfall_preprocessing
    hoặc:
        python src/sisfall_preprocessing.py
    """
    run_pipeline()
