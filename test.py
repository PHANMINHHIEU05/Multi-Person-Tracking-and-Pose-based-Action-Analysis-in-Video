"""
Test script cho Milestone 1: Video I/O
"""

import cv2
import sys
from pathlib import Path

# Thêm project root vào Python path
sys.path.append(str(Path(__file__).parent))

from utils.video_io import VideoReader, VideoWriter


def test_video_read():
    """Test đọc video và hiển thị 1 frame"""
    
    # TODO: Thay bằng đường dẫn video của bạn
    video_path = "data/videos/test_video.mp4"
    
    # Kiểm tra file tồn tại
    if not Path(video_path).exists():
        print(f"❌ Video không tồn tại: {video_path}")
        print("💡 Hãy tải video test từ Pexels hoặc dùng video có sẵn")
        return
    
    print("\n=== TEST 1: Đọc Video ===\n")
    
    # Sử dụng context manager
    with VideoReader(video_path, target_size=(1280, 720)) as reader:
        
        # Đọc frame đầu tiên
        frame = reader.read_frame()
        
        if frame is None:
            print("❌ Không đọc được frame")
            return
        
        print(f"\n✅ Frame shape: {frame.shape}")  # Phải (720, 1280, 3)
        print(f"   - Height: {frame.shape[0]}")
        print(f"   - Width: {frame.shape[1]}")
        print(f"   - Channels: {frame.shape[2]}")
        
        # Hiển thị frame (tắt để test tự động)
        # cv2.imshow("Test Frame - Press any key to close", frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Đọc thêm 9 frames để test
        print("\n📊 Testing reading 10 frames...")
        for i in range(2, 11):
            frame = reader.read_frame()
            if frame is None:
                print(f"   Frame {i}: End of video")
                break
            print(f"   Frame {i}: OK (shape {frame.shape})")


def test_video_write():
    """Test ghi video"""
    
    video_path = "data/videos/test_video.mp4"
    output_path = "output/test_output.mp4"
    
    # Tạo thư mục output nếu chưa có
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if not Path(video_path).exists():
        print(f"❌ Video không tồn tại: {video_path}")
        return
    
    print("\n=== TEST 2: Ghi Video ===\n")
    
    with VideoReader(video_path, target_size=(1280, 720)) as reader:
        with VideoWriter(output_path, reader.fps, (1280, 720)) as writer:
            
            # Đọc và ghi 100 frames đầu tiên
            frame_count = 0
            max_frames = 100
            
            while frame_count < max_frames:
                frame = reader.read_frame()
                if frame is None:
                    break
                
                # Vẽ text lên frame để test
                cv2.putText(
                    frame,
                    f"Frame: {frame_count}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                writer.write_frame(frame)
                frame_count += 1
                
                if frame_count % 10 == 0:
                    print(f"   Processed {frame_count} frames...")
            
            print(f"\n✅ Đã ghi {frame_count} frames")


def test_fps_calculation():
    """Test tính FPS thực tế khi xử lý"""
    
    video_path = "data/videos/test_video.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Video không tồn tại: {video_path}")
        return
    
    print("\n=== TEST 3: FPS Calculation ===\n")
    
    import time
    
    with VideoReader(video_path, target_size=(1280, 720)) as reader:
        
        frame_count = 0
        max_frames = 100
        start_time = time.time()
        
        while frame_count < max_frames:
            frame = reader.read_frame()
            if frame is None:
                break
            
            # Giả lập processing (chỉ resize, chưa có YOLO)
            frame_count += 1
        
        end_time = time.time()
        elapsed = end_time - start_time
        fps = frame_count / elapsed
        
        print(f"✅ Đọc {frame_count} frames trong {elapsed:.2f}s")
        print(f"   Processing FPS: {fps:.2f}")
        print(f"   Video FPS gốc: {reader.fps:.2f}")
        
        if fps < reader.fps:
            print(f"   ⚠️  Processing chậm hơn video gốc → Sẽ cần optimize hoặc frame skip")
        else:
            print(f"   ✅ Processing nhanh hơn video → Đủ real-time!")


if __name__ == "__main__":
    print("🚀 Starting M1 Tests...\n")
    
    # Chạy từng test
    test_video_read()
    test_video_write()
    test_fps_calculation()
    
    print("\n" + "="*50)
    print("✅ M1 Tests completed!")
    print("="*50)