import cv2 
import numpy as np
from typing import List, Tuple , Optional

class VideoReader:
    """hỗ trợ dọc video và resize về target resolution"""
    def __init__(self , video_path: str, target_size: Tuple[int , int ] = (1280 , 720)):
        self.video_path = video_path
        self.target_size = target_size
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Không thể mở video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video mở thành công: {video_path}, Kích thước gốc: {self.width}x{self.height}, FPS: {self.fps}, Tổng số khung hình: {self.frame_count}")
    def read_frame(self) -> Optional[np.ndarray]:
        """đọc 1 frame và resize 
        return về frame đã resize hoặc none nếu hết video"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        return frame_resized
    def release(self):
        """giải phóng tài nguyên"""
        self.cap.release()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
class VideoWriter:
    """ghi video output với bbox và tracking infor"""
    def __init__(self , output_path: str, fps : float, frame_size: Tuple[int , int ]):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, frame_size
        )
        if not self.writer.isOpened():
            raise ValueError(f"Không thể mở VideoWriter: {output_path}")
        print(f"VideoWriter mở thành công: {output_path}, Kích thước: {frame_size}, FPS: {fps}")        
    def write_frame(self, frame: np.ndarray):
        """ghi 1 frame vào video"""
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            raise ValueError(f"Kích thước frame không khớp: {frame.shape[1]}x{frame.shape[0]} != {self.frame_size}")
        self.writer.write(frame)
    def release(self):
        """giải phóng tài nguyên"""
        self.writer.release()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        