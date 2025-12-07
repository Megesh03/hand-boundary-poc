# src/camera/capture.py
"""
Webcam capture wrapper with configuration support.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class CameraCapture:
    """
    OpenCV VideoCapture wrapper with automatic configuration.
    
    Handles camera initialization, frame capture, and cleanup.
    
    Attributes:
        camera_index: Camera device index (default 0)
        width: Desired frame width
        height: Desired frame height
        cap: OpenCV VideoCapture object
    """
    
    def __init__(self,
                 camera_index: int = 0,
                 width: int = 640,
                 height: int = 480,
                 fps_target: int = 30):
        """
        Initialize camera capture.
        
        Args:
            camera_index: Camera device index
            width: Desired frame width
            height: Desired frame height
            fps_target: Target FPS for camera
        """
        self.device_index = camera_index
        self.width = width
        self.height = height
        self.fps_target = fps_target
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_open = False
    
    def open(self):
        """
        Open the webcam with Windows backend fallbacks.
        
        Tries DirectShow first (most stable on Windows), then MSMF, then default.
        
        Raises:
            RuntimeError: If camera cannot be opened with any backend
        """
        # Try DirectShow first (most stable on Windows)
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        
        # If DirectShow fails, try MSMF
        if not self.cap.isOpened():
            print("[WARN] DirectShow failed, trying MSMF backend...")
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_MSMF)
        
        # If MSMF fails, fall back to default OpenCV behavior
        if not self.cap.isOpened():
            print("[WARN] MSMF failed, trying default OpenCV camera backend...")
            self.cap = cv2.VideoCapture(self.device_index)
        
        # Final check
        if not self.cap.isOpened():
            raise RuntimeError("❌ Could not open camera using any backend")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        
        print(f"Camera opened: {self.width}x{self.height} @ {self.fps_target} FPS")
        
        self._is_open = True
    
    def read(self) -> np.ndarray:
        """
        Read a frame from the camera; retry a few times if needed.
        
        Returns:
            Frame as numpy array
            
        Raises:
            RuntimeError: If frame cannot be read after multiple attempts
        """
        if self.cap is None or not self._is_open:
            raise RuntimeError("❌ Camera not opened")
        
        for _ in range(3):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
        
        raise RuntimeError("❌ Camera error: Could not read frame after multiple attempts")
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self._is_open = False
            print("Camera released")
    
    def is_open(self) -> bool:
        """Check if camera is open."""
        return self._is_open and self.cap is not None and self.cap.isOpened()
    
    def get_frame_size(self) -> Tuple[int, int]:
        """Get current frame size (width, height)."""
        if self.cap is not None and self._is_open:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return self.width, self.height
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
