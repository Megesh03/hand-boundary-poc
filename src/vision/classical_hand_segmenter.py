# src/vision/classical_hand_segmenter.py
"""
Classical computer vision based hand segmentation.

Uses YCrCb color space for skin detection, motion differencing,
morphological operations, and contour filtering for robust hand detection.

This serves as a reliable fallback when ML segmentation fails or is disabled.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class ClassicalHandSegmenter:
    """
    Classical CV hand segmentation using YCrCb skin detection.
    
    Pipeline:
    1. Convert frame to YCrCb color space
    2. Apply skin color thresholding
    3. Motion differencing (if previous frame available)
    4. Morphological cleanup
    5. Contour filtering
    6. Return binary mask
    
    Attributes:
        use_motion: Enable motion differencing
        prev_gray: Previous frame for motion detection
    """
    
    def __init__(self,
                 skin_cr_min: int = 133,
                 skin_cr_max: int = 173,
                 skin_cb_min: int = 77,
                 skin_cb_max: int = 127,
                 motion_threshold: int = 20,
                 min_contour_area: int = 2000,
                 use_motion: bool = True):
        """
        Initialize classical hand segmenter.
        
        Args:
            skin_cr_min: Minimum Cr value for skin (133 typical)
            skin_cr_max: Maximum Cr value for skin (173 typical)
            skin_cb_min: Minimum Cb value for skin (77 typical)
            skin_cb_max: Maximum Cb value for skin (127 typical)
            motion_threshold: Frame difference threshold for motion
            min_contour_area: Minimum contour area to keep
            use_motion: Enable motion differencing
        """
        self.skin_cr_min = skin_cr_min
        self.skin_cr_max = skin_cr_max
        self.skin_cb_min = skin_cb_min
        self.skin_cb_max = skin_cb_max
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.use_motion = use_motion
        
        # State for motion detection
        self.prev_gray = None
        
        # Morphological kernels
        self.morph_kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.morph_kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    def segment(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment hand from frame using classical CV.
        
        Args:
            frame: BGR frame from camera
        
        Returns:
            Binary mask (0 or 255) with hand region
        """
        h, w = frame.shape[:2]
        
        # Convert to YCrCb for skin detection
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Split channels
        y, cr, cb = cv2.split(ycrcb)
        
        # Apply skin color thresholds
        skin_mask = cv2.inRange(
            ycrcb,
            np.array([0, self.skin_cr_min, self.skin_cb_min], dtype=np.uint8),
            np.array([255, self.skin_cr_max, self.skin_cb_max], dtype=np.uint8)
        )
        
        # Motion differencing (if enabled and previous frame exists)
        if self.use_motion and self.prev_gray is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Frame difference
            diff = cv2.absdiff(gray, self.prev_gray)
            _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Dilate motion mask to connect nearby regions
            motion_mask = cv2.dilate(motion_mask, self.morph_kernel_small, iterations=2)
            
            # Combine skin and motion
            combined_mask = cv2.bitwise_and(skin_mask, motion_mask)
            
            # Update previous frame
            self.prev_gray = gray
        else:
            # First frame or motion disabled - use skin only
            combined_mask = skin_mask
            
            # Initialize previous frame
            if self.use_motion:
                self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Morphological cleanup
        # 1. Open to remove noise
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, 
                                self.morph_kernel_small, iterations=1)
        
        # 2. Close to fill gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, 
                               self.morph_kernel_large, iterations=2)
        
        # 3. Median blur for smoothing
        mask = cv2.medianBlur(mask, 5)
        
        # Contour filtering - keep only large regions (likely hands)
        mask = self._filter_contours(mask)
        
        return mask
    
    def _filter_contours(self, mask: np.ndarray) -> np.ndarray:
        """
        Filter contours by area and select best hand candidate.
        
        Args:
            mask: Binary mask
        
        Returns:
            Filtered binary mask with only valid hand regions
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(mask)
        
        # Filter by minimum area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        
        if not valid_contours:
            return np.zeros_like(mask)
        
        # Create new mask with only valid contours
        filtered_mask = np.zeros_like(mask)
        
        # Draw the largest contour (most likely the hand)
        largest_contour = max(valid_contours, key=cv2.contourArea)
        cv2.drawContours(filtered_mask, [largest_contour], -1, 255, -1)
        
        return filtered_mask
    
    def reset(self):
        """Reset motion tracking state."""
        self.prev_gray = None
    
    def calibrate_skin_color(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> None:
        """
        Calibrate skin color thresholds from a region of interest.
        
        Args:
            frame: BGR frame
            roi: (x, y, w, h) region containing skin
        """
        x, y, w, h = roi
        skin_sample = frame[y:y+h, x:x+w]
        
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(skin_sample, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)
        
        # Calculate mean and std
        cr_mean, cr_std = cv2.meanStdDev(cr)
        cb_mean, cb_std = cv2.meanStdDev(cb)
        
        # Set thresholds as mean ± 2*std
        self.skin_cr_min = max(0, int(cr_mean[0] - 2 * cr_std[0]))
        self.skin_cr_max = min(255, int(cr_mean[0] + 2 * cr_std[0]))
        self.skin_cb_min = max(0, int(cb_mean[0] - 2 * cb_std[0]))
        self.skin_cb_max = min(255, int(cb_mean[0] + 2 * cb_std[0]))
        
        print(f"Calibrated skin color:")
        print(f"  Cr range: [{self.skin_cr_min}, {self.skin_cr_max}]")
        print(f"  Cb range: [{self.skin_cb_min}, {self.skin_cb_max}]")
