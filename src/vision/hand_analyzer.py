# src/vision/hand_analyzer.py
"""
Hand analysis module for contour, convex hull, and fingertip detection.

Uses classical computer vision on the binary mask from segmentation.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List


class HandAnalyzer:
    """
    Analyzes binary hand mask to extract contour, hull, and fingertip.
    
    Pipeline:
    1. Find all contours in binary mask
    2. Select largest contour (the hand)
    3. Compute convex hull
    4. Find fingertip (topmost point on hull)
    5. Fallback: if topmost is below centroid, use farthest from centroid
    
    Attributes:
        min_contour_area: Minimum contour area to consider as hand
        fallback_to_farthest: Use farthest point fallback if needed
    """
    
    def __init__(self,
                 min_contour_area: int = 2000,
                 fallback_to_centroid: bool = False):
        """
        Initialize hand analyzer.
        
        Args:
            min_contour_area: Minimum contour area threshold
            fallback_to_centroid: Use farthest point if topmost fails
        """
        self.min_contour_area = min_contour_area
        self.fallback_to_centroid = fallback_to_centroid
        
        # Fingertip smoothing buffer (temporal stability)
        from collections import deque
        self.fingertip_buffer = deque(maxlen=5)  # Last 5 fingertip positions
    
    def analyze(self, mask: np.ndarray) -> Dict:
        """
        Analyze binary mask to extract hand features.
        
        Args:
            mask: Binary mask (0 or 255) from segmentation model
        
        Returns:
            Dictionary with keys:
                - 'contour': Largest hand contour or None
                - 'hull': Convex hull of contour or None
                - 'fingertip': (x, y) tuple of fingertip or None
                - 'centroid': (x, y) tuple of hand centroid or None
                - 'area': Contour area or 0
        """
        result = {
            'contour': None,
            'hull': None,
            'fingertip': None,
            'centroid': None,
            'area': 0
        }
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return result
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Check minimum area
        if area < self.min_contour_area:
            return result
        
        result['contour'] = largest_contour
        result['area'] = area
        
        # Compute centroid
        centroid = self._compute_centroid(largest_contour)
        result['centroid'] = centroid
        
        # Smooth contour using polygon approximation before hull computation
        # Uses Douglas-Peucker algorithm to reduce vertices while preserving shape
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)  # 0.5% of perimeter
        smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Compute convex hull from smoothed contour (fewer vertices = more stable)
        hull = cv2.convexHull(smoothed_contour, returnPoints=True)
        result['hull'] = hull
        
        # Find fingertip
        if hull is not None and len(hull) > 0:
            fingertip = self._find_fingertip(hull, centroid)
            result['fingertip'] = fingertip
        
        return result
    
    def _compute_centroid(self, contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Compute centroid of contour using moments.
        
        Args:
            contour: OpenCV contour
        
        Returns:
            (x, y) tuple or None if moments are zero
        """
        moments = cv2.moments(contour)
        
        if moments['m00'] == 0:
            return None
        
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        return (cx, cy)
    
    def _find_fingertip(self,
                        hull: np.ndarray,
                        centroid: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Find fingertip from convex hull.
        
        Strategy:
        1. Find topmost point on hull (minimum y)
        2. If topmost is below centroid and fallback enabled,
           use farthest point from centroid instead
        
        Args:
            hull: Convex hull points
            centroid: Hand centroid (x, y)
        
        Returns:
            (x, y) tuple of fingertip or None
        """
        if hull is None or len(hull) == 0:
            return None
        
        # Reshape hull to list of points
        points = hull.reshape(-1, 2)
        
        if len(points) == 0:
            return None
        
        # Find topmost point (minimum y coordinate)
        topmost_idx = np.argmin(points[:, 1])
        raw_fingertip = tuple(points[topmost_idx])
        
        # Check if fallback is needed
        if self.fallback_to_centroid and centroid is not None:
            # If topmost point is below centroid (larger y), use farthest point
            if raw_fingertip[1] > centroid[1]:
                # Find farthest point from centroid
                distances = np.sqrt(
                    (points[:, 0] - centroid[0]) ** 2 +
                    (points[:, 1] - centroid[1]) ** 2
                )
                farthest_idx = np.argmax(distances)
                raw_fingertip = tuple(points[farthest_idx])
        
        # TEMPORAL SMOOTHING: Average last N fingertip positions
        self.fingertip_buffer.append(raw_fingertip)
        
        if len(self.fingertip_buffer) >= 3:  # Need at least 3 samples for stability
            # Compute median position (robust to outliers)
            x_coords = [pt[0] for pt in self.fingertip_buffer]
            y_coords = [pt[1] for pt in self.fingertip_buffer]
            smoothed_x = int(np.median(x_coords))
            smoothed_y = int(np.median(y_coords))
            return (smoothed_x, smoothed_y)
        else:
            # Insufficient history, use raw
            return raw_fingertip
    
    def get_hull_points(self, hull: np.ndarray) -> List[Tuple[int, int]]:
        """
        Convert hull to list of (x, y) tuples.
        
        Args:
            hull: OpenCV convex hull
        
        Returns:
            List of (x, y) point tuples
        """
        if hull is None:
            return []
        
        points = hull.reshape(-1, 2)
        return [tuple(p) for p in points]
    
    def get_bounding_box(self, contour: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box of contour.
        
        Args:
            contour: OpenCV contour
        
        Returns:
            (x, y, width, height) or None
        """
        if contour is None:
            return None
        
        return cv2.boundingRect(contour)
