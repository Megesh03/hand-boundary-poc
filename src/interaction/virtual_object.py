# src/interaction/virtual_object.py
"""
Virtual object (rectangle) for boundary detection.
"""

import numpy as np
from typing import Tuple


class VirtualRectangle:
    """
    Virtual rectangle boundary for hand interaction.
    
    Represents a rectangular region on screen that the hand
    should not touch (boundary violation detection).
    
    Attributes:
        x: Top-left x coordinate
        y: Top-left y coordinate
        width: Rectangle width
        height: Rectangle height
    """
    
    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Initialize virtual rectangle.
        
        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Rectangle width
            height: Rectangle height
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """Get rectangle as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of rectangle."""
        cx = self.x + self.width // 2
        cy = self.y + self.height // 2
        return (cx, cy)
    
    @property
    def corners(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get top-left and bottom-right corners."""
        top_left = (self.x, self.y)
        bottom_right = (self.x + self.width, self.y + self.height)
        return (top_left, bottom_right)
    
    def distance_to_point(self, point: Tuple[int, int]) -> float:
        """
        Calculate minimum distance from a point to the rectangle edge.
        
        Uses proper edge distance calculation:
        - If point is inside rectangle: distance = 0
        - If point is outside: distance to nearest edge
        
        Algorithm:
        1. Clamp point to rectangle bounds
        2. Calculate Euclidean distance from point to clamped point
        
        Args:
            point: (x, y) coordinates of the point
        
        Returns:
            Minimum distance to rectangle edge (0 if inside)
        """
        px, py = point
        
        # Clamp point to rectangle bounds
        closest_x = max(self.x, min(px, self.x + self.width))
        closest_y = max(self.y, min(py, self.y + self.height))
        
        # Calculate distance to closest point on rectangle
        dx = px - closest_x
        dy = py - closest_y
        distance = np.sqrt(dx * dx + dy * dy)
        
        return float(distance)
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """
        Check if point is inside rectangle.
        
        Args:
            point: (x, y) coordinates
        
        Returns:
            True if point is inside or on edge
        """
        px, py = point
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)
    
    def move_to(self, x: int, y: int):
        """
        Move rectangle to new position.
        
        Args:
            x: New top-left x coordinate
            y: New top-left y coordinate
        """
        self.x = x
        self.y = y
    
    def resize(self, width: int, height: int):
        """
        Resize rectangle.
        
        Args:
            width: New width
            height: New height
        """
        self.width = width
        self.height = height
