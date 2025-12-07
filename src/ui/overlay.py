# src/ui/overlay.py
"""
UI overlay rendering for hand tracking visualization.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from interaction.state_machine import State


class OverlayRenderer:
    """
    Renders UI overlays for hand tracking visualization.
    
    Draws:
    - Virtual boundary rectangle (colored by state)
    - Hand contour (magenta)
    - Convex hull (cyan)
    - Fingertip marker (yellow circle with outline)
    - State banner with distance
    - "DANGER DANGER" overlay when in DANGER state
    - FPS counter
    """
    
    # Colors (BGR format)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_MAGENTA = (255, 0, 255)
    COLOR_CYAN = (255, 255, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_ORANGE = (0, 165, 255)
    COLOR_RED = (0, 0, 255)
    
    # State to color mapping
    STATE_COLORS = {
        State.SAFE: (0, 255, 0),      # Green
        State.WARNING: (0, 165, 255),  # Orange
        State.DANGER: (0, 0, 255)      # Red
    }
    
    def __init__(self):
        """Initialize overlay renderer."""
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_large = 1.0
        self.font_scale_small = 0.6
        self.font_thickness = 2
        
        # Fingertip marker settings
        self.fingertip_radius = 12
        self.fingertip_thickness = 3
    
    def draw_all(self,
                 frame: np.ndarray,
                 rect: Tuple[int, int, int, int],
                 contour: Optional[np.ndarray],
                 hull: Optional[np.ndarray],
                 fingertip: Optional[Tuple[int, int]],
                 state: State,
                 fps: float,
                 distance: float) -> np.ndarray:
        """
        Draw all overlays on frame.
        
        Args:
            frame: BGR frame to draw on
            rect: Virtual rectangle (x, y, w, h)
            contour: Hand contour or None
            hull: Convex hull or None
            fingertip: (x, y) of fingertip or None
            state: Current interaction state
            fps: Current FPS
            distance: Smoothed distance value
        
        Returns:
            Frame with all overlays drawn
        """
        output = frame.copy()
        
        # Get state color
        state_color = self.STATE_COLORS.get(state, self.COLOR_WHITE)
        
        # Draw virtual rectangle
        output = self.draw_rectangle(output, rect, state_color)
        
        # Draw hand contour
        if contour is not None:
            output = self.draw_contour(output, contour)
        
        # Draw convex hull
        if hull is not None:
            output = self.draw_hull(output, hull)
        
        # Draw fingertip
        if fingertip is not None:
            output = self.draw_fingertip(output, fingertip)
        
        # Draw state banner
        output = self.draw_state_banner(output, state, distance)
        
        # Draw FPS counter
        output = self.draw_fps(output, fps)
        
        # Draw DANGER overlay if in DANGER state
        if state == State.DANGER:
            output = self.draw_danger_overlay(output)
        
        return output
    
    def draw_rectangle(self,
                       frame: np.ndarray,
                       rect: Tuple[int, int, int, int],
                       color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw virtual boundary rectangle.
        
        Args:
            frame: Frame to draw on
            rect: (x, y, width, height)
            color: BGR color tuple
        
        Returns:
            Frame with rectangle drawn
        """
        x, y, w, h = rect
        
        # Draw filled rectangle with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        return frame
    
    def draw_contour(self,
                     frame: np.ndarray,
                     contour: np.ndarray) -> np.ndarray:
        """
        Draw hand contour.
        
        Args:
            frame: Frame to draw on
            contour: OpenCV contour
        
        Returns:
            Frame with contour drawn
        """
        cv2.drawContours(frame, [contour], -1, self.COLOR_MAGENTA, 2)
        return frame
    
    def draw_hull(self,
                  frame: np.ndarray,
                  hull: np.ndarray) -> np.ndarray:
        """
        Draw convex hull.
        
        Args:
            frame: Frame to draw on
            hull: Convex hull points
        
        Returns:
            Frame with hull drawn
        """
        cv2.drawContours(frame, [hull], -1, self.COLOR_CYAN, 2)
        return frame
    
    def draw_fingertip(self,
                       frame: np.ndarray,
                       fingertip: Tuple[int, int]) -> np.ndarray:
        """
        Draw fingertip marker.
        
        Args:
            frame: Frame to draw on
            fingertip: (x, y) coordinates
        
        Returns:
            Frame with fingertip marker
        """
        x, y = fingertip
        
        # Draw outer circle (black outline)
        cv2.circle(frame, (x, y), self.fingertip_radius + 2, 
                   self.COLOR_BLACK, -1)
        
        # Draw filled yellow circle
        cv2.circle(frame, (x, y), self.fingertip_radius, 
                   self.COLOR_YELLOW, -1)
        
        # Draw white highlight
        cv2.circle(frame, (x - 3, y - 3), 4, self.COLOR_WHITE, -1)
        
        return frame
    
    def draw_state_banner(self,
                          frame: np.ndarray,
                          state: State,
                          distance: float) -> np.ndarray:
        """
        Draw state banner at top of frame.
        
        Args:
            frame: Frame to draw on
            state: Current state
            distance: Smoothed distance
        
        Returns:
            Frame with state banner
        """
        h, w = frame.shape[:2]
        
        # Get state color
        color = self.STATE_COLORS.get(state, self.COLOR_WHITE)
        
        # Draw background bar
        cv2.rectangle(frame, (0, 0), (w, 50), color, -1)
        
        # Prepare text
        state_text = state.value
        if distance < float('inf'):
            dist_text = f"Distance: {distance:.1f}px"
        else:
            dist_text = "Distance: --"
        
        # Draw state text (left side)
        cv2.putText(frame, state_text, (20, 35),
                    self.font, self.font_scale_large, 
                    self.COLOR_WHITE, self.font_thickness)
        
        # Draw distance text (right side)
        text_size = cv2.getTextSize(dist_text, self.font, 
                                     self.font_scale_small, 1)[0]
        cv2.putText(frame, dist_text, (w - text_size[0] - 20, 35),
                    self.font, self.font_scale_small,
                    self.COLOR_WHITE, 1)
        
        return frame
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw FPS counter.
        
        Args:
            frame: Frame to draw on
            fps: Current FPS value
        
        Returns:
            Frame with FPS counter
        """
        h, w = frame.shape[:2]
        
        fps_text = f"FPS: {fps:.1f}"
        
        # Draw with shadow for visibility
        cv2.putText(frame, fps_text, (12, h - 18),
                    self.font, self.font_scale_small,
                    self.COLOR_BLACK, 2)
        cv2.putText(frame, fps_text, (10, h - 20),
                    self.font, self.font_scale_small,
                    self.COLOR_WHITE, 1)
        
        return frame
    
    def draw_danger_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw "DANGER DANGER" flashing overlay.
        
        Args:
            frame: Frame to draw on
        
        Returns:
            Frame with danger overlay
        """
        h, w = frame.shape[:2]
        
        # Add red tint to frame
        red_overlay = np.zeros_like(frame)
        red_overlay[:, :, 2] = 100  # Red channel
        cv2.addWeighted(red_overlay, 0.3, frame, 1.0, 0, frame)
        
        # Draw "DANGER DANGER" text in center
        text = "DANGER DANGER"
        font_scale = 2.0
        thickness = 4
        
        # Get text size for centering
        text_size = cv2.getTextSize(text, self.font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        # Draw text with shadow
        cv2.putText(frame, text, (text_x + 3, text_y + 3),
                    self.font, font_scale, self.COLOR_BLACK, thickness + 2)
        cv2.putText(frame, text, (text_x, text_y),
                    self.font, font_scale, self.COLOR_RED, thickness)
        
        # Draw border
        cv2.rectangle(frame, (10, 10), (w - 10, h - 10), self.COLOR_RED, 4)
        
        return frame
    
    def draw_debug_info(self,
                        frame: np.ndarray,
                        info: dict) -> np.ndarray:
        """
        Draw debug information.
        
        Args:
            frame: Frame to draw on
            info: Dictionary of debug info
        
        Returns:
            Frame with debug info
        """
        h, w = frame.shape[:2]
        y_offset = 80
        
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset),
                        self.font, 0.5, self.COLOR_WHITE, 1)
            y_offset += 20
        
        return frame
