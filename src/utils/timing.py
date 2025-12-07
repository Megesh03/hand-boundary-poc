# src/utils/timing.py
"""
Timing utilities for FPS calculation.
"""

import time
from collections import deque
from typing import Optional


class FPSCounter:
    """
    FPS counter with rolling average.
    
    Uses a buffer of recent frame times to compute
    a smoothed FPS value.
    
    Attributes:
        buffer_size: Number of frame times to average
        fps: Current FPS value
    """
    
    def __init__(self, buffer_size: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            buffer_size: Number of frame times to average
        """
        self.buffer_size = buffer_size
        self.frame_times = deque(maxlen=buffer_size)
        self.last_time: Optional[float] = None
        self.fps = 0.0
    
    def tick(self) -> float:
        """
        Record a frame tick and return current FPS.
        
        Call this once per frame.
        
        Returns:
            Current smoothed FPS value
        """
        current_time = time.perf_counter()
        
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
            
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                if avg_frame_time > 0:
                    self.fps = 1.0 / avg_frame_time
        
        self.last_time = current_time
        return self.fps
    
    def get_fps(self) -> float:
        """Get current FPS value."""
        return self.fps
    
    def reset(self):
        """Reset FPS counter."""
        self.frame_times.clear()
        self.last_time = None
        self.fps = 0.0
    
    def get_frame_time_ms(self) -> float:
        """Get average frame time in milliseconds."""
        if len(self.frame_times) > 0:
            return (sum(self.frame_times) / len(self.frame_times)) * 1000
        return 0.0


class Timer:
    """
    Simple timer for profiling code sections.
    
    Usage:
        timer = Timer()
        timer.start()
        # ... do work ...
        elapsed = timer.stop()
    """
    
    def __init__(self):
        """Initialize timer."""
        self._start_time: Optional[float] = None
        self._elapsed: float = 0.0
    
    def start(self):
        """Start the timer."""
        self._start_time = time.perf_counter()
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self._start_time is not None:
            self._elapsed = time.perf_counter() - self._start_time
            self._start_time = None
        return self._elapsed
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self._elapsed * 1000
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
