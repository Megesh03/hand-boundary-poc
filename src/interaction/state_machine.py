# src/interaction/state_machine.py
"""
State machine for SAFE/WARNING/DANGER states with hysteresis and smoothing.
"""

from collections import deque
from enum import Enum
from typing import Optional


class State(Enum):
    """Interaction states."""
    SAFE = "SAFE"
    WARNING = "WARNING"
    DANGER = "DANGER"


class StateMachine:
    """
    Three-state machine with hysteresis and distance smoothing.
    
    States:
        SAFE: distance > DIST_WARNING
        WARNING: DIST_DANGER < distance <= DIST_WARNING
        DANGER: distance <= DIST_DANGER
    
    Features:
        - Rolling average of last N distances for smoothing
        - Hysteresis margin to prevent state flickering
        - Immediate transitions to more dangerous states
        - Delayed transitions to safer states (requires margin)
    
    Thresholds (default):
        - DANGER: <= 10px
        - WARNING: <= 80px
        - SAFE: > 80px
        - Hysteresis margin: 15px
    """
    
    def __init__(self,
                 dist_danger: float = 10.0,
                 dist_warning: float = 80.0,
                 hysteresis_margin: float = 15.0,
                 buffer_size: int = 5):
        """
        Initialize state machine.
        
        Args:
            dist_danger: Distance threshold for DANGER state
            dist_warning: Distance threshold for WARNING state
            hysteresis_margin: Additional margin for exiting states
            buffer_size: Number of distances to average
        """
        self.dist_danger = dist_danger
        self.dist_warning = dist_warning
        self.hysteresis_margin = hysteresis_margin
        self.buffer_size = buffer_size
        
        # State and distance tracking
        self.distance_buffer = deque(maxlen=buffer_size)
        self.current_state = State.SAFE
        self.raw_distance = float('inf')
        self.smoothed_distance = float('inf')
    
    def update(self, distance: float) -> State:
        """
        Update state machine with new distance measurement.
        
        Args:
            distance: Distance from fingertip to boundary (pixels)
        
        Returns:
            Current state after update
        """
        self.raw_distance = distance
        
        # Add to buffer and compute smoothed distance
        self.distance_buffer.append(distance)
        self.smoothed_distance = self._compute_average()
        
        # Determine desired state based on smoothed distance
        desired_state = self._get_desired_state(self.smoothed_distance)
        
        # Apply hysteresis for state transitions
        self.current_state = self._apply_hysteresis(desired_state)
        
        return self.current_state
    
    def _compute_average(self) -> float:
        """Compute average distance from buffer."""
        if not self.distance_buffer:
            return float('inf')
        return sum(self.distance_buffer) / len(self.distance_buffer)
    
    def _get_desired_state(self, distance: float) -> State:
        """
        Get desired state based on distance thresholds.
        
        Args:
            distance: Smoothed distance value
        
        Returns:
            Desired state without hysteresis
        """
        if distance <= self.dist_danger:
            return State.DANGER
        elif distance <= self.dist_warning:
            return State.WARNING
        else:
            return State.SAFE
    
    def _apply_hysteresis(self, desired_state: State) -> State:
        """
        Apply hysteresis to prevent rapid state transitions.
        
        Rules:
        - Can always transition to a more dangerous state (immediate)
        - Transitioning to a safer state requires extra margin
        
        Args:
            desired_state: State without hysteresis
        
        Returns:
            State with hysteresis applied
        """
        current = self.current_state
        d = self.smoothed_distance
        margin = self.hysteresis_margin
        
        # State severity ordering (higher = more dangerous)
        severity = {State.SAFE: 0, State.WARNING: 1, State.DANGER: 2}
        
        # If transitioning to more dangerous state, do it immediately
        if severity[desired_state] > severity[current]:
            return desired_state
        
        # If transitioning to safer state, require extra margin
        if current == State.DANGER:
            # Exit DANGER only if distance > danger threshold + margin
            if d > self.dist_danger + margin:
                # Can exit to WARNING or SAFE
                if d > self.dist_warning + margin:
                    return State.SAFE
                elif d > self.dist_warning:
                    return State.WARNING
                else:
                    return State.WARNING
            else:
                return State.DANGER
        
        elif current == State.WARNING:
            # Exit WARNING to SAFE only if distance > warning threshold + margin
            if d > self.dist_warning + margin:
                return State.SAFE
            elif d <= self.dist_danger:
                return State.DANGER
            else:
                return State.WARNING
        
        # Default: stay in current state
        return current
    
    def reset(self):
        """Reset state machine to initial state."""
        self.distance_buffer.clear()
        self.current_state = State.SAFE
        self.raw_distance = float('inf')
        self.smoothed_distance = float('inf')
    
    def get_state(self) -> State:
        """Get current state."""
        return self.current_state
    
    def get_state_string(self) -> str:
        """Get current state as string."""
        return self.current_state.value
    
    def get_raw_distance(self) -> float:
        """Get most recent raw distance."""
        return self.raw_distance
    
    def get_smoothed_distance(self) -> float:
        """Get smoothed (averaged) distance."""
        return self.smoothed_distance
    
    def is_danger(self) -> bool:
        """Check if in DANGER state."""
        return self.current_state == State.DANGER
    
    def is_warning(self) -> bool:
        """Check if in WARNING state."""
        return self.current_state == State.WARNING
    
    def is_safe(self) -> bool:
        """Check if in SAFE state."""
        return self.current_state == State.SAFE
