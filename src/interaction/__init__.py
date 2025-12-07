# src/interaction/__init__.py
"""Interaction module for virtual objects and state machine."""

from .virtual_object import VirtualRectangle
from .state_machine import StateMachine, State

__all__ = ['VirtualRectangle', 'StateMachine', 'State']
