# src/vision/__init__.py
"""
Vision module for hand segmentation and analysis.

IMPORTANT: This module uses ONLY TinyUNet ONNX model for segmentation.
No fallback segmentation methods are implemented.
"""

from .segmentation_model import SegmentationModel
from .hand_analyzer import HandAnalyzer

__all__ = ['SegmentationModel', 'HandAnalyzer']
