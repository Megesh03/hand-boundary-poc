# src/config.py
"""
Configuration constants for Hand Boundary POC.
All tunable parameters are defined here for easy adjustment.

IMPORTANT: This system uses ONLY TinyUNet ONNX model for segmentation.
No fallback modes (HSV, motion, skin-color) are allowed.
"""

# =============================================================================
# Frame Configuration
# =============================================================================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0

# =============================================================================
# Segmentation Mode Configuration
# =============================================================================
# Toggle between ML (TinyUNet) and Classical CV segmentation
USE_ML_SEGMENTATION = False  # Set to True to use TinyUNet, False for classical CV
ENABLE_FALLBACK = True  # Automatically fallback to classical if ML produces empty mask

# =============================================================================
# Model Configuration (TinyUNet ONNX - OPTIONAL when USE_ML_SEGMENTATION=False)
# =============================================================================
MODEL_INPUT_SZ = 160  # TinyUNet input size (160x160)
MODEL_PATH = "models/handseg.onnx"

# Inference settings
MASK_THRESHOLD = 0.5  # Sigmoid threshold for binary mask

# =============================================================================
# Classical Segmentation Configuration (YCrCb + Morphology)
# =============================================================================
# YCrCb skin color thresholds (typical for various skin tones)
SKIN_CR_MIN = 133  # Minimum Cr (red-difference) channel value
SKIN_CR_MAX = 173  # Maximum Cr channel value
SKIN_CB_MIN = 77   # Minimum Cb (blue-difference) channel value
SKIN_CB_MAX = 127  # Maximum Cb channel value

# Motion differencing
USE_MOTION = True  # Enable motion differencing for better hand isolation
MOTION_THRESHOLD = 20  # Frame difference threshold (lower = more sensitive)

# =============================================================================
# Segmentation Postprocessing
# =============================================================================
MORPH_KERNEL_SIZE = (5, 5)
MORPH_CLOSE_ITERS = 2
MEDIAN_BLUR_K = 5
MIN_CONTOUR_FILL_AREA = 50  # Minimum area for hole filling

# =============================================================================
# Hand Detection
# =============================================================================
MIN_HAND_CONTOUR_AREA = 2000  # Minimum pixels for valid hand contour

# =============================================================================
# Fingertip Detection
# =============================================================================
FALLBACK_FAR_FROM_CENTROID = True  # Use farthest point if topmost is below centroid

# =============================================================================
# Distance Thresholds (pixels to box edge)
# =============================================================================
DIST_DANGER = 10.0    # DANGER if distance <= this
DIST_WARNING = 80.0   # WARNING if distance <= this (and > DANGER)
HYSTERESIS_MARGIN = 15.0  # Margin to prevent state flickering
SMOOTHING_BUFFER_SIZE = 5  # Number of samples for distance smoothing

# =============================================================================
# Performance
# =============================================================================
FPS_TARGET = 8  # Minimum target FPS

# =============================================================================
# Virtual Boundary Rectangle (UI)
# =============================================================================
VBOX_X = 400
VBOX_Y = 150
VBOX_W = 200
VBOX_H = 200

# =============================================================================
# UI Colors (BGR format for OpenCV)
# =============================================================================
COLOR_WHITE = (255, 255, 255)
COLOR_MAGENTA = (255, 0, 255)    # Hand contour
COLOR_CYAN = (255, 255, 0)       # Convex hull
COLOR_YELLOW = (0, 255, 255)     # Fingertip
COLOR_GREEN = (0, 255, 0)        # SAFE state
COLOR_ORANGE = (0, 165, 255)     # WARNING state
COLOR_RED = (0, 0, 255)          # DANGER state
COLOR_BLACK = (0, 0, 0)

# =============================================================================
# State Names
# =============================================================================
STATE_SAFE = "SAFE"
STATE_WARNING = "WARNING"
STATE_DANGER = "DANGER"
