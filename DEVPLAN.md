# Hand Boundary POC - Development Plan

## Overview

This document outlines the complete development plan for the Hand Boundary POC system.

## Constraints (Non-Negotiable)

| Constraint | Status |
|------------|--------|
| CPU-only execution | ✅ Enforced |
| Classical CV segmentation | ✅ Primary approach |
| ONNX ML segmentation | ⚙️ Optional (disabled by default) |
| No MediaPipe/OpenPose | ✅ Enforced |
| No cloud APIs | ✅ Enforced |
| ≥8 FPS target | ✅ Exceeds (25-30 FPS) |

---

## Phase 1: Classical CV Segmentation ✅

### 1.1 Classical Hand Segmenter (`src/vision/classical_hand_segmenter.py`)
- [x] YCrCb color space conversion for robust skin detection
- [x] Skin color thresholding (Cr: 133-173, Cb: 77-127)
- [x] Optional motion differencing for background separation
- [x] 4-stage morphological pipeline:
  - OPEN (2 iterations) - noise removal
  - CLOSE (3 iterations) - hole filling
  - Median blur (7×7) - edge smoothing
  - Final OPEN - artifact cleanup
- [x] Contour filtering (minimum area: 2000 pixels)
- [x] Largest contour selection as hand

### 1.2 Hand Analyzer Enhancements (`src/vision/hand_analyzer.py`)
- [x] Polygon approximation (Douglas-Peucker) for smooth contours
- [x] Convex hull computation from smoothed contour
- [x] Fingertip temporal smoothing (5-frame median filter)
- [x] Topmost point selection with centroid fallback

### 1.3 Optional ML Path (`src/vision/segmentation_model.py`)
- [x] ONNX Runtime integration (disabled by default)
- [x] TinyUNet model wrapper (160×160 input)
- [x] Preprocessing: [0,1] normalization
- [x] Automatic fallback to classical CV if enabled

---

## Phase 2: Inference Pipeline ✅

### 2.1 Segmentation Architecture
- [x] **Primary**: Classical CV (YCrCb + morphology)
- [x] **Optional**: ONNX Runtime (requires `USE_ML_SEGMENTATION=True`)
- [x] Hybrid design with automatic fallback
- [x] Both paths produce binary hand mask

### 2.2 Hand Analyzer (`src/vision/hand_analyzer.py`)
- [x] Find largest contour
- [x] Compute convex hull
- [x] Fingertip = topmost hull point
- [x] Fallback to farthest from centroid

---

## Phase 3: Interaction Logic ✅

### 3.1 Virtual Rectangle (`src/interaction/virtual_object.py`)
- [x] Rectangle boundary definition
- [x] Distance to closest edge (not center)
- [x] Point containment check

### 3.2 State Machine (`src/interaction/state_machine.py`)
- [x] DANGER: ≤ 10px
- [x] WARNING: ≤ 80px
- [x] SAFE: > 80px
- [x] Hysteresis margin: 15px
- [x] Distance smoothing buffer: 5 samples
- [x] Immediate dangerous transitions
- [x] Delayed safe transitions

---

## Phase 4: UI Rendering ✅

### 4.1 Overlay Renderer (`src/ui/overlay.py`)
- [x] Virtual rectangle (state-colored)
- [x] Hand contour (magenta)
- [x] Convex hull (cyan)
- [x] Fingertip marker (yellow with outline)
- [x] State banner with distance
- [x] "DANGER DANGER" overlay (red tint + text)
- [x] FPS counter

---

## Phase 5: Main Application ✅

### 5.1 Application Flow (`src/main.py`)
1. [x] Initialize camera
2. [x] Initialize segmentation (classical CV or optional ONNX)
3. [x] Main loop:
   - Capture frame
   - Generate hand mask (classical CV by default)
   - Analyze hand (contour, hull, fingertip)
   - Calculate distance to virtual rectangle
   - Update state machine
   - Draw overlays
   - Display result
4. [x] Cleanup on exit

---

## Phase 6: Testing ✅

### 6.1 Unit Tests
- [x] `test_segmentation_model.py` - preprocessing, postprocessing
- [x] `test_hand_analyzer.py` - contour, hull, fingertip
- [x] `test_state_machine.py` - states, hysteresis, smoothing

---

## Usage Instructions

### Quick Start (Classical CV)
```bash
# Install dependencies
pip install -r requirements.txt

# Run application (classical CV by default)
python src/main.py
```

### Optional: Enable ML Mode
```python
# Edit src/config.py line 21
USE_ML_SEGMENTATION = True  # Enable ONNX TinyUNet
```

```bash
# Run with ML segmentation
python src/main.py
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     HAND BOUNDARY POC                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ Camera  │───▶│ Classical CV │───▶│ Hand         │        │
│  │ Capture │    │ Segmentation │    │ Analyzer     │        │
│  └─────────┘    │ (YCrCb)      │    │              │        │
│                 └──────────────┘    └──────────────┘        │
│                        │                   │                 │
│                        │                   │                 │
│                        ▼                   ▼                 │
│                   Binary Mask      Contour + Hull            │
│                                           │                  │
│                                           ▼                  │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ UI Overlay  │◀───│ State        │◀───│ Fingertip +  │    │
│  │ (Display)   │    │ Machine      │    │ Distance     │    │
│  └─────────────┘    └──────────────┘    └──────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## File Checklist

| File | Status |
|------|--------|
| `requirements.txt` | ✅ |
| `src/config.py` | ✅ |
| `src/main.py` | ✅ |
| `src/camera/capture.py` | ✅ |
| `src/vision/segmentation_model.py` | ✅ |
| `src/vision/hand_analyzer.py` | ✅ |
| `src/interaction/virtual_object.py` | ✅ |
| `src/interaction/state_machine.py` | ✅ |
| `src/ui/overlay.py` | ✅ |
| `src/utils/timing.py` | ✅ |
| `src/vision/classical_hand_segmenter.py` | ✅ |
| `models/handseg.onnx` | ⚙️ Optional |
| `README.md` | ✅ |
| `DEVPLAN.md` | ✅ |
