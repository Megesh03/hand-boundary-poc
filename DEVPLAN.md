# Hand Boundary POC - Development Plan

## Overview

This document outlines the complete development plan for the Hand Boundary POC system.

## Constraints (Non-Negotiable)

| Constraint | Status |
|------------|--------|
| CPU-only execution | ✅ Enforced |
| TinyUNet ONNX segmentation only | ✅ Enforced |
| No MediaPipe/OpenPose | ✅ Enforced |
| No cloud APIs | ✅ Enforced |
| No HSV/skin segmentation | ✅ Enforced |
| No motion segmentation | ✅ Enforced |
| No fallback modes | ✅ Enforced |
| ≥8 FPS target | ✅ Designed for |

---

## Phase 1: Model Training Pipeline ✅

### 1.1 Dataset Preparation (`train/prepare_dataset.py`)
- [x] Generate synthetic hand images with masks
- [x] Support for EgoHands dataset processing
- [x] Train/val split creation
- [x] Configurable sample counts

### 1.2 TinyUNet Architecture (`train/train_unet.py`)
- [x] 3-level encoder: 32 → 64 → 128 channels
- [x] 3-level decoder: 128 → 64 → 32 channels
- [x] Skip connections
- [x] ~500K parameters

### 1.3 Training Loop
- [x] BCE + Dice combined loss
- [x] Adam optimizer with ReduceLROnPlateau
- [x] Heavy augmentations (albumentations)
- [x] Early stopping (patience=5)
- [x] Best model checkpointing

### 1.4 ONNX Export (`train/export_onnx.py`)
- [x] Export to ONNX format
- [x] Dynamic batch size support
- [x] Model verification

---

## Phase 2: Inference Pipeline ✅

### 2.1 Segmentation Model (`src/vision/segmentation_model.py`)
- [x] ONNX Runtime CPU session
- [x] Preprocessing: BGR→RGB, resize 160×160, normalize, NCHW
- [x] Postprocessing: sigmoid, threshold, resize, morphology, hole fill
- [x] **No fallback modes** - raises error if model missing

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
2. [x] Load ONNX model (error if missing)
3. [x] Main loop:
   - Capture frame
   - Predict mask (ONNX only)
   - Analyze hand
   - Calculate distance
   - Update state
   - Draw overlays
   - Display
4. [x] Cleanup on exit

---

## Phase 6: Testing ✅

### 6.1 Unit Tests
- [x] `test_segmentation_model.py` - preprocessing, postprocessing
- [x] `test_hand_analyzer.py` - contour, hull, fingertip
- [x] `test_state_machine.py` - states, hysteresis, smoothing

---

## Usage Instructions

### Step 1: Prepare Dataset
```bash
python train/prepare_dataset.py --output_dir ./data --num_synthetic 500
```

### Step 2: Train Model
```bash
python train/train_unet.py --data_dir ./data/synthetic --epochs 30
```

### Step 3: Export ONNX
```bash
python train/export_onnx.py
```

### Step 4: Run Application
```bash
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
│  │ Camera  │───▶│ TinyUNet     │───▶│ Hand         │        │
│  │ Capture │    │ ONNX Model   │    │ Analyzer     │        │
│  └─────────┘    │ (ONLY)       │    │              │        │
│                 └──────────────┘    └──────────────┘        │
│                        │                   │                 │
│                        │                   │                 │
│                        ▼                   ▼                 │
│                   Binary Mask         Fingertip              │
│                                           │                  │
│                                           ▼                  │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ UI Overlay  │◀───│ State        │◀───│ Virtual      │    │
│  │ (Display)   │    │ Machine      │    │ Rectangle    │    │
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
| `src/utils/image_utils.py` | ✅ |
| `train/prepare_dataset.py` | ✅ |
| `train/train_unet.py` | ✅ |
| `train/export_onnx.py` | ✅ |
| `tests/test_segmentation_model.py` | ✅ |
| `tests/test_hand_analyzer.py` | ✅ |
| `tests/test_state_machine.py` | ✅ |
| `README.md` | ✅ |
| `DEVPLAN.md` | ✅ |
