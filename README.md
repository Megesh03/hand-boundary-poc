# Hand Boundary POC

Real-time hand tracking system with **TinyUNet ONNX segmentation** and SAFE/WARNING/DANGER state machine.

> ⚠️ **IMPORTANT**: This system uses **ONLY** the TinyUNet ONNX model for segmentation.
> **No fallback modes** (HSV, skin-color, motion) are available.
> You **must train the model** before running the application.

![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![ONNXRuntime](https://img.shields.io/badge/ONNX-1.15+-purple.svg)

## Features

- **Real-time webcam tracking** of hand and fingertip
- **TinyUNet CNN** for binary hand mask prediction (160×160 input)
- **Classical CV** contour + convex hull fingertip detection
- **Distance-based state machine** with hysteresis:
  - 🟢 **SAFE**: distance > 80px
  - 🟡 **WARNING**: 10px < distance ≤ 80px
  - 🔴 **DANGER**: distance ≤ 10px + "DANGER DANGER" overlay
- **≥8 FPS** CPU-only inference
- **No MediaPipe/OpenPose/Cloud APIs**
- **No fallback segmentation** — model must be trained

## Quick Start

### 1. Install Dependencies

```bash
cd hand_boundary_poc
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
python train/prepare_dataset.py --output_dir ./data --num_synthetic 500
```

### 3. Train the Model

```bash
python train/train_unet.py --data_dir ./data/synthetic --epochs 30
```

### 4. Export to ONNX

```bash
python train/export_onnx.py --checkpoint ./checkpoints/best_model.pth --output ./models/handseg.onnx
```

### 5. Run the Application

```bash
python src/main.py
```

**Controls:**
- Press `Q` to quit

## Architecture

### TinyUNet Model

```
Input: 160×160×3 RGB

Encoder:
├── DoubleConv: 3 → 32 channels (160×160)
├── Down + DoubleConv: 32 → 64 channels (80×80)
└── Down + DoubleConv: 64 → 128 channels (40×40)

Decoder:
├── Up + DoubleConv: 128 → 64 channels (80×80)
├── Up + DoubleConv: 64 → 32 channels (160×160)
└── 1×1 Conv: 32 → 1 channel (mask output)

Output: 160×160×1 binary mask
```

### Processing Pipeline

```
Camera Frame
     ↓
TinyUNet ONNX Model
     ↓
Binary Mask → Morphological Cleanup
     ↓
Find Contour → Convex Hull → Fingertip
     ↓
Distance to Rectangle
     ↓
State Machine (SAFE/WARNING/DANGER)
     ↓
Draw Overlays → Display
```

## Project Structure

```
hand_boundary_poc/
├── README.md                         # This file
├── DEVPLAN.md                        # Development plan
├── requirements.txt                  # Dependencies
├── models/
│   └── handseg.onnx                  # Trained ONNX model
├── train/
│   ├── prepare_dataset.py            # Dataset preparation
│   ├── train_unet.py                 # Training script
│   └── export_onnx.py                # ONNX export
├── src/
│   ├── main.py                       # Main application
│   ├── config.py                     # Configuration
│   ├── camera/
│   │   └── capture.py                # Webcam wrapper
│   ├── vision/
│   │   ├── segmentation_model.py     # ONNX inference (NO FALLBACK)
│   │   └── hand_analyzer.py          # Contour/hull/fingertip
│   ├── interaction/
│   │   ├── virtual_object.py         # Rectangle boundary
│   │   └── state_machine.py          # State logic
│   ├── ui/
│   │   └── overlay.py                # UI rendering
│   └── utils/
│       ├── timing.py                 # FPS counter
│       └── image_utils.py            # Image utilities
└── tests/
    ├── test_segmentation_model.py
    ├── test_hand_analyzer.py
    └── test_state_machine.py
```

## Configuration

Edit `src/config.py` to adjust:

```python
# Frame settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Model
MODEL_INPUT_SZ = 160
MODEL_PATH = "models/handseg.onnx"

# Distance thresholds
DIST_DANGER = 10.0     # pixels
DIST_WARNING = 80.0    # pixels
HYSTERESIS_MARGIN = 15.0
SMOOTHING_BUFFER_SIZE = 5

# Virtual rectangle position
VBOX_X = 400
VBOX_Y = 150
VBOX_W = 200
VBOX_H = 200
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_state_machine.py -v
```

## Requirements

### Runtime
- Python 3.8+
- OpenCV 4.8+
- NumPy 1.24+
- ONNXRuntime 1.15+
- Webcam

### Training
- PyTorch 2.0+
- torchvision 0.15+
- albumentations 1.3+

## Performance

| Metric | Target | Typical |
|--------|--------|---------|
| FPS | ≥ 8 | 15-30 |
| Model Size | < 5MB | ~2MB |
| Latency | < 125ms | 33-66ms |

## Troubleshooting

### Model not found error
```
ERROR: ONNX model not found!
```
**Solution:** Train the model first using the steps in Quick Start.

### Camera not detected
```bash
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Low FPS
1. Reduce `MODEL_INPUT_SZ` to 128 in `config.py`
2. Ensure no other applications using webcam
3. Check CPU usage

## Constraints (Recruiter Requirements)

This project strictly adheres to the following constraints:

- ✅ **CPU-only** inference
- ✅ **TinyUNet ONNX** model only
- ✅ **No MediaPipe/OpenPose**
- ✅ **No cloud APIs**
- ❌ **No HSV/skin-color segmentation**
- ❌ **No motion-based segmentation**
- ❌ **No fallback modes**

## License

MIT License

## Acknowledgments

- TinyUNet architecture inspired by U-Net (Ronneberger et al., 2015)
