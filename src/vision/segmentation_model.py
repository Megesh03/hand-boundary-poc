# src/vision/segmentation_model.py
"""
TinyUNet ONNX model wrapper for hand segmentation.

IMPORTANT: This is the ONLY segmentation method allowed.
No fallback modes (HSV, skin-color, motion) are implemented.
The ONNX model MUST be trained and available for the system to work.
"""

import os
import cv2
import numpy as np
from typing import Tuple

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class SegmentationModel:
    """
    ONNX Runtime wrapper for TinyUNet hand segmentation model.
    
    This is the ONLY segmentation method in the system.
    No fallback modes are provided - the model MUST be available.
    
    Pipeline:
    1. Preprocessing: BGR→RGB, resize to 160×160, normalize, NCHW format
    2. Inference: Run ONNX model
    3. Postprocessing: sigmoid threshold, resize, morphological cleanup, hole fill
    
    Attributes:
        session: ONNX Runtime inference session
        input_size: Model input size (160)
        input_name: Model input tensor name
    """
    
    def __init__(self,
                 model_path: str,
                 input_size: int = 160,
                 threshold: float = 0.35,
                 provider: str = 'CPUExecutionProvider'):
        """
        Initialize segmentation model.
        
        Args:
            model_path: Path to ONNX model file
            input_size: Model input size (assumes square)
            threshold: Sigmoid threshold for binary mask
            provider: ONNX Runtime execution provider
        
        Raises:
            RuntimeError: If ONNX runtime not available
            FileNotFoundError: If model file doesn't exist
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is required but not installed. "
                "Install with: pip install onnxruntime"
            )
        
        self.model_path = model_path
        self.input_size = input_size
        self.threshold = threshold
        self.session = None
        self.input_name = None
        self._model_loaded = False
        
        # Morphological kernel for post-processing (smaller kernel to preserve details)
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )
        
        # Load model
        self._load_model(provider)
    
    def _load_model(self, provider: str):
        """
        Load the ONNX model.
        
        Args:
            provider: ONNX Runtime execution provider
        
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"ONNX model not found at: {self.model_path}\n"
                f"Please train the model first using train/train_unet.py\n"
                f"Then export using train/export_onnx.py"
            )
        
        try:
            # Configure session options for performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            # Create inference session
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options,
                providers=[provider]
            )
            
            # Get input name
            self.input_name = self.session.get_inputs()[0].name
            self._model_loaded = True
            
            print(f"Model loaded successfully: {self.model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._model_loaded and self.session is not None
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model inference.
        
        Pipeline:
        1. BGR to RGB conversion
        2. Resize to model input size (160×160)
        3. Normalize to [0, 1] (MUST match training!)
        4. Convert HWC to NCHW format
        
        Args:
            frame: BGR frame from OpenCV (H, W, 3)
        
        Returns:
            Preprocessed tensor (1, 3, input_size, input_size) as float32
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(
            rgb, 
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize to [0, 1] - MATCHES TRAINING NORMALIZATION
        # Training uses: A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        # which outputs pixel values / 255.0 in range [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert HWC to NCHW format (batch, channels, height, width)
        nchw = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        
        return nchw
    
    def postprocess(self,
                    output: np.ndarray,
                    original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output to binary mask.
        
        Pipeline:
        1. Extract mask from output tensor
        2. Apply sigmoid always
        3. Threshold to binary (0.2)
        4. Resize to original frame size
        5. Morphological cleanup (CLOSE with 5x5 kernel)
        6. Debug visualization
        
        Args:
            output: Model output tensor
            original_size: Original frame size (width, height)
        
        Returns:
            Binary mask (0 or 255) at original size
        """
        # Extract mask from output tensor
        if output.ndim == 4:
            if output.shape[1] == 1:
                pred = output[0, 0]  # Shape (1, 1, H, W)
            else:
                pred = output[0, :, :, 0]  # Shape (1, H, W, 1)
        elif output.ndim == 3:
            pred = output[0]
        else:
            pred = output
        
        # Apply sigmoid always
        pred = 1 / (1 + np.exp(-pred))
        
        # Threshold to binary (0.2)
        mask = (pred > 0.2).astype(np.uint8) * 255
        
        # Resize to original frame size
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        # Morphological cleanup - CLOSE with 5x5 kernel
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Debug visualization
        cv2.imshow("MASK", mask)
        
        return mask
    
    def _fill_holes(self, mask: np.ndarray, min_area: int = 50) -> np.ndarray:
        """
        Fill holes in the mask using contour filling.
        
        Args:
            mask: Binary mask
            min_area: Minimum contour area to keep
        
        Returns:
            Mask with holes filled
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        filled = np.zeros_like(mask)
        
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(filled, [contour], -1, 255, -1)
        
        return filled
    
    def predict_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Predict hand segmentation mask for a frame.
        
        This is the main inference method.
        
        Args:
            frame: BGR frame from OpenCV
        
        Returns:
            Binary mask (0 or 255) at frame resolution.
        
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded():
            raise RuntimeError(
                "Model is not loaded. Cannot perform inference.\n"
                "Please ensure the ONNX model exists at the configured path."
            )
        
        h, w = frame.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(frame)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        output = outputs[0]
        
        # Postprocess
        mask = self.postprocess(output, (w, h))
        
        return mask
    
    def predict_raw(self, frame: np.ndarray) -> np.ndarray:
        """
        Get raw model output without postprocessing.
        
        Useful for debugging and visualization.
        
        Args:
            frame: BGR frame from OpenCV
        
        Returns:
            Raw model output tensor
        
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded")
        
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        return outputs[0]
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model details
        """
        if not self.is_loaded():
            return {"loaded": False, "path": self.model_path}
        
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        return {
            "loaded": True,
            "path": self.model_path,
            "input_size": self.input_size,
            "threshold": self.threshold,
            "inputs": [
                {"name": inp.name, "shape": inp.shape, "type": inp.type}
                for inp in inputs
            ],
            "outputs": [
                {"name": out.name, "shape": out.shape, "type": out.type}
                for out in outputs
            ]
        }
