# src/main.py
"""
Hand Boundary POC - Main Application

Real-time hand tracking with TinyUNet ONNX segmentation and 
SAFE/WARNING/DANGER state machine.

IMPORTANT: This application uses ONLY the TinyUNet ONNX model for segmentation.
No fallback segmentation methods are available.
The model MUST be trained and exported before running.

Usage:
    python src/main.py

Controls:
    Q - Quit application
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

# Import modules
from config import (
    FRAME_WIDTH, FRAME_HEIGHT, MODEL_PATH, MODEL_INPUT_SZ,
    VBOX_X, VBOX_Y, VBOX_W, VBOX_H,
    DIST_DANGER, DIST_WARNING, HYSTERESIS_MARGIN, SMOOTHING_BUFFER_SIZE,
    MIN_HAND_CONTOUR_AREA, FALLBACK_FAR_FROM_CENTROID, MASK_THRESHOLD,
    USE_ML_SEGMENTATION, ENABLE_FALLBACK,
    SKIN_CR_MIN, SKIN_CR_MAX, SKIN_CB_MIN, SKIN_CB_MAX,
    USE_MOTION, MOTION_THRESHOLD
)
from camera.capture import CameraCapture
from vision.segmentation_model import SegmentationModel
from vision.classical_hand_segmenter import ClassicalHandSegmenter
from vision.hand_analyzer import HandAnalyzer
from interaction.virtual_object import VirtualRectangle
from interaction.state_machine import StateMachine, State
from ui.overlay import OverlayRenderer
from utils.timing import FPSCounter


class HandBoundaryApp:
    """
    Main application class for hand boundary tracking.
    
    HYBRID SEGMENTATION:
    - Uses TinyUNet ONNX model when USE_ML_SEGMENTATION=True
    - Falls back to classical CV (YCrCb + morphology) when:
      * USE_ML_SEGMENTATION=False, OR
      * ML produces empty mask and ENABLE_FALLBACK=True
    
    Pipeline:
    1. Capture frame from webcam
    2. Predict hand mask (ML or classical)
    3. Analyze mask (contour, hull, fingertip)
    4. Calculate distance to virtual rectangle
    5. Update state machine
    6. Render overlays
    7. Display result
    """
    
    def __init__(self):
        """Initialize application components."""
        print("=" * 60)
        print("Hand Boundary POC")
        print("Hybrid Segmentation (ML + Classical CV)")
        print("=" * 60)
        print()
        print("Initializing...")
        
        # Initialize camera
        self.camera = CameraCapture(
            camera_index=0,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT
        )
        
        # Initialize segmentation (hybrid approach)
        self.use_ml = USE_ML_SEGMENTATION
        self.enable_fallback = ENABLE_FALLBACK
        
        # Initialize ML segmentation (if enabled)
        self.ml_segmenter = None
        if self.use_ml:
            model_path = self._get_model_path()
            print(f"ML Segmentation: ENABLED")
            print(f"Loading TinyUNet model: {model_path}")
            
            try:
                self.ml_segmenter = SegmentationModel(
                    model_path=model_path,
                    input_size=MODEL_INPUT_SZ,
                    threshold=MASK_THRESHOLD
                )
                print(f"✓ TinyUNet loaded successfully")
            except FileNotFoundError as e:
                print(f"✗ TinyUNet model not found: {model_path}")
                if not self.enable_fallback:
                    raise SystemExit(1)
                print("  → Will use classical segmentation only")
                self.use_ml = False
        else:
            print(f"ML Segmentation: DISABLED (using classical CV only)")
        
        # Initialize classical segmentation (always available as fallback)
        print(f"Classical Segmentation: ENABLED")
        self.classical_segmenter = ClassicalHandSegmenter(
            skin_cr_min=SKIN_CR_MIN,
            skin_cr_max=SKIN_CR_MAX,
            skin_cb_min=SKIN_CB_MIN,
            skin_cb_max=SKIN_CB_MAX,
            motion_threshold=MOTION_THRESHOLD,
            min_contour_area=MIN_HAND_CONTOUR_AREA,
            use_motion=USE_MOTION
        )
        print(f"✓ Classical segmenter initialized")
        
        # Track which mode is being used
        self.using_ml_mode = self.use_ml
        
        # Initialize hand analyzer
        self.analyzer = HandAnalyzer(
            min_contour_area=MIN_HAND_CONTOUR_AREA,
            fallback_to_centroid=FALLBACK_FAR_FROM_CENTROID
        )
        
        # Initialize virtual rectangle
        self.virtual_rect = VirtualRectangle(
            VBOX_X, VBOX_Y, VBOX_W, VBOX_H
        )
        
        # Initialize state machine
        self.state_machine = StateMachine(
            dist_danger=DIST_DANGER,
            dist_warning=DIST_WARNING,
            hysteresis_margin=HYSTERESIS_MARGIN,
            buffer_size=SMOOTHING_BUFFER_SIZE
        )
        
        # Initialize UI overlay renderer
        self.overlay = OverlayRenderer()
        
        # Initialize FPS counter
        self.fps_counter = FPSCounter()
        
        # Window name
        self.window_name = "Hand Boundary POC"
        
        print()
        print("Initialization complete!")
        print()
    
    def _get_model_path(self) -> str:
        """Get the path to the ONNX model file."""
        # Try multiple paths
        paths = [
            MODEL_PATH,
            os.path.join(os.path.dirname(__file__), '..', MODEL_PATH),
            os.path.join(os.path.dirname(__file__), '..', 'models', 'handseg.onnx')
        ]
        
        for path in paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        # Return default path (will raise error in SegmentationModel)
        return os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', MODEL_PATH
        ))
    
    def run(self):
        """Run the main application loop."""
        print("=" * 60)
        print("Starting Hand Boundary POC")
        print("Press 'Q' to quit")
        print("=" * 60)
        print()
        
        # Open camera
        try:
            self.camera.open()
        except RuntimeError as e:
            print(f"Error: {e}")
            return
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, FRAME_WIDTH, FRAME_HEIGHT)
        
        try:
            self._main_loop()
        finally:
            self._cleanup()
    
    def _main_loop(self):
        """Main processing loop."""
        while True:
            # Capture frame
            try:
                frame = self.camera.read()
            except RuntimeError as e:
                print(f"Error: {e}")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Update FPS
            fps = self.fps_counter.tick()
            
            # Process frame
            output_frame = self._process_frame(frame, fps)
            
            # Display
            cv2.imshow(self.window_name, output_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print()
                print("Quitting...")
                break
    
    def _process_frame(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame: BGR frame from camera
            fps: Current FPS value
        
        Returns:
            Frame with overlays drawn
        """
        # Get hand segmentation mask (hybrid approach)
        mask = None
        used_classical = False
        
        # Try ML segmentation first (if enabled)
        if self.use_ml and self.ml_segmenter is not None:
            try:
                mask = self.ml_segmenter.predict_mask(frame)
                
                # Check if mask is empty (no hand detected by ML)
                if self.enable_fallback and np.count_nonzero(mask) == 0:
                    # Fall back to classical segmentation
                    mask = self.classical_segmenter.segment(frame)
                    used_classical = True
                    if not self.using_ml_mode:
                        print("→ ML produced empty mask, using classical fallback")
                        self.using_ml_mode = False
                else:
                    # ML produced valid mask
                    if not self.using_ml_mode:
                        print("→ ML segmentation working")
                        self.using_ml_mode = True
            except Exception as e:
                # ML inference failed, use classical
                print(f"ML inference error: {e}")
                mask = self.classical_segmenter.segment(frame)
                used_classical = True
                self.using_ml_mode = False
        else:
            # Use classical segmentation
            mask = self.classical_segmenter.segment(frame)
            used_classical = True
        
        # Analyze hand (contour, hull, fingertip)
        analysis = self.analyzer.analyze(mask)
        
        contour = analysis['contour']
        hull = analysis['hull']
        fingertip = analysis['fingertip']
        
        # Calculate distance and update state
        if fingertip is not None:
            distance = self.virtual_rect.distance_to_point(fingertip)
            state = self.state_machine.update(distance)
        else:
            distance = float('inf')
            # Don't update state if no fingertip detected
            state = self.state_machine.get_state()
        
        # Draw overlays
        output = self.overlay.draw_all(
            frame=frame,
            rect=self.virtual_rect.rect,
            contour=contour,
            hull=hull,
            fingertip=fingertip,
            state=state,
            fps=fps,
            distance=self.state_machine.get_smoothed_distance()
        )
        
        # Add segmentation mode indicator
        mode_text = "Classical CV" if used_classical else "ML (TinyUNet)"
        cv2.putText(
            output,
            f"Mode: {mode_text}",
            (10, FRAME_HEIGHT - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        return output
    
    def _cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        self.camera.release()
        cv2.destroyAllWindows()
        print("Done!")


def main():
    """Entry point."""
    app = HandBoundaryApp()
    app.run()


if __name__ == "__main__":
    main()
