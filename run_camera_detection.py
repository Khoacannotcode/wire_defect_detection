#!/usr/bin/env python3
"""
Task 18, Phase 4: Verify TensorRT Performance with Live Camera
- Uses the new TRTDetector for real-time inference.
- Displays FPS and detections on the live video stream.

Task 21: LiveWireDetector wrapper for GUI compatibility
- Provides interface compatible with GUI application
- Wraps TRTDetector with ROI cropping, visualization, and threshold management
"""
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque
import sys

# Import TRTDetector - when run from shipping directory, use direct import
from trt_inference import TRTDetector
from trt_converter import build_engine

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

# Visualization Standards (from visualization_standards.py)
CLASS_COLORS = {
    'NOK': (0, 165, 255),      # Orange (BGR)
    'breaks': (0, 255, 255),   # Yellow (BGR)
    'damage': (0, 0, 255),     # Red (BGR)
    'drops': (128, 128, 128),  # Gray (BGR)
    'normal': (0, 128, 0),      # Dark green (BGR)
    'shift': (255, 0, 0),      # Blue (BGR)
    # Legacy classes (backward compatibility)
    'fail': (0, 0, 255),       # Red
    'pagan': (255, 0, 0),      # Blue
    'valid': (0, 255, 0),      # Green
}
DEFAULT_COLOR = (128, 128, 128)


def open_capture(source, width=1280, height=720, fps=30, use_gstreamer=True):
    """
    Open camera capture with proper settings and fallback.
    Compatible with both USB and CSI cameras on Jetson Nano.
    Uses proven GStreamer pipeline from simple_recorder.py (tested on Jetson).
    
    Args:
        source: Camera source (int for index, or string for path)
        width: Frame width
        height: Frame height
        fps: Frames per second
        use_gstreamer: Whether to use GStreamer pipeline (default: True for Jetson)
    
    Returns:
        cv2.VideoCapture object or None if failed
    """
    try:
        source_int = int(source)
    except ValueError:
        source_int = source
    
    cap = None
    
    # Try GStreamer first (proven to work on Jetson)
    if use_gstreamer:
        try:
            # Camera tuning parameters (from simple_recorder.py - tested on Jetson)
            CAMERA_EXPOSURE_TIME = 250000  # microseconds
            CAMERA_ANALOG_GAIN = 7.0
            ISP_DIGITAL_GAIN = 1.0
            TNR_MODE = 0               # 0=Off, 1=Fast, 2=HighQuality
            TNR_STRENGTH = 0.4         # -1 (auto) to 1
            EE_MODE = 0                # 0=Off, 1=Fast, 2=HighQuality
            EE_STRENGTH = 0.2          # -1 (auto) to 1
            EXPOSURE_COMPENSATION = -2.0
            SENSOR_MODE = 5            # 1280x720 @120fps
            
            # Build camera property string (exact format from simple_recorder.py)
            camera_props = (
                f'sensor-mode={SENSOR_MODE} '
                f'exposuretimerange="{CAMERA_EXPOSURE_TIME} {CAMERA_EXPOSURE_TIME}" '
                f'gainrange="{CAMERA_ANALOG_GAIN} {CAMERA_ANALOG_GAIN}" '
                f'ispdigitalgainrange="{ISP_DIGITAL_GAIN} {ISP_DIGITAL_GAIN}" '
                f'tnr-mode={TNR_MODE} tnr-strength={TNR_STRENGTH} '
                f'ee-mode={EE_MODE} ee-strength={EE_STRENGTH} '
                f'exposurecompensation={EXPOSURE_COMPENSATION}'
            )
            
            # Build GStreamer pipeline (exact format from simple_recorder.py - proven to work)
            # Note: sensor-id is added if source_int > 0, otherwise use default
            if source_int == 0:
                pipeline = (
                    f'nvarguscamerasrc {camera_props} ! '
                    f'video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! '
                    f'nvvidconv ! video/x-raw, format=BGRx ! '
                    f'videoconvert ! video/x-raw, format=BGR ! '
                    f'appsink max-buffers=1 drop=true'
                )
            else:
                pipeline = (
                    f'nvarguscamerasrc {camera_props} sensor-id={source_int} ! '
                    f'video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! '
                    f'nvvidconv ! video/x-raw, format=BGRx ! '
                    f'videoconvert ! video/x-raw, format=BGR ! '
                    f'appsink max-buffers=1 drop=true'
                )
            
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                # Test read to verify camera actually works
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"[INFO] Camera opened successfully using GStreamer (with tuning parameters)")
                    return cap
                else:
                    print("[WARN] GStreamer opened but cannot read frames, falling back to OpenCV")
                    cap.release()
                    cap = None
            else:
                print("[WARN] GStreamer failed to open camera, falling back to OpenCV")
                cap = None
        except Exception as e:
            print(f"[WARN] GStreamer error: {e}, falling back to OpenCV")
            if cap:
                cap.release()
            cap = None
    
    # Fallback to standard OpenCV VideoCapture
    if cap is None:
        try:
            cap = cv2.VideoCapture(source_int)
            if cap.isOpened():
                # Set properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                
                # Test read to verify camera actually works
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[INFO] Camera opened successfully using OpenCV (actual resolution: {actual_width}x{actual_height})")
                    return cap
                else:
                    print("[ERROR] Camera opened but cannot read frames")
                    cap.release()
                    return None
            else:
                print(f"[ERROR] Failed to open camera source: {source_int}")
                return None
        except Exception as e:
            print(f"[ERROR] OpenCV camera error: {e}")
            return None
    
    return cap


class LiveWireDetector:
    """
    Wrapper class for TRTDetector that provides GUI-compatible interface.
    Implements ROI cropping, visualization, threshold management, and statistics tracking.
    """
    
    def __init__(self, model_path):
        """
        Initialize LiveWireDetector.
        
        Args:
            model_path: Path to model file (.onnx or .engine)
                       If .onnx, will look for corresponding .engine file
        """
        model_path = Path(model_path)
        
        # Determine engine path
        if model_path.suffix == '.onnx':
            # Look for corresponding .engine file
            engine_path = model_path.with_suffix('.engine')
            # Auto-build engine if it doesn't exist (like test_with_images.py)
            if not engine_path.exists():
                print(f"[INFO] TensorRT engine not found at '{engine_path}'.")
                print("[INFO] Attempting to build engine from ONNX model...")
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"ONNX model not found at '{model_path}'. Cannot build engine."
                    )
                
                if build_engine(model_path, engine_path):
                    print("✅ Successfully built TensorRT engine!")
                else:
                    raise RuntimeError(
                        f"Failed to build TensorRT engine from {model_path}. "
                        f"Please check TensorRT installation and ONNX model validity."
                    )
        elif model_path.suffix == '.engine':
            engine_path = model_path
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}. Use .onnx or .engine")
        
        # Initialize TRTDetector
        self.trt_detector = TRTDetector(str(engine_path))
        
        # Get class names from detector
        self.class_names = self.trt_detector.class_names
        
        # Defect classes (exclude 'normal')
        self.defect_classes = [name for name in self.class_names if name != 'normal']
        
        # Colors for visualization
        self.colors = {name: CLASS_COLORS.get(name, DEFAULT_COLOR) for name in self.class_names}
        
        # Per-class thresholds (default 0.25)
        self.class_thresholds = {name: 0.25 for name in self.class_names}
        
        # Statistics tracking
        self.stats = {
            'detection_count': 0,
            'processing_times': deque(maxlen=30),  # Keep last 30 processing times
            'fps_history': deque(maxlen=30)
        }
        
        # ROI cropping ratio (60% center width)
        self.roi_ratio = 0.6
    
    def crop_to_roi(self, image, crop_ratio=None):
        """
        Crop image to ROI (Region of Interest) - center 60% width, full height.
        
        Args:
            image: Input image (numpy array)
            crop_ratio: Crop ratio (default: 0.6 for 60% center width)
        
        Returns:
            (cropped_image, roi_info_dict)
            roi_info_dict contains: {'start_x': int, 'end_x': int, 'width': int, 'height': int}
        """
        if crop_ratio is None:
            crop_ratio = self.roi_ratio
        
        h, w = image.shape[:2]
        crop_width = int(w * crop_ratio)
        start_x = (w - crop_width) // 2
        end_x = start_x + crop_width
        
        cropped = image[:, start_x:end_x]
        
        roi_info = {
            'start_x': start_x,
            'end_x': end_x,
            'width': crop_width,
            'height': h
        }
        
        return cropped, roi_info
    
    def detect_frame(self, frame):
        """
        Detect defects in a frame with visualization.
        
        Args:
            frame: Input frame (numpy array)
        
        Returns:
            (annotated_frame, detections, processing_time)
            - annotated_frame: Frame with bounding boxes drawn
            - detections: List of detection dicts with 'box', 'confidence', 'class_name'
            - processing_time: Time taken for detection in seconds
        """
        start_time = time.perf_counter()
        
        # Crop to ROI for detection
        cropped_frame, roi_info = self.crop_to_roi(frame)
        
        # Run detection on cropped frame
        detections = self.trt_detector.detect(cropped_frame)
        
        # Filter by per-class thresholds
        filtered_detections = []
        for det in detections:
            class_name = det['class_name']
            threshold = self.class_thresholds.get(class_name, 0.25)
            if det['confidence'] >= threshold:
                # Adjust box coordinates back to original frame
                box = det['box']
                adjusted_box = [
                    box[0] + roi_info['start_x'],  # x1
                    box[1],                         # y1
                    box[2] + roi_info['start_x'],  # x2
                    box[3]                          # y2
                ]
                filtered_detections.append({
                    'box': adjusted_box,
                    'confidence': det['confidence'],
                    'class_name': det['class_name']
                })
        
        processing_time = time.perf_counter() - start_time
        
        # Draw detections on original frame
        annotated_frame = frame.copy()
        for det in filtered_detections:
            box = det['box']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Only draw defects (exclude 'normal')
            if class_name in self.defect_classes:
                color = self.colors.get(class_name, DEFAULT_COLOR)
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Draw bounding box (minimal: colored rectangle only, no label)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        return annotated_frame, filtered_detections, processing_time
    
    def set_class_thresholds(self, thresholds_dict):
        """
        Set thresholds for multiple classes at once.
        
        Args:
            thresholds_dict: Dict mapping class names to threshold values
        """
        for class_name, threshold in thresholds_dict.items():
            if class_name in self.class_names:
                self.class_thresholds[class_name] = float(threshold)
    
    def set_class_threshold(self, class_name, threshold):
        """
        Set threshold for a single class.
        
        Args:
            class_name: Name of the class
            threshold: Threshold value (0.0 to 1.0)
        """
        if class_name in self.class_names:
            self.class_thresholds[class_name] = float(max(0.0, min(1.0, threshold)))
    
    def update_stats(self, detections, processing_time):
        """
        Update internal statistics.
        
        Args:
            detections: List of detections
            processing_time: Processing time in seconds
        """
        self.stats['detection_count'] += len(detections)
        self.stats['processing_times'].append(processing_time)
        
        if processing_time > 0:
            fps = 1.0 / processing_time
            self.stats['fps_history'].append(fps)

# Export MODELS_DIR for use in other modules
__all__ = ['LiveWireDetector', 'open_capture', 'MODELS_DIR', 'CLASS_COLORS']


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time wire defect detection with TensorRT")
    parser.add_argument(
        "--model",
        default=str(MODELS_DIR / "best_cropped.engine"),
        help="Path to TensorRT engine file.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera source (e.g., 0 for default camera) or path to video file.",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Frame width for camera capture."
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Frame height for camera capture."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Initialize the TensorRT detector
    engine_path = Path(args.model)
    if not engine_path.exists():
        print(f"❌ ERROR: TensorRT engine not found at {engine_path}")
        return
        
    detector = TRTDetector(str(engine_path))
    print("[OK] TensorRT Detector initialized successfully.")
    
    # 2. Setup camera capture
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video source: {source}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"[INFO] Video source opened: {source}")

    # 3. Main loop
    fps_start_time = time.perf_counter()
    fps_frame_count = 0
    display_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream or camera disconnected.")
            break

        # Run detection
        detections = detector.detect(frame)

        # Calculate FPS
        fps_frame_count += 1
        if time.perf_counter() - fps_start_time >= 1.0:
            display_fps = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.perf_counter()

        # Draw detections and FPS on the frame
        for det in detections:
            box = det['box']
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            color = CLASS_COLORS.get(det['class_name'], DEFAULT_COLOR)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"FPS: {display_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("TensorRT Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Cleanup complete.")

if __name__ == "__main__":
    main()
