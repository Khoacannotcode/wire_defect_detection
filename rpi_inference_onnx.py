#!/usr/bin/env python3
"""
Wire Defect Detection using ONNX Runtime (Fallback for NCNN)
Alternative inference engine when NCNN is not available
"""

import cv2
import numpy as np
import time
from collections import deque
import sys
import os

try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ picamera2 not available. Please install: sudo apt install python3-picamera2")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("Installing ONNX Runtime...")
    os.system("pip install onnxruntime")
    import onnxruntime as ort

class ONNXWireDefectDetector:
    """Wire Defect Detector using ONNX Runtime"""
    
    def __init__(self, model_path, conf_threshold=0.25):
        # Initialize ONNX Runtime session
        print(f"Loading ONNX model from: {model_path}")
        
        # Configure ONNX Runtime for CPU
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 4  # Use 4 threads
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            model_path, 
            providers=providers,
            sess_options=sess_options
        )
        
        print("✅ ONNX model loaded successfully")
        
        self.conf_threshold = conf_threshold
        self.input_size = 416
        self.crop_ratio = 0.6
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Class names
        self.class_names = ['fail', 'pagan', 'valid']
        
        # Stats tracking
        self.detection_counts = {'fail': 0, 'pagan': 0, 'valid': 0}
        self.fps_queue = deque(maxlen=30)
    
    def crop_to_roi(self, frame):
        """Crop frame to center ROI (60% width)"""
        h, w = frame.shape[:2]
        crop_width = int(w * self.crop_ratio)
        start_x = (w - crop_width) // 2
        end_x = start_x + crop_width
        return frame[:, start_x:end_x]
    
    def preprocess(self, frame):
        """Preprocess frame for ONNX input"""
        # Resize
        img = cv2.resize(frame, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)   # Add batch dimension
        
        return img
    
    def postprocess(self, output):
        """Postprocess ONNX output"""
        detections = []
        
        # ONNX output format: [batch, num_boxes, 6]
        # where 6 = [x, y, w, h, confidence, class_id]
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        for detection in output:
            if len(detection) >= 6:
                conf = detection[4]
                if conf > self.conf_threshold:
                    class_id = int(detection[5])
                    if class_id < len(self.class_names):
                        detections.append({
                            'class_id': class_id,
                            'class_name': self.class_names[class_id],
                            'confidence': conf
                        })
        
        return detections
    
    def run_inference(self, frame):
        """Run inference on frame using ONNX Runtime"""
        # Preprocess
        input_data = self.preprocess(frame)
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        inference_time = time.time() - start_time
        
        # Postprocess
        detections = self.postprocess(outputs[0])
        
        return detections, inference_time
    
    def update_stats(self, detections):
        """Update detection statistics"""
        for det in detections:
            class_name = det['class_name']
            self.detection_counts[class_name] += 1
    
    def print_stats(self):
        """Print current statistics"""
        total = sum(self.detection_counts.values())
        avg_fps = np.mean(self.fps_queue) if self.fps_queue else 0
        
        print(f"\r[ONNX] FPS: {avg_fps:.1f} | "
              f"Total: {total} | "
              f"Fail: {self.detection_counts['fail']} | "
              f"Pagan: {self.detection_counts['pagan']} | "
              f"Valid: {self.detection_counts['valid']}", 
              end='', flush=True)

def main():
    """Main inference loop"""
    print("=== Wire Defect Detection with ONNX Runtime ===")
    print("Fallback inference engine for Raspberry Pi")
    print()
    
    # Check for model file
    model_path = 'models/best_cropped.onnx'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the ONNX model is in the models/ directory")
        return 1
    
    # Initialize detector
    try:
        detector = ONNXWireDefectDetector(
            model_path=model_path,
            conf_threshold=0.25
        )
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return 1
    
    # Initialize camera
    print("Initializing camera...")
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (1280, 720)},
            buffer_count=2
        )
        picam2.configure(config)
        picam2.start()
        print("✅ Camera ready")
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        return 1
    
    print()
    print("Starting inference... (Press Ctrl+C to stop)")
    print("Note: ONNX Runtime is slower than NCNN but more compatible")
    print()
    
    try:
        frame_count = 0
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Convert RGBA to BGR if needed
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Crop to ROI
            cropped_frame = detector.crop_to_roi(frame)
            
            # Run inference
            detections, inference_time = detector.run_inference(cropped_frame)
            
            # Update statistics
            detector.update_stats(detections)
            fps = 1.0 / inference_time if inference_time > 0 else 0
            detector.fps_queue.append(fps)
            
            # Print stats every 10 frames
            frame_count += 1
            if frame_count % 10 == 0:
                detector.print_stats()
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
        print("\n=== Final Statistics ===")
        detector.print_stats()
        print("\n")
        
    finally:
        picam2.stop()
        print("Camera stopped.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
