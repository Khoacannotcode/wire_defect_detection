#!/usr/bin/env python3
"""
Wire Defect Detection - Live Camera Detection
Simple real-time detection using Raspberry Pi camera
"""

import cv2
import numpy as np
import sys
import os
import time
from collections import deque

# Add system packages to path for compatibility
sys.path.insert(0, '/usr/lib/python3/dist-packages')

# Import required packages
try:
    import onnxruntime as ort
    print("✅ ONNX Runtime loaded")
except ImportError:
    print("❌ ONNX Runtime not found")
    print("Install with: pip install onnxruntime")
    sys.exit(1)

try:
    from picamera2 import Picamera2
    print("✅ Picamera2 loaded")
except ImportError:
    print("❌ Picamera2 not found")
    print("Install with: sudo apt install python3-picamera2")
    sys.exit(1)

class LiveWireDetector:
    """Simple live wire defect detector"""
    
    def __init__(self, model_path):
        print(f"Loading model: {model_path}")
        
        # Create ONNX session
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 2
        
        self.session = ort.InferenceSession(
            model_path, 
            providers=providers,
            sess_options=sess_options
        )
        
        # Settings
        self.input_size = 416
        self.crop_ratio = 0.6
        self.conf_threshold = 0.25
        
        # Class info
        self.class_names = ['fail', 'pagan', 'valid']
        
        # Statistics
        self.detection_counts = {'fail': 0, 'pagan': 0, 'valid': 0}
        self.fps_history = deque(maxlen=30)
        
        print("✅ Detector ready")
    
    def crop_to_roi(self, frame):
        """Crop frame to center ROI"""
        h, w = frame.shape[:2]
        crop_width = int(w * self.crop_ratio)
        start_x = (w - crop_width) // 2
        end_x = start_x + crop_width
        return frame[:, start_x:end_x]
    
    def preprocess(self, frame):
        """Prepare frame for inference"""
        # Resize
        img = cv2.resize(frame, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Format for ONNX (NCHW)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def detect_frame(self, frame):
        """Detect defects in a frame"""
        # Crop to ROI
        cropped_frame = self.crop_to_roi(frame)
        
        # Preprocess
        input_data = self.preprocess(cropped_frame)
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(None, {'images': input_data})
        inference_time = time.time() - start_time
        
        # Extract detections
        detections = []
        output = outputs[0]
        
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
        
        return detections, inference_time
    
    def update_stats(self, detections, inference_time):
        """Update detection statistics"""
        # Update detection counts
        for det in detections:
            self.detection_counts[det['class_name']] += 1
        
        # Update FPS
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)
    
    def print_stats(self, frame_count):
        """Print current statistics"""
        total_detections = sum(self.detection_counts.values())
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        print(f"\r[Frame {frame_count:4d}] "
              f"FPS: {avg_fps:4.1f} | "
              f"Total: {total_detections:4d} | "
              f"Fail: {self.detection_counts['fail']:3d} | "
              f"Pagan: {self.detection_counts['pagan']:3d} | "
              f"Valid: {self.detection_counts['valid']:3d}", 
              end='', flush=True)

def main():
    """Main detection loop"""
    print("=" * 60)
    print("📹 Wire Defect Detection - Live Camera")
    print("=" * 60)
    print()
    
    # Check model
    model_path = "models/best_cropped.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please ensure the ONNX model is available")
        return 1
    
    # Initialize detector
    try:
        detector = LiveWireDetector(model_path)
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return 1
    
    # Initialize camera
    print("Initializing camera...")
    try:
        picam2 = Picamera2()
        
        # Configure camera
        config = picam2.create_preview_configuration(
            main={"size": (1280, 720)},
            buffer_count=2
        )
        picam2.configure(config)
        picam2.start()
        
        print("✅ Camera initialized")
        
        # Wait for camera to stabilize
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check camera connection")
        print("  2. Enable camera: sudo raspi-config nonint do_camera 0")
        print("  3. Test camera: libcamera-hello --timeout 5000")
        return 1
    
    print()
    print("🎬 Starting live detection...")
    print("Press Ctrl+C to stop")
    print()
    
    # Main detection loop
    try:
        frame_count = 0
        
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Convert RGBA to BGR if needed
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Detect defects
            detections, inference_time = detector.detect_frame(frame)
            
            # Update statistics
            detector.update_stats(detections, inference_time)
            
            # Print stats every 10 frames
            frame_count += 1
            if frame_count % 10 == 0:
                detector.print_stats(frame_count)
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Detection stopped by user")
        
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        return 1
        
    finally:
        # Cleanup
        try:
            picam2.stop()
            print("📷 Camera stopped")
        except:
            pass
    
    # Final statistics
    print()
    print("=" * 60)
    print("📊 FINAL STATISTICS")
    print("=" * 60)
    
    total_detections = sum(detector.detection_counts.values())
    avg_fps = np.mean(detector.fps_history) if detector.fps_history else 0
    
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Average FPS: {avg_fps:.1f}")
    print()
    
    print("Detection breakdown:")
    for class_name, count in detector.detection_counts.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print()
    print("🎉 Detection session complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
