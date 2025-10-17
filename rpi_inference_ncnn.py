import cv2
import numpy as np
import sys
import time
from collections import deque

# Try to import ncnn
try:
    import ncnn
except ImportError:
    print("❌ ncnn-python not available")
    print("Please use ONNX Runtime version: python rpi_inference_onnx.py")
    sys.exit(1)

# Try to import picamera2 with fallback
try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ picamera2 not available in virtual environment")
    print("Trying to use system picamera2...")
    
    # Try to add system packages to Python path
    import site
    import subprocess
    
    try:
        result = subprocess.run([
            'python3', '-c', 
            'import site; print(site.getsitepackages()[0])'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            system_site = result.stdout.strip()
            if system_site not in sys.path:
                sys.path.insert(0, system_site)
                print(f"Added system packages: {system_site}")
        
        from picamera2 import Picamera2
        print("✅ Successfully imported system picamera2")
        
    except ImportError:
        print("❌ Still cannot import picamera2")
        print()
        print("Solutions:")
        print("1. Install in venv: pip install picamera2")
        print("2. Use system Python: deactivate && python3 rpi_inference_ncnn.py")
        print("3. Link system package: ln -s /usr/lib/python3/dist-packages/picamera2 venv/lib/python*/site-packages/")
        sys.exit(1)

class NCNNWireDefectDetector:
    """Wire Defect Detector using NCNN framework"""
    
    def __init__(self, param_path, bin_path, conf_threshold=0.25):
        # Initialize NCNN network
        self.net = ncnn.Net()
        
        # Enable ARM NEON optimization
        self.net.opt.use_packing_layout = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.use_int8_storage = True
        self.net.opt.use_int8_arithmetic = True
        
        # Use all CPU cores
        self.net.opt.num_threads = 4  # Raspberry Pi 3B has 4 cores
        
        # Load model
        print(f"Loading NCNN model...")
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        print("✓ Model loaded successfully")
        
        self.conf_threshold = conf_threshold
        self.input_size = 416
        self.crop_ratio = 0.6
        
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
        """Preprocess frame for NCNN input"""
        # Resize
        img = cv2.resize(frame, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create NCNN Mat from numpy array
        mat = ncnn.Mat.from_pixels(
            img,
            ncnn.Mat.PixelType.PIXEL_RGB,
            self.input_size,
            self.input_size
        )
        
        # Normalize (mean=[0, 0, 0], norm=[1/255, 1/255, 1/255])
        mat.substract_mean_normalize(
            [0, 0, 0],
            [1/255.0, 1/255.0, 1/255.0]
        )
        
        return mat
    
    def postprocess(self, output):
        """Postprocess NCNN output"""
        detections = []
        
        # NCNN output format: [num_boxes, 6]
        # where 6 = [x, y, w, h, confidence, class_id]
        for i in range(output.h):
            values = output.row(i)
            
            if len(values) >= 6:
                conf = values[4]
                if conf > self.conf_threshold:
                    class_id = int(values[5])
                    if class_id < len(self.class_names):
                        detections.append({
                            'class_id': class_id,
                            'class_name': self.class_names[class_id],
                            'confidence': conf
                        })
        
        return detections
    
    def run_inference(self, frame):
        """Run inference on frame using NCNN"""
        # Preprocess
        mat_in = self.preprocess(frame)
        
        # Create extractor
        ex = self.net.create_extractor()
        ex.input("images", mat_in)
        
        # Run inference
        start_time = time.time()
        ret, mat_out = ex.extract("output0")
        inference_time = time.time() - start_time
        
        # Postprocess
        detections = []
        if ret == 0:
            detections = self.postprocess(mat_out)
        
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
        
        print(f"\r[NCNN] FPS: {avg_fps:.1f} | "
              f"Total: {total} | "
              f"Fail: {self.detection_counts['fail']} | "
              f"Pagan: {self.detection_counts['pagan']} | "
              f"Valid: {self.detection_counts['valid']}", 
              end='', flush=True)

def main():
    """Main inference loop"""
    print("=== Wire Defect Detection with NCNN ===")
    print("Optimized for Raspberry Pi ARM CPU")
    print()
    
    # Initialize detector
    detector = NCNNWireDefectDetector(
        param_path='models/best_cropped.param',
        bin_path='models/best_cropped.bin',
        conf_threshold=0.25
    )
    
    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720)},
        buffer_count=2
    )
    picam2.configure(config)
    picam2.start()
    print("✓ Camera ready")
    print()
    print("Starting inference... (Press Ctrl+C to stop)")
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

if __name__ == "__main__":
    main()
