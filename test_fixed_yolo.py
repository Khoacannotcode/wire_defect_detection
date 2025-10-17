#!/usr/bin/env python3
"""
Fixed Wire Defect Detection - Image Testing
Corrected for proper YOLO output format
"""

import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path

# Add system packages to path
sys.path.insert(0, '/usr/lib/python3/dist-packages')

try:
    import onnxruntime as ort
except ImportError:
    print("❌ ONNX Runtime not found")
    sys.exit(1)

class FixedWireDetector:
    """Fixed wire defect detector with correct YOLO output parsing"""
    
    def __init__(self, model_path):
        print(f"Loading model: {model_path}")
        
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Model settings
        self.input_size = 416
        self.crop_ratio = 0.6
        self.conf_threshold = 0.25
        
        # Class info
        self.class_names = ['fail', 'pagan', 'valid']
        self.colors = {
            'fail': (0, 0, 255),    # Red
            'pagan': (255, 0, 0),   # Blue  
            'valid': (0, 255, 0)    # Green
        }
        
        print("✅ Model loaded")
    
    def crop_to_roi(self, image):
        """Crop image to center 60% width"""
        h, w = image.shape[:2]
        crop_width = int(w * self.crop_ratio)
        start_x = (w - crop_width) // 2
        end_x = start_x + crop_width
        return image[:, start_x:end_x]
    
    def preprocess(self, image):
        """Preprocess image for model input"""
        img = cv2.resize(image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def postprocess_yolo(self, output):
        """Parse YOLO output format correctly"""
        detections = []
        
        print(f"Raw output shape: {output.shape}")
        
        # YOLO output format: (1, 7, num_detections)
        # Where 7 = [x, y, w, h, conf, class_id, ...]
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension: (7, num_detections)
        
        print(f"After batch removal: {output.shape}")
        
        # Transpose to get (num_detections, 7)
        output = output.T  # Shape: (num_detections, 7)
        
        print(f"After transpose: {output.shape}")
        
        # Extract detections
        for i, detection in enumerate(output):
            if len(detection) >= 6:
                x_center, y_center, width, height, conf, class_id = detection[:6]
                
                print(f"Detection {i}: conf={conf:.4f}, class={int(class_id)}, bbox=[{x_center:.1f},{y_center:.1f},{width:.1f},{height:.1f}]")
                
                if conf > self.conf_threshold and int(class_id) < len(self.class_names):
                    # Convert to pixel coordinates
                    x1 = int((x_center - width/2) * self.input_size)
                    y1 = int((y_center - height/2) * self.input_size)
                    x2 = int((x_center + width/2) * self.input_size)
                    y2 = int((y_center + height/2) * self.input_size)
                    
                    # Ensure bbox is within image bounds
                    x1 = max(0, min(x1, self.input_size))
                    y1 = max(0, min(y1, self.input_size))
                    x2 = max(0, min(x2, self.input_size))
                    y2 = max(0, min(y2, self.input_size))
                    
                    detections.append({
                        'class_id': int(class_id),
                        'class_name': self.class_names[int(class_id)],
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
                    print(f"✅ Added detection: {self.class_names[int(class_id)]} conf={conf:.3f} bbox=[{x1},{y1},{x2},{y2}]")
        
        print(f"Total detections found: {len(detections)}")
        return detections
    
    def detect_image(self, image_path):
        """Detect defects in image"""
        print(f"\nProcessing: {image_path.name}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            return None, []
        
        # Crop to ROI
        cropped_image = self.crop_to_roi(image)
        
        # Preprocess
        input_data = self.preprocess(cropped_image)
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(None, {'images': input_data})
        inference_time = time.time() - start_time
        
        # Postprocess
        detections = self.postprocess_yolo(outputs[0])
        
        # Draw results
        result_image = cropped_image.copy()
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            color = self.colors[class_name]
            
            # Draw bounding box
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(result_image, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            print(f"Drawing bbox: {bbox} with color {color}")
        
        return result_image, detections, inference_time

def main():
    """Test with corrected YOLO parsing"""
    print("🔧 Fixed Wire Defect Detection - Corrected YOLO Parsing")
    print("=" * 60)
    
    model_path = "models/best_cropped.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return 1
    
    detector = FixedWireDetector(model_path)
    
    test_dir = Path("test_images")
    image_files = list(test_dir.glob("*.jpg"))[:3]  # Test first 3 images
    
    for image_path in image_files:
        result_image, detections, inference_time = detector.detect_image(image_path)
        
        if result_image is not None:
            output_path = f"fixed_result_{image_path.name}"
            cv2.imwrite(output_path, result_image)
            print(f"✅ Saved: {output_path} with {len(detections)} detections")
        else:
            print(f"❌ Failed to process {image_path.name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
