#!/usr/bin/env python3
"""
Wire Defect Detection - Image Testing
Test the inference pipeline with static images before using camera
"""

import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path

# Add system packages to path for compatibility
sys.path.insert(0, '/usr/lib/python3/dist-packages')

try:
    import onnxruntime as ort
    print("✅ ONNX Runtime available")
except ImportError:
    print("❌ ONNX Runtime not found")
    print("Install with: pip install onnxruntime")
    sys.exit(1)

class SimpleWireDetector:
    """Simple wire defect detector for testing"""
    
    def __init__(self, model_path):
        print(f"Loading model: {model_path}")
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Model settings
        self.input_size = 416
        self.crop_ratio = 0.6
        self.conf_threshold = 0.25  # Reset to normal threshold
        
        # Class info
        self.class_names = ['fail', 'pagan', 'valid']
        self.colors = {
            'fail': (0, 0, 255),    # Red
            'pagan': (255, 0, 0),   # Blue  
            'valid': (0, 255, 0)    # Green
        }
        
        print("✅ Model loaded successfully")
    
    def crop_to_roi(self, image):
        """Crop image to center 60% width"""
        h, w = image.shape[:2]
        crop_width = int(w * self.crop_ratio)
        start_x = (w - crop_width) // 2
        end_x = start_x + crop_width
        return image[:, start_x:end_x]
    
    def preprocess(self, image):
        """Preprocess image for model input"""
        # Resize to model input size
        img = cv2.resize(image, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpose to NCHW format and add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def postprocess(self, output):
        """Extract detections from YOLO model output"""
        detections = []
        
        print(f"Raw model output shape: {output.shape}")
        
        # YOLO output format: (1, 7, num_detections)
        # Where 7 = [x, y, w, h, conf, class_id, ...]
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension: (7, num_detections)
        
        print(f"After batch removal: {output.shape}")
        
        # Transpose to get (num_detections, 7)
        output = output.T  # Shape: (num_detections, 7)
        
        print(f"After transpose: {output.shape}")
        
        # Debug: Check output values
        if len(output) > 0:
            print(f"Sample detection: {output[0]}")
            confidences = output[:, 4] if len(output.shape) > 1 else []
            if len(confidences) > 0:
                print(f"Confidence range: {confidences.min():.4f} - {confidences.max():.4f}")
        
        # Extract detections above confidence threshold
        for i, detection in enumerate(output):
            if len(detection) >= 6:
                x_center, y_center, width, height, conf, class_id = detection[:6]
                
                # Debug: Print all detections above very low threshold
                if conf > 0.01:  # Very low threshold for debugging
                    print(f"Detection {i}: conf={conf:.4f}, class={int(class_id)}, bbox=[{x_center:.1f},{y_center:.1f},{width:.1f},{height:.1f}]")
                
                if conf > self.conf_threshold and int(class_id) < len(self.class_names):
                    # Convert normalized coordinates to pixel coordinates
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
        """Detect defects in a single image"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, []
        
        original_image = image.copy()
        
        # Crop to ROI
        cropped_image = self.crop_to_roi(image)
        
        # Preprocess
        input_data = self.preprocess(cropped_image)
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(None, {'images': input_data})
        inference_time = time.time() - start_time
        
        # Postprocess
        detections = self.postprocess(outputs[0])
        
        # Draw results on cropped image
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
        
        return result_image, detections, inference_time

def test_images():
    """Test detection with sample images"""
    print("=" * 60)
    print("🧪 Wire Defect Detection - Image Testing")
    print("=" * 60)
    
    # Check model file
    model_path = "models/best_cropped.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the ONNX model is in the models/ directory")
        return 1
    
    # Initialize detector
    try:
        detector = SimpleWireDetector(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Get test images
    test_dir = Path("test_images")
    if not test_dir.exists():
        print(f"❌ Test images directory not found: {test_dir}")
        return 1
    
    image_files = list(test_dir.glob("*.jpg"))[:10]  # Test first 10 images
    
    if not image_files:
        print(f"❌ No test images found in {test_dir}")
        return 1
    
    print(f"📷 Found {len(image_files)} test images")
    print()
    
    # Test each image
    total_detections = 0
    total_time = 0
    class_counts = {'fail': 0, 'pagan': 0, 'valid': 0}
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Testing: {image_path.name}")
        
        try:
            result_image, detections, inference_time = detector.detect_image(image_path)
            
            if result_image is not None:
                # Count detections
                total_detections += len(detections)
                total_time += inference_time
                
                # Update class counts
                for det in detections:
                    class_counts[det['class_name']] += 1
                
                # Print results
                print(f"  ⏱️  Inference: {inference_time*1000:.1f}ms")
                print(f"  🎯 Detections: {len(detections)}")
                
                for det in detections:
                    print(f"    - {det['class_name']}: {det['confidence']:.3f}")
                
                # Save result (optional)
                output_path = f"test_results_{image_path.name}"
                cv2.imwrite(output_path, result_image)
                print(f"  💾 Result saved: {output_path}")
                
            else:
                print("  ❌ Failed to process image")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    avg_time = total_time / len(image_files) if image_files else 0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"Images tested: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Average inference time: {avg_time*1000:.1f}ms")
    print(f"Average FPS: {avg_fps:.1f}")
    print()
    
    print("Class distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print()
    
    # Performance assessment
    if avg_fps >= 3:
        print("🎉 Performance looks good for real-time detection!")
    elif avg_fps >= 1:
        print("✅ Performance acceptable for real-time detection")
    else:
        print("⚠️  Performance may be slow for real-time detection")
    
    print()
    print("Next step: python run_camera_detection.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(test_images())
