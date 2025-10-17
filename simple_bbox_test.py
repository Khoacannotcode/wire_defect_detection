#!/usr/bin/env python3
"""
Simple test to check if bounding boxes are drawn correctly
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add system packages to path
sys.path.insert(0, '/usr/lib/python3/dist-packages')

try:
    import onnxruntime as ort
except ImportError:
    print("❌ ONNX Runtime not found")
    sys.exit(1)

def simple_bbox_test():
    """Simple test with manual bbox drawing"""
    print("🔧 Simple Bounding Box Test")
    print("=" * 40)
    
    # Load test image
    test_dir = Path("test_images")
    image_files = list(test_dir.glob("*.jpg"))
    if not image_files:
        print("❌ No test images found")
        return
    
    image_path = image_files[0]
    print(f"Testing with: {image_path.name}")
    
    # Load image
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    
    # Crop to ROI (60% center)
    crop_width = int(w * 0.6)
    start_x = (w - crop_width) // 2
    end_x = start_x + crop_width
    cropped = image[:, start_x:end_x]
    
    # Resize to 416x416
    resized = cv2.resize(cropped, (416, 416))
    
    # Create test image with manual bbox
    test_image = resized.copy()
    
    # Draw some test bounding boxes manually
    test_bboxes = [
        (100, 100, 200, 150, (0, 0, 255)),    # Red box
        (250, 200, 350, 280, (0, 255, 0)),   # Green box
        (150, 300, 250, 380, (255, 0, 0)),   # Blue box
    ]
    
    for i, (x1, y1, x2, y2, color) in enumerate(test_bboxes):
        cv2.rectangle(test_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(test_image, f"Test {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"Drew test bbox {i+1}: [{x1},{y1},{x2},{y2}] color={color}")
    
    # Save test image
    cv2.imwrite("test_manual_bboxes.jpg", test_image)
    print("✅ Saved test image: test_manual_bboxes.jpg")
    
    # Now test with model
    print("\nTesting with model...")
    
    model_path = "models/best_cropped.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Preprocess
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    nchw = np.transpose(normalized, (2, 0, 1))
    batch_input = np.expand_dims(nchw, axis=0)
    
    # Run inference
    outputs = session.run(None, {'images': batch_input})
    output = outputs[0]
    
    # Parse output
    if len(output.shape) == 3:
        output = output[0]
    output = output.T
    
    # Find detections
    detections = []
    for detection in output:
        if len(detection) >= 6:
            x_center, y_center, width, height, conf, class_id = detection[:6]
            
            if conf > 0.25:
                # Try both coordinate systems
                print(f"\nDetection: conf={conf:.3f}, class={int(class_id)}")
                print(f"Raw: x={x_center:.1f}, y={y_center:.1f}, w={width:.1f}, h={height:.1f}")
                
                # Method 1: Assume normalized coordinates
                if x_center <= 1.0 and y_center <= 1.0:
                    x1_norm = int((x_center - width/2) * 416)
                    y1_norm = int((y_center - height/2) * 416)
                    x2_norm = int((x_center + width/2) * 416)
                    y2_norm = int((y_center + height/2) * 416)
                    print(f"Normalized method: [{x1_norm},{y1_norm},{x2_norm},{y2_norm}]")
                    
                    # Draw with normalized method
                    if x2_norm > x1_norm and y2_norm > y1_norm:
                        detections.append((x1_norm, y1_norm, x2_norm, y2_norm, conf, int(class_id)))
                
                # Method 2: Assume absolute coordinates
                x1_abs = int(x_center - width/2)
                y1_abs = int(y_center - height/2)
                x2_abs = int(x_center + width/2)
                y2_abs = int(y_center + height/2)
                print(f"Absolute method: [{x1_abs},{y1_abs},{x2_abs},{y2_abs}]")
    
    # Draw model detections
    model_image = resized.copy()
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Red, Blue, Green
    
    for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
        color = colors[class_id] if class_id < len(colors) else (255, 255, 255)
        cv2.rectangle(model_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(model_image, f"Model {i+1}: {conf:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"Drew model bbox {i+1}: [{x1},{y1},{x2},{y2}] color={color}")
    
    # Save model result
    cv2.imwrite("test_model_bboxes.jpg", model_image)
    print("✅ Saved model result: test_model_bboxes.jpg")
    
    print(f"\nFound {len(detections)} valid detections")
    print("Check the saved images to see if bboxes are visible!")

if __name__ == "__main__":
    simple_bbox_test()
