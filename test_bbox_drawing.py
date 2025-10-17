#!/usr/bin/env python3
"""
Simple test to verify bounding box drawing works
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

def test_bbox_drawing():
    """Test if bounding box drawing works"""
    print("🔧 Testing Bounding Box Drawing")
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
    
    # Test 1: Manual bbox drawing
    print("\nTest 1: Manual bbox drawing")
    test_image = resized.copy()
    
    # Draw manual bboxes
    manual_bboxes = [
        (100, 100, 200, 150, (0, 0, 255)),    # Red
        (250, 200, 350, 280, (0, 255, 0)),   # Green
        (150, 300, 250, 380, (255, 0, 0)),   # Blue
    ]
    
    for i, (x1, y1, x2, y2, color) in enumerate(manual_bboxes):
        cv2.rectangle(test_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(test_image, f"Manual {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"  Drew manual bbox {i+1}: [{x1},{y1},{x2},{y2}] color={color}")
    
    cv2.imwrite("test_manual_bboxes.jpg", test_image)
    print("  ✅ Saved: test_manual_bboxes.jpg")
    
    # Test 2: Model detection
    print("\nTest 2: Model detection")
    
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
                # Convert to bbox coordinates
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                # Validate bbox
                if x1 >= 0 and y1 >= 0 and x2 <= 416 and y2 <= 416 and x2 > x1 and y2 > y1:
                    detections.append((x1, y1, x2, y2, conf, int(class_id)))
                    print(f"  Found detection: conf={conf:.3f}, class={int(class_id)}, bbox=[{x1},{y1},{x2},{y2}]")
    
    # Draw model detections
    model_image = resized.copy()
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Red, Blue, Green
    
    for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
        color = colors[class_id] if class_id < len(colors) else (255, 255, 255)
        
        # Draw bbox
        cv2.rectangle(model_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"Model {i+1}: {conf:.2f}"
        cv2.putText(model_image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"  Drew model bbox {i+1}: [{x1},{y1},{x2},{y2}] color={color}")
    
    cv2.imwrite("test_model_bboxes.jpg", model_image)
    print("  ✅ Saved: test_model_bboxes.jpg")
    
    print(f"\nSummary:")
    print(f"  Manual bboxes: {len(manual_bboxes)}")
    print(f"  Model detections: {len(detections)}")
    print(f"  Check saved images to verify bboxes are visible!")

if __name__ == "__main__":
    test_bbox_drawing()
