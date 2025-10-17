#!/usr/bin/env python3
"""
Final verification test for bounding box fix
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

def final_verification_test():
    """Final test to verify bounding box fix works"""
    print("🎯 Final Verification Test - Bounding Box Fix")
    print("=" * 60)
    
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
    
    print(f"Image dimensions:")
    print(f"  Original: {w}x{h}")
    print(f"  Cropped: {cropped.shape[1]}x{cropped.shape[0]}")
    print(f"  Resized: {resized.shape[1]}x{resized.shape[0]}")
    
    # Load model
    model_path = "models/best_cropped.onnx"
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
    conf_threshold = 0.25
    class_names = ['fail', 'pagan', 'valid']
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Red, Blue, Green
    
    for detection in output:
        if len(detection) >= 6:
            x_center, y_center, width, height, conf, class_id = detection[:6]
            
            if conf > conf_threshold:
                # Convert to bbox coordinates
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                # Validate bbox
                if x1 >= 0 and y1 >= 0 and x2 <= 416 and y2 <= 416 and x2 > x1 and y2 > y1:
                    detections.append((x1, y1, x2, y2, conf, int(class_id)))
    
    print(f"\nFound {len(detections)} detections:")
    for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
        print(f"  {i+1}. {class_names[class_id]} conf={conf:.3f} bbox=[{x1},{y1},{x2},{y2}]")
    
    # Test 1: Draw on resized image (416x416) - SHOULD WORK
    print(f"\nTest 1: Drawing on resized image (416x416)")
    result_resized = resized.copy()
    
    for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
        color = colors[class_id] if class_id < len(colors) else (255, 255, 255)
        cv2.rectangle(result_resized, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result_resized, f"{class_names[class_id]}: {conf:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"  Drew bbox {i+1}: [{x1},{y1},{x2},{y2}] on 416x416")
    
    cv2.imwrite("final_test_resized.jpg", result_resized)
    print("  ✅ Saved: final_test_resized.jpg")
    
    # Test 2: Draw on cropped image with coordinate scaling - FOR COMPARISON
    print(f"\nTest 2: Drawing on cropped image with scaling")
    result_cropped = cropped.copy()
    
    # Calculate scale factors
    scale_x = cropped.shape[1] / 416
    scale_y = cropped.shape[0] / 416
    
    print(f"  Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
    
    for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
        # Scale coordinates to cropped image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)
        
        color = colors[class_id] if class_id < len(colors) else (255, 255, 255)
        cv2.rectangle(result_cropped, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
        cv2.putText(result_cropped, f"{class_names[class_id]}: {conf:.2f}", (x1_scaled, y1_scaled-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"  Drew bbox {i+1}: [{x1_scaled},{y1_scaled},{x2_scaled},{y2_scaled}] on {cropped.shape[1]}x{cropped.shape[0]}")
    
    cv2.imwrite("final_test_cropped.jpg", result_cropped)
    print("  ✅ Saved: final_test_cropped.jpg")
    
    print(f"\n🎯 Verification complete!")
    print(f"Check both images:")
    print(f"  - final_test_resized.jpg (416x416) - should show bboxes")
    print(f"  - final_test_cropped.jpg ({cropped.shape[1]}x{cropped.shape[0]}) - should show bboxes")
    print(f"\nThe fix uses the resized image approach for simplicity.")

if __name__ == "__main__":
    final_verification_test()
