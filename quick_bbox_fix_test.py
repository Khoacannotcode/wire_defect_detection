#!/usr/bin/env python3
"""
Quick test to verify bounding box fix
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

def quick_bbox_test():
    """Quick test with detailed debugging"""
    print("🔧 Quick Bounding Box Fix Test")
    print("=" * 50)
    
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
    print(f"1. Original image: {w}x{h}")
    
    # Crop to ROI (60% center)
    crop_width = int(w * 0.6)
    start_x = (w - crop_width) // 2
    end_x = start_x + crop_width
    cropped = image[:, start_x:end_x]
    print(f"2. Cropped image: {cropped.shape[1]}x{cropped.shape[0]}")
    
    # Resize to 416x416
    resized = cv2.resize(cropped, (416, 416))
    print(f"3. Resized image: {resized.shape[1]}x{resized.shape[0]}")
    
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
    
    print(f"\n4. Processing {len(output)} detections...")
    
    for i, detection in enumerate(output):
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
                    print(f"   Detection {len(detections)}: conf={conf:.3f}, class={int(class_id)}, bbox=[{x1},{y1},{x2},{y2}]")
    
    print(f"\n5. Found {len(detections)} valid detections")
    
    # Draw on resized image (416x416)
    result_image = resized.copy()
    print(f"6. Drawing on resized image: {result_image.shape[1]}x{result_image.shape[0]}")
    
    for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
        color = colors[class_id] if class_id < len(colors) else (255, 255, 255)
        
        # Draw bbox
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_names[class_id]}: {conf:.2f}"
        cv2.putText(result_image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"   Drew bbox {i+1}: [{x1},{y1},{x2},{y2}] color={color}")
    
    # Save result
    output_path = f"quick_test_{image_path.name}"
    cv2.imwrite(output_path, result_image)
    print(f"\n7. ✅ Saved result: {output_path}")
    
    # Also save original resized for comparison
    cv2.imwrite(f"quick_original_{image_path.name}", resized)
    print(f"   ✅ Saved original: quick_original_{image_path.name}")
    
    print(f"\n🎯 Test complete! Check {output_path} to see if bboxes are visible.")

if __name__ == "__main__":
    quick_bbox_test()
