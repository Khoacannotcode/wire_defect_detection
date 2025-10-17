#!/usr/bin/env python3
"""
Final test to verify bounding box visualization works correctly
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

def final_bbox_test():
    """Final test with exact same logic as fixed test script"""
    print("🎯 Final Bounding Box Test")
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
    
    # Parse YOLO output
    if len(output.shape) == 3:
        output = output[0]  # Remove batch dimension
    output = output.T  # Transpose to (num_detections, 7)
    
    print(f"Output shape: {output.shape}")
    
    # Find detections
    detections = []
    conf_threshold = 0.25
    class_names = ['fail', 'pagan', 'valid']
    colors = {
        'fail': (0, 0, 255),    # Red
        'pagan': (255, 0, 0),   # Blue  
        'valid': (0, 255, 0)    # Green
    }
    
    for i, detection in enumerate(output):
        if len(detection) >= 6:
            x_center, y_center, width, height, conf, class_id = detection[:6]
            
            if conf > conf_threshold and int(class_id) < len(class_names):
                # Convert center+size to x1,y1,x2,y2
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                # Ensure bbox is within bounds
                x1 = max(0, min(x1, 416))
                y1 = max(0, min(y1, 416))
                x2 = max(0, min(x2, 416))
                y2 = max(0, min(y2, 416))
                
                # Validate bbox
                if x2 > x1 and y2 > y1:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_name': class_names[int(class_id)]
                    })
                    print(f"Detection {len(detections)}: {class_names[int(class_id)]} conf={conf:.3f} bbox=[{x1},{y1},{x2},{y2}]")
    
    # Draw results
    result_image = resized.copy()
    
    for i, det in enumerate(detections):
        bbox = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']
        color = colors[class_name]
        
        # Draw bounding box
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(result_image, label, (bbox[0], bbox[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"Drew bbox {i+1}: {bbox} with color {color}")
    
    # Save result
    output_path = f"final_test_{image_path.name}"
    cv2.imwrite(output_path, result_image)
    print(f"\n✅ Saved result: {output_path}")
    print(f"Found {len(detections)} detections")
    
    # Also save original for comparison
    cv2.imwrite(f"final_original_{image_path.name}", resized)
    print(f"✅ Saved original: final_original_{image_path.name}")
    
    print("\nCheck the saved images to verify bboxes are visible!")

if __name__ == "__main__":
    final_bbox_test()
