#!/usr/bin/env python3
"""
Debug bounding box coordinates issue
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

def debug_bbox_coordinates():
    """Debug bounding box coordinate conversion"""
    print("🔍 Debugging Bounding Box Coordinates")
    print("=" * 50)
    
    # Load model
    model_path = "models/best_cropped.onnx"
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Load test image
    test_dir = Path("test_images")
    image_files = list(test_dir.glob("*.jpg"))
    if not image_files:
        print("❌ No test images found")
        return
    
    image_path = image_files[0]
    print(f"Testing with: {image_path.name}")
    
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    print(f"Original image: {w}x{h}")
    
    # Crop to ROI (60% center)
    crop_width = int(w * 0.6)
    start_x = (w - crop_width) // 2
    end_x = start_x + crop_width
    cropped = image[:, start_x:end_x]
    print(f"Cropped image: {cropped.shape[1]}x{cropped.shape[0]}")
    
    # Resize to 416x416
    resized = cv2.resize(cropped, (416, 416))
    print(f"Resized image: {resized.shape[1]}x{resized.shape[0]}")
    
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
    
    print(f"\nModel output shape: {output.shape}")
    
    # Find detections with confidence > 0.25
    detections = []
    for i, detection in enumerate(output):
        if len(detection) >= 6:
            x_center, y_center, width, height, conf, class_id = detection[:6]
            
            if conf > 0.25:
                print(f"\nDetection {i}:")
                print(f"  Raw values: x_center={x_center:.3f}, y_center={y_center:.3f}, width={width:.3f}, height={height:.3f}")
                print(f"  Confidence: {conf:.3f}, Class: {int(class_id)}")
                
                # Check if coordinates are normalized (0-1) or absolute pixels
                if x_center <= 1.0 and y_center <= 1.0 and width <= 1.0 and height <= 1.0:
                    print("  → Coordinates appear to be NORMALIZED (0-1)")
                    # Convert normalized to pixels
                    x1 = int((x_center - width/2) * 416)
                    y1 = int((y_center - height/2) * 416)
                    x2 = int((x_center + width/2) * 416)
                    y2 = int((y_center + height/2) * 416)
                else:
                    print("  → Coordinates appear to be ABSOLUTE pixels")
                    # Use coordinates directly
                    x1 = int(x_center - width/2)
                    y1 = int(y_center - height/2)
                    x2 = int(x_center + width/2)
                    y2 = int(y_center + height/2)
                
                print(f"  Converted bbox: [{x1},{y1},{x2},{y2}]")
                
                # Validate bbox
                if x1 >= 0 and y1 >= 0 and x2 <= 416 and y2 <= 416 and x2 > x1 and y2 > y1:
                    print("  ✅ Valid bbox")
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': int(class_id)
                    })
                else:
                    print("  ❌ Invalid bbox")
    
    print(f"\nValid detections: {len(detections)}")
    
    # Draw and save result
    if detections:
        result_image = resized.copy()
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            class_id = det['class_id']
            
            # Draw bounding box
            color = (0, 0, 255) if class_id == 0 else (255, 0, 0) if class_id == 1 else (0, 255, 0)
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"class_{class_id}: {conf:.2f}"
            cv2.putText(result_image, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            print(f"Drew bbox {i+1}: {bbox} with color {color}")
        
        # Save result
        output_path = f"debug_bbox_{image_path.name}"
        cv2.imwrite(output_path, result_image)
        print(f"\n✅ Saved result: {output_path}")
        
        # Also save original cropped image for comparison
        cv2.imwrite(f"debug_original_{image_path.name}", cropped)
        print(f"✅ Saved original: debug_original_{image_path.name}")
    else:
        print("❌ No valid detections found")

if __name__ == "__main__":
    debug_bbox_coordinates()
