#!/usr/bin/env python3
"""
Quick test to check model output format and fix visualization
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

def quick_test():
    """Quick test of model output"""
    print("🔍 Quick Model Output Test")
    print("=" * 40)
    
    # Check model
    model_path = "models/best_cropped.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load model
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Get model info
    print("Model Info:")
    for input_info in session.get_inputs():
        print(f"  Input: {input_info.name}, Shape: {input_info.shape}")
    for output_info in session.get_outputs():
        print(f"  Output: {output_info.name}, Shape: {output_info.shape}")
    
    # Test with dummy data
    print("\nTesting with dummy data:")
    dummy_input = np.random.rand(1, 3, 416, 416).astype(np.float32)
    outputs = session.run(None, {'images': dummy_input})
    
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output dtype: {outputs[0].dtype}")
    print(f"Output range: {outputs[0].min():.4f} to {outputs[0].max():.4f}")
    
    # Test with real image
    print("\nTesting with real image:")
    test_dir = Path("test_images")
    if test_dir.exists():
        image_files = list(test_dir.glob("*.jpg"))
        if image_files:
            image_path = image_files[0]
            print(f"Using: {image_path.name}")
            
            # Load and preprocess image
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]
            
            # Crop to ROI (60% center)
            crop_width = int(w * 0.6)
            start_x = (w - crop_width) // 2
            end_x = start_x + crop_width
            cropped = image[:, start_x:end_x]
            
            # Resize and normalize
            resized = cv2.resize(cropped, (416, 416))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            
            # Transpose to NCHW
            nchw = np.transpose(normalized, (2, 0, 1))
            batch_input = np.expand_dims(nchw, axis=0)
            
            # Run inference
            outputs = session.run(None, {'images': batch_input})
            output = outputs[0]
            
            print(f"Real image output shape: {output.shape}")
            
            # Analyze detections
            if len(output.shape) == 3:
                output = output[0]
            
            print(f"After batch removal: {output.shape}")
            
            # Check confidence values
            confidences = output[:, 4] if len(output.shape) > 1 else []
            if len(confidences) > 0:
                print(f"Confidence range: {confidences.min():.4f} to {confidences.max():.4f}")
                print(f"Max confidence: {confidences.max():.4f}")
                
                # Count detections at different thresholds
                for threshold in [0.01, 0.05, 0.1, 0.25, 0.5]:
                    count = sum(1 for c in confidences if c > threshold)
                    print(f"  Threshold {threshold}: {count} detections")
            
            # Show sample detections
            print("\nSample detections (top 5 by confidence):")
            if len(output) > 0:
                # Sort by confidence
                sorted_indices = np.argsort(output[:, 4])[::-1]
                for i in range(min(5, len(sorted_indices))):
                    idx = sorted_indices[i]
                    det = output[idx]
                    conf = det[4]
                    class_id = int(det[5])
                    bbox = det[:4]
                    print(f"  {i+1}. conf={conf:.4f}, class={class_id}, bbox={bbox}")
        else:
            print("No test images found")
    else:
        print("No test_images directory")

if __name__ == "__main__":
    quick_test()
