#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test TensorRT engine inference on images
Uses TRTDetector for inference - matches Jetson deployment
This helps identify differences between ONNX and TensorRT inference
"""
import cv2
from pathlib import Path
import time
import json
import numpy as np
from trt_inference import TRTDetector

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CONFIG_PATH = SCRIPT_DIR / "model_config.json"
TEST_IMAGES_DIR = SCRIPT_DIR / "test_images"
OUTPUT_DIR = SCRIPT_DIR / "test_results" / "trt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_model_config():
    """Load model paths from config.json"""
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Model config not found: {MODEL_CONFIG_PATH}")
    
    with open(MODEL_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Resolve relative paths
    engine_path = SCRIPT_DIR / config['tensorrt_engine_path']
    class_names_path = SCRIPT_DIR / config['class_names_path']
    
    return {
        'engine_path': engine_path,
        'class_names_path': class_names_path
    }

def load_class_names(file_path):
    """Load class names from file"""
    try:
        with open(file_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print("[WARN] Class names file not found, using default")
        return []

def main():
    print("=" * 60)
    print("[TEST] Wire Defect Detection - TensorRT Image Testing")
    print("=" * 60)

    # Load config
    try:
        config = load_model_config()
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return
    
    engine_path = config['engine_path']
    class_names_path = config['class_names_path']
    
    if not engine_path.exists():
        print(f"[ERROR] TensorRT engine not found: {engine_path}")
        print(f"[INFO] Please build engine first using trt_converter.py or rebuild_engine.sh")
        return
    
    # Initialize TensorRT detector
    print(f"[INFO] Loading TensorRT engine: {engine_path}")
    try:
        detector = TRTDetector(str(engine_path))
        print("[OK] TensorRT Detector initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize detector: {e}")
        return
    
    # Load class names
    class_names = load_class_names(class_names_path)
    if class_names:
        print(f"[INFO] Loaded {len(class_names)} class names")
    
    # Find test images
    image_files = sorted(list(TEST_IMAGES_DIR.glob("*.jpg")))
    if not image_files:
        print(f"[ERROR] No test images found in {TEST_IMAGES_DIR}")
        return
    print(f"[INFO] Found {len(image_files)} test images")

    # Process each image
    total_time = 0
    total_detections = 0

    for image_path in image_files:
        print(f"\n--- Processing: {image_path.name} ---")
        frame = cv2.imread(str(image_path))
        if frame is None:
            print("  [WARN] Could not read image.")
            continue

        # Convert to grayscale 3-channel (model expects 3-channel grayscale format)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 2:
            frame_gray_3ch = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_gray_3ch = frame

        # Run inference using TRTDetector
        start_time = time.perf_counter()
        detections = detector.detect(frame_gray_3ch)
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000
        total_time += inference_time
        total_detections += len(detections)

        print(f"  Inference time: {inference_time:.2f} ms")
        print(f"  Found {len(detections)} detections")

        # Draw detections on the frame
        for det in detections:
            box = det['box']
            label = "{}: {:.2f}".format(det['class_name'], det['confidence'])
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save the output image
        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), frame)
        print(f"  Saved result to: {output_path}")

    # Print summary
    if not image_files:
        print("\nNo images were processed.")
        return
        
    avg_time = total_time / len(image_files)
    avg_fps = 1000 / avg_time if avg_time > 0 else 0
    print("\n" + "=" * 60)
    print("[SUMMARY] TensorRT Test Complete")
    print("=" * 60)
    print(f"Images tested: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

