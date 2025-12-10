#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test ONNX model inference using Ultralytics YOLO
Compare with custom ONNX Runtime implementation
"""
import cv2
from pathlib import Path
import time
import json
import numpy as np
from ultralytics import YOLO
from visualization_standards import draw_multiple_bboxes

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CONFIG_PATH = SCRIPT_DIR / "model_config.json"
TEST_IMAGES_DIR = SCRIPT_DIR / "test_images"
OUTPUT_DIR = SCRIPT_DIR / "test_results" / "ultralytics_onnx"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_model_config():
    """Load model paths from config.json"""
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Model config not found: {MODEL_CONFIG_PATH}")
    
    with open(MODEL_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Resolve relative paths
    onnx_path = SCRIPT_DIR / config['onnx_model_path']
    class_names_path = SCRIPT_DIR / config['class_names_path']
    
    return {
        'onnx_path': onnx_path,
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
    print("[TEST] Wire Defect Detection - Ultralytics ONNX Testing")
    print("=" * 60)

    # Load config
    try:
        config = load_model_config()
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return
    
    onnx_path = config['onnx_path']
    class_names_path = config['class_names_path']
    
    if not onnx_path.exists():
        print(f"[ERROR] ONNX model not found: {onnx_path}")
        return
    
    # Load ONNX model using Ultralytics YOLO
    print(f"[INFO] Loading ONNX model with Ultralytics: {onnx_path}")
    model = YOLO(str(onnx_path))
    print("[OK] ONNX model loaded successfully")
    
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
    all_detections = []  # Store all detections for comparison

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

        # Run inference using Ultralytics YOLO.predict()
        start_time = time.perf_counter()
        results = model.predict(
            source=frame_gray_3ch,
            imgsz=[416, 256],  # Rectangular input matching training
            conf=0.25,
            verbose=False
        )
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000
        total_time += inference_time
        
        result = results[0]
        boxes = result.boxes
        num_detections = len(boxes) if boxes is not None else 0
        total_detections += num_detections

        # Store detections for comparison
        image_detections = []
        if boxes is not None:
            for box in boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                
                image_detections.append({
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class_name': class_name,
                    'class_id': class_id
                })
        
        all_detections.append({
            'image': image_path.name,
            'detections': image_detections,
            'count': num_detections
        })

        print(f"  Inference time: {inference_time:.2f} ms")
        print(f"  Found {num_detections} detections")

        # Draw detections using visualization standards
        annotated = frame.copy()
        bboxes_info = []
        for det in image_detections:
            box = det['box']
            label = "{}: {:.2f}".format(det['class_name'], det['confidence'])
            bboxes_info.append({
                'x1': int(box[0]),
                'y1': int(box[1]),
                'x2': int(box[2]),
                'y2': int(box[3]),
                'class_name': det['class_name'],
                'label': label
            })
        
        # Draw all bboxes with standards
        annotated = draw_multiple_bboxes(annotated, bboxes_info)
        
        # Save the output image
        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), annotated)
        print(f"  Saved result to: {output_path}")

    # Print summary
    if not image_files:
        print("\nNo images were processed.")
        return
        
    avg_time = total_time / len(image_files)
    avg_fps = 1000 / avg_time if avg_time > 0 else 0
    print("\n" + "=" * 60)
    print("[SUMMARY] Ultralytics ONNX Test Complete")
    print("=" * 60)
    print(f"Images tested: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Save detections to JSON for comparison
    comparison_file = OUTPUT_DIR / "detections.json"
    with open(comparison_file, 'w') as f:
        json.dump(all_detections, f, indent=2)
    print(f"Detections saved to: {comparison_file}")

if __name__ == "__main__":
    main()

