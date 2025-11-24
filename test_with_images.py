#!/usr-bin/env python3
"""
Task 18, Phase 4: Verify TensorRT Performance with Image Testing
- Uses the new TRTDetector for inference.
- Measures performance and accuracy on a set of test images.
"""
import cv2
from pathlib import Path
import time
import numpy as np
from trt_inference import TRTDetector  # <-- Import the new TensorRT detector

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
ENGINE_PATH = MODELS_DIR / "best_cropped.engine"
TEST_IMAGES_DIR = SCRIPT_DIR / "test_images"
OUTPUT_DIR = SCRIPT_DIR / "test_images_output_trt"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- MODIFIED: Load class names and generate colors dynamically ---
CLASS_NAMES_PATH = MODELS_DIR / "class_names.txt"
def load_class_names(file_path):
    if not file_path.exists():
        print(f"[WARN] Class names file not found at {file_path}, colors will not be specific.")
        return []
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

CLASS_NAMES = load_class_names(CLASS_NAMES_PATH)
# Generate a consistent color for each class name
np.random.seed(42) 
CLASS_COLORS = {name: np.random.randint(0, 255, size=3).tolist() for name in CLASS_NAMES}
DEFAULT_COLOR = (255, 0, 0)

def main():
    print("=" * 60)
    print("[TEST] Wire Defect Detection - TensorRT Image Testing")
    print("=" * 60)

    # 1. Initialize the TensorRT detector
    if not ENGINE_PATH.exists():
        print(f"❌ ERROR: TensorRT engine not found at {ENGINE_PATH}")
        print("Please run trt_converter.py first.")
        return
        
    detector = TRTDetector(str(ENGINE_PATH))
    print("[OK] TensorRT Detector initialized successfully.")

    # 2. Find test images
    image_files = sorted(list(TEST_IMAGES_DIR.glob("*.jpg")))
    if not image_files:
        print(f"❌ ERROR: No test images found in {TEST_IMAGES_DIR}")
        return
    print(f"[INFO] Found {len(image_files)} test images.")

    # 3. Process each image
    total_time = 0
    total_detections = 0

    for image_path in image_files:
        print(f"\n--- Processing: {image_path.name} ---")
        frame = cv2.imread(str(image_path))
        if frame is None:
            print("  [WARN] Could not read image.")
            continue

        start_time = time.perf_counter()
        detections = detector.detect(frame)
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000
        total_time += inference_time
        total_detections += len(detections)

        print(f"  Inference time: {inference_time:.2f} ms")
        print(f"  Found {len(detections)} detections.")

        # Draw detections on the frame
        for det in detections:
            box = det['box']
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            color = CLASS_COLORS.get(det['class_name'], DEFAULT_COLOR)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save the output image
        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), frame)
        print(f"  Saved result to: {output_path}")

    # 4. Print summary
    avg_time = total_time / len(image_files)
    avg_fps = 1000 / avg_time if avg_time > 0 else 0
    print("\n" + "=" * 60)
    print("[SUMMARY] Test Complete")
    print("=" * 60)
    print(f"Images tested: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
