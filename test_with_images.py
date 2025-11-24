#!/usr-bin/env python3
"""
Task 18, Phase 4: Verify TensorRT Performance with Image Testing (Re-created)
- Uses the new TRTDetector for inference.
- Measures performance and accuracy on a set of test images.
- Loads class names dynamically to ensure synchronization.
"""
import cv2
from pathlib import Path
import time
import numpy as np
from trt_inference import TRTDetector 
from trt_converter import build_engine # Import the builder

# --- Configuration using relative paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
ONNX_PATH = MODELS_DIR / "best_cropped.onnx"
ENGINE_PATH = MODELS_DIR / "best_cropped.engine"
TEST_IMAGES_DIR = SCRIPT_DIR / "test_images"
OUTPUT_DIR = SCRIPT_DIR / "test_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Load class names and generate colors dynamically ---
CLASS_NAMES_PATH = MODELS_DIR / "class_names.txt"
def load_class_names(file_path):
    try:
        with open(file_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print("[ERROR] '{}' not found. Cannot determine class names for visualization.".format(file_path))
        return []

CLASS_NAMES = load_class_names(CLASS_NAMES_PATH)
# Generate a consistent, random color for each class name
np.random.seed(42) 
CLASS_COLORS = {name: np.random.randint(0, 255, size=3).tolist() for name in CLASS_NAMES}
DEFAULT_COLOR = (255, 0, 0)

def main():
    print("=" * 60)
    print("[TEST] Wire Defect Detection - TensorRT Image Testing")
    print("=" * 60)

    # --- Auto-convert to TensorRT engine if it doesn't exist ---
    if not ENGINE_PATH.exists():
        print("[INFO] TensorRT engine not found at '{}'.".format(ENGINE_PATH))
        print("[INFO] Attempting to build engine from ONNX model...")
        if not ONNX_PATH.exists():
            print("[ERROR] ONNX model not found at '{}'. Cannot build engine.".format(ONNX_PATH))
            return
        
        if build_engine(ONNX_PATH, ENGINE_PATH):
            print("\n🎉 Successfully built TensorRT engine!")
        else:
            print("\n🔥 Failed to build TensorRT engine. Aborting.")
            return

    # 1. Initialize the TensorRT detector
    detector = TRTDetector(str(ENGINE_PATH))
    print("[OK] TensorRT Detector initialized successfully.")

    # 2. Find test images
    image_files = sorted(list(TEST_IMAGES_DIR.glob("*.jpg")))
    if not image_files:
        print("❌ ERROR: No test images found in {}".format(TEST_IMAGES_DIR))
        return
    print("[INFO] Found {} test images.".format(len(image_files)))

    # 3. Process each image
    total_time = 0
    total_detections = 0

    for image_path in image_files:
        print("\n--- Processing: {} ---".format(image_path.name))
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

        print("  Inference time: {:.2f} ms".format(inference_time))
        print("  Found {} detections.".format(len(detections)))

        # Draw detections on the frame
        for det in detections:
            box = det['box']
            label = "{}: {:.2f}".format(det['class_name'], det['confidence'])
            color = CLASS_COLORS.get(det['class_name'], DEFAULT_COLOR)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save the output image
        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), frame)
        print("  Saved result to: {}".format(output_path))

    # 4. Print summary
    if not image_files:
        print("\nNo images were processed.")
        return
        
    avg_time = total_time / len(image_files)
    avg_fps = 1000 / avg_time if avg_time > 0 else 0
    print("\n" + "=" * 60)
    print("[SUMMARY] Test Complete")
    print("=" * 60)
    print("Images tested: {}".format(len(image_files)))
    print("Total detections: {}".format(total_detections))
    print("Average inference time: {:.2f} ms".format(avg_time))
    print("Average FPS: {:.2f}".format(avg_fps))

if __name__ == "__main__":
    main()
