#!/usr-bin/env python3
"""
Task 18, Phase 4: Verify TensorRT Performance with Live Camera
- Uses the new TRTDetector for real-time inference.
- Displays FPS and detections on the live video stream.
"""
import cv2
import time
import argparse
from pathlib import Path
from trt_inference import TRTDetector # <-- Import the new TensorRT detector

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

# Visualization Standards (simplified)
CLASS_COLORS = {
    'NOK': (0, 0, 255), 'breaks': (0, 0, 255), 'damage': (0, 0, 255), 
    'drops': (0, 0, 255), 'shift': (0, 0, 255),
    'normal': (0, 255, 0)
}
DEFAULT_COLOR = (255, 0, 0)

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time wire defect detection with TensorRT")
    parser.add_argument(
        "--model",
        default=str(MODELS_DIR / "best_cropped.engine"),
        help="Path to TensorRT engine file.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera source (e.g., 0 for default camera) or path to video file.",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Frame width for camera capture."
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Frame height for camera capture."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Initialize the TensorRT detector
    engine_path = Path(args.model)
    if not engine_path.exists():
        print(f"❌ ERROR: TensorRT engine not found at {engine_path}")
        return
        
    detector = TRTDetector(str(engine_path))
    print("[OK] TensorRT Detector initialized successfully.")
    
    # 2. Setup camera capture
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video source: {source}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"[INFO] Video source opened: {source}")

    # 3. Main loop
    fps_start_time = time.perf_counter()
    fps_frame_count = 0
    display_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream or camera disconnected.")
            break

        # Run detection
        detections = detector.detect(frame)

        # Calculate FPS
        fps_frame_count += 1
        if time.perf_counter() - fps_start_time >= 1.0:
            display_fps = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.perf_counter()

        # Draw detections and FPS on the frame
        for det in detections:
            box = det['box']
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            color = CLASS_COLORS.get(det['class_name'], DEFAULT_COLOR)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"FPS: {display_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("TensorRT Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Cleanup complete.")

if __name__ == "__main__":
    main()
