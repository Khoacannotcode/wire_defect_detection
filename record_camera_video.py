#!/usr/bin/env python3
"""
Wire Defect Detection - Camera Video Recording Script
Records video with current camera settings for comparison with training data.

Usage:
    python record_camera_video.py [--duration SECONDS] [--exposure MICROSECONDS] [--gain VALUE] 
                                  [--output-dir DIR] [--width WIDTH] [--height HEIGHT] [--fps FPS]

Example:
    python record_camera_video.py --duration 5 --exposure 200000 --gain 2.0
"""

import argparse
import cv2
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Import camera capture function
from run_camera_detection import open_capture, CAMERA_EXPOSURE_TIME, CAMERA_ANALOG_GAIN

# Add system packages to path for compatibility
sys.path.insert(0, '/usr/lib/python3/dist-packages')


def override_camera_settings(exposure=None, gain=None):
    """
    Temporarily override camera settings in run_camera_detection module.
    This modifies the module-level constants before open_capture() is called.
    """
    import run_camera_detection as rcd_module
    
    # Update exposure if provided
    if exposure is not None:
        rcd_module.CAMERA_EXPOSURE_TIME = exposure
        print(f"[INFO] Overriding exposure: {exposure} microseconds")
    
    # Update gain if provided
    if gain is not None:
        rcd_module.CAMERA_ANALOG_GAIN = gain
        print(f"[INFO] Overriding gain: {gain}")
    
    # Update CAMERA_PROPERTY_STRING with current values (after both may have been updated)
    final_exposure = exposure if exposure is not None else rcd_module.CAMERA_EXPOSURE_TIME
    final_gain = gain if gain is not None else rcd_module.CAMERA_ANALOG_GAIN
    
    rcd_module.CAMERA_PROPERTY_STRING = (
        f'exposuretimerange="{final_exposure} {final_exposure}" '
        f'gainrange="{final_gain} {final_gain}"'
    )


def record_video(duration=5, exposure=None, gain=None, output_dir=None, 
                width=1280, height=720, fps=30, source='0', use_gstreamer=False):
    """
    Record video from camera with specified settings.
    
    Args:
        duration: Recording duration in seconds (default: 5)
        exposure: Exposure time in microseconds (None = use default)
        gain: Analog gain (None = use default)
        output_dir: Output directory (default: shipping/recorded_videos/)
        width: Video width (default: 1280)
        height: Video height (default: 720)
        fps: Video FPS (default: 30)
        source: Camera source (default: '0')
        use_gstreamer: Use GStreamer backend (default: False)
    
    Returns:
        Path to recorded video file, or None if failed
    """
    # Override camera settings if provided
    if exposure is not None or gain is not None:
        override_camera_settings(exposure, gain)
    
    # Get actual camera settings being used
    import run_camera_detection as rcd_module
    actual_exposure = exposure if exposure is not None else rcd_module.CAMERA_EXPOSURE_TIME
    actual_gain = gain if gain is not None else rcd_module.CAMERA_ANALOG_GAIN
    
    print("=" * 60)
    print("Wire Defect Detection - Camera Video Recording")
    print("=" * 60)
    print(f"[INFO] Camera settings:")
    print(f"  - Exposure: {actual_exposure} microseconds")
    print(f"  - Gain: {actual_gain}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Duration: {duration} seconds")
    print(f"  - Source: {source}")
    print("=" * 60)
    
    # Create output directory
    if output_dir is None:
        output_dir = ROOT_DIR / 'recorded_videos'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Generate output filename with timestamp and settings
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"camera_recording_{timestamp}_exposure{actual_exposure}_gain{actual_gain}.mp4"
    output_path = output_dir / filename
    
    print(f"[INFO] Output file: {output_path}")
    
    # Open camera capture
    print("\n[INFO] Opening camera...")
    capture = open_capture(source, width, height, fps, use_gstreamer)
    
    if not capture or not capture.isOpened():
        print("[ERROR] Failed to open camera")
        return None
    
    print("[INFO] Camera opened successfully")
    
    # Get actual frame dimensions (may differ from requested)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = fps
    
    print(f"[INFO] Actual camera properties:")
    print(f"  - Resolution: {actual_width}x{actual_height}")
    print(f"  - FPS: {actual_fps}")
    
    # Setup video writer
    # Try H.264 codec first (better compatibility), fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, actual_fps, (actual_width, actual_height))
    
    # Fallback to mp4v if H264 fails
    if not video_writer.isOpened():
        print("[WARN] H264 codec not available, trying mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, actual_fps, (actual_width, actual_height))
    
    if not video_writer.isOpened():
        print("[ERROR] Failed to initialize video writer")
        capture.release()
        return None
    
    print(f"\n[INFO] Recording video for {duration} seconds...")
    print("[INFO] Press Ctrl+C to stop early")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            
            ret, frame = capture.read()
            if not ret or frame is None:
                print(f"[WARN] Failed to read frame at {elapsed:.2f}s")
                continue
            
            # Resize frame if dimensions don't match
            if frame.shape[1] != actual_width or frame.shape[0] != actual_height:
                frame = cv2.resize(frame, (actual_width, actual_height))
            
            video_writer.write(frame)
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:  # Every 30 frames
                remaining = duration - elapsed
                print(f"[INFO] Recording... {elapsed:.1f}s / {duration}s (remaining: {remaining:.1f}s)")
    
    except KeyboardInterrupt:
        print("\n[INFO] Recording stopped by user")
    
    # Cleanup
    video_writer.release()
    capture.release()
    
    elapsed_time = time.time() - start_time
    print(f"\n[INFO] Recording completed:")
    print(f"  - Duration: {elapsed_time:.2f} seconds")
    print(f"  - Frames recorded: {frame_count}")
    print(f"  - Average FPS: {frame_count / elapsed_time:.2f}")
    print(f"  - Output file: {output_path}")
    
    # Verify file was created
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  - File size: {file_size:.2f} MB")
        print(f"\n[SUCCESS] Video saved successfully!")
        return output_path
    else:
        print(f"\n[ERROR] Video file was not created")
        return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Record video from camera with specified settings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record 5 seconds with default settings
  python record_camera_video.py
  
  # Record 10 seconds with custom exposure and gain
  python record_camera_video.py --duration 10 --exposure 200000 --gain 2.0
  
  # Record to custom directory
  python record_camera_video.py --output-dir /tmp/videos
        """
    )
    
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Recording duration in seconds (default: 5.0)')
    parser.add_argument('--exposure', type=int, default=None,
                       help='Exposure time in microseconds (default: use from run_camera_detection.py)')
    parser.add_argument('--gain', type=float, default=None,
                       help='Analog gain (default: use from run_camera_detection.py)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: shipping/recorded_videos/)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Video width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Video height (default: 720)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video FPS (default: 30)')
    parser.add_argument('--source', type=str, default='0',
                       help='Camera source (default: 0)')
    parser.add_argument('--use-gstreamer', action='store_true',
                       help='Use GStreamer backend')
    
    args = parser.parse_args()
    
    # Record video
    output_path = record_video(
        duration=args.duration,
        exposure=args.exposure,
        gain=args.gain,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        source=args.source,
        use_gstreamer=args.use_gstreamer
    )
    
    if output_path:
        print(f"\n[INFO] Video ready for review: {output_path}")
        print("[INFO] Compare with training data in: learning_based/data_preparation/v2_new_classes/frames/all")
        sys.exit(0)
    else:
        print("\n[ERROR] Failed to record video")
        sys.exit(1)


if __name__ == '__main__':
    main()

