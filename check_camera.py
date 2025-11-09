#!/usr/bin/env python3
"""
Comprehensive Camera Check for Jetson Nano

This script systematically checks for available cameras using various OpenCV backends
and provides diagnostic information. It's designed to identify both USB (V4L2)
and CSI cameras connected to a Jetson device.
"""

import cv2
import subprocess

def get_csi_pipeline(capture_width=1280, capture_height=720, framerate=30):
    """
    Returns the GStreamer pipeline string for a CSI camera on Jetson.
    """
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        "nvvidconv flip-method=0 ! "
        f"video/x-raw, width=(int){capture_width}, height=(int){capture_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

def check_v4l2_devices():
    """
    Lists devices found by the v4l2-ctl command-line tool.
    This is useful for identifying USB webcams.
    """
    print("=" * 60)
    print("1. Checking for V4L2 devices (e.g., USB Webcams)...")
    print("-" * 60)
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print(result.stdout.decode('utf-8'))
        print("NOTE: If you see devices listed, they should be accessible via an index like /dev/video0.\n")
    except FileNotFoundError:
        print("  'v4l2-ctl' command not found. Is 'v4l-utils' installed?")
        print("  You can install it with: sudo apt install v4l-utils\n")
    except subprocess.CalledProcessError as e:
        print("  'v4l2-ctl --list-devices' returned an error. No V4L2 devices found or another issue occurred.")
        print(f"  Error details: {e.stderr.decode('utf-8')}\n")

def test_camera_indices(max_indices=10):
    """
    Tries to open cameras by index (0, 1, 2, ...) using the default backend.
    """
    print("=" * 60)
    print(f"2. Testing camera indices from 0 to {max_indices-1} (USB/V4L2)...")
    print("-" * 60)
    
    # Check OpenCV version for compatibility
    cv_version = cv2.__version__
    print(f"  OpenCV version: {cv_version}")
    
    found_cameras = []
    for i in range(max_indices):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"  [SUCCESS] Found camera at index {i}")
                print(f"    - Resolution: {int(width)}x{int(height)}")
                found_cameras.append(i)
                cap.release()
            else:
                # Print a dot for indices that don't exist to show progress
                print(f".", end='', flush=True)
        except Exception as e:
            print(f"  [ERROR] Exception at index {i}: {e}")
    
    print("\n") # Newline after the progress dots
    if not found_cameras:
        print("  No cameras found by scanning indices.\n")
    return found_cameras

def test_csi_camera():
    """
    Tries to open the CSI camera using a GStreamer pipeline.
    This is the standard method for RPi cameras on Jetson.
    """
    print("=" * 60)
    print("3. Testing for CSI Camera (e.g., Raspberry Pi Camera)...")
    print("-" * 60)
    
    pipeline = get_csi_pipeline()
    print("  Using GStreamer pipeline:")
    print(f"  {pipeline}\n")
    
    try:
        cap = cv2.VideoCapture(pipeline)
        
        if cap.isOpened():
            print("  [SUCCESS] CSI Camera opened successfully via GStreamer.")
            # Read a frame to get resolution
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"    - Resolution: {width}x{height}")
            else:
                print("    - NOTE: Camera opened but failed to capture a frame.")
            cap.release()
            return True
        else:
            print("  [FAIL] Could not open CSI camera via GStreamer.")
            cap.release()
            return False
    except Exception as e:
        print(f"  [ERROR] Exception while testing CSI camera: {e}")
        return False

def main():
    print("============================================================")
    print("            Comprehensive Camera Check for Jetson")
    print("============================================================")
    print("This script will help diagnose camera connectivity issues.\n")
    
    # 1. List V4L2 devices
    check_v4l2_devices()
    
    # 2. Test indices
    found_v4l2 = test_camera_indices()
    
    # 3. Test CSI
    found_csi = test_csi_camera()
    
    # 4. Final summary and troubleshooting
    print("=" * 60)
    print("SUMMARY & NEXT STEPS")
    print("-" * 60)
    
    if not found_v4l2 and not found_csi:
        print("❌ No cameras were detected by any method.")
        print("\nTroubleshooting tips:")
        print("  1. Physical Connection: Double-check that the camera is securely connected.")
        print("     - For CSI cameras, ensure the ribbon cable is inserted correctly on both ends (not upside down).")
        print("     - For USB cameras, try a different USB port.")
        print("  2. Power Supply: Ensure your Jetson has an adequate power supply (e.g., 5V/4A).")
        print("  3. Reboot: A simple reboot can sometimes resolve camera detection issues.")
        print("  4. Kernel Drivers: Your camera sensor might not be supported by the default kernel.")
        print("     - The RPi Camera v2 (IMX219) IS supported by default.")
        print("     - The RPi Camera v3 (IMX708) IS NOT supported by default and requires kernel patching.")
    else:
        print("✅ Camera(s) detected!")
        if found_csi:
            print("  - Your CSI camera is working correctly.")
            print("    In your code, use the GStreamer pipeline to access it.")
        if found_v4l2:
            print(f"  - Your USB camera(s) at index/indices {found_v4l2} are working.")
            print("    In your code, use the corresponding index (e.g., cv2.VideoCapture(0)) to access it.")
    
    print("\nCheck complete.\n")

if __name__ == "__main__":
    main()
