#!/usr/bin/env python3
"""
Install OpenCV with GStreamer support on Jetson Nano
This script tries multiple approaches to get working OpenCV with GStreamer
"""

import subprocess
import sys
import os
import time

def run_command(cmd, capture_output=True, check=False):
    """Run command and return output"""
    try:
        print(f"Running: {cmd}")
        if capture_output:
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True, check=check)
            return result.returncode, "", ""
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout if hasattr(e, 'stdout') else "", e.stderr if hasattr(e, 'stderr') else str(e)
    except Exception as e:
        return 1, "", str(e)

def backup_current_opencv():
    """Backup current OpenCV installation"""
    print("=" * 60)
    print("BACKING UP CURRENT OPENCV")
    print("=" * 60)
    
    # Create backup directory
    backup_dir = "/tmp/opencv_backup"
    run_command(f"mkdir -p {backup_dir}")
    
    # Backup system OpenCV
    ret, stdout, stderr = run_command("find /usr/lib/python3/dist-packages -name '*cv2*' -type f")
    if ret == 0 and stdout:
        for file_path in stdout.strip().split('\n'):
            if file_path.strip():
                backup_path = f"{backup_dir}/{os.path.basename(file_path)}"
                run_command(f"cp '{file_path}' '{backup_path}'")
                print(f"Backed up: {file_path}")
    
    print("Backup completed")
    print()

def try_nvidia_opencv():
    """Try installing NVIDIA's OpenCV packages"""
    print("=" * 60)
    print("TRYING NVIDIA OPENCV PACKAGES")
    print("=" * 60)
    
    # Update package lists
    print("Updating package lists...")
    ret, stdout, stderr = run_command("sudo apt update")
    if ret != 0:
        print(f"Warning: apt update failed: {stderr}")
    
    # Try different NVIDIA OpenCV packages
    nvidia_packages = [
        "nvidia-opencv",
        "libopencv-dev",
        "libopencv-contrib-dev", 
        "python3-opencv"
    ]
    
    for package in nvidia_packages:
        print(f"\nTrying to install {package}...")
        ret, stdout, stderr = run_command(f"sudo apt install -y {package}")
        if ret == 0:
            print(f"Successfully installed {package}")
            
            # Test if this package provides GStreamer support
            if test_opencv_gstreamer():
                print(f"SUCCESS! {package} provides OpenCV with GStreamer support")
                return True
            else:
                print(f"{package} installed but no GStreamer support")
        else:
            print(f"Failed to install {package}: {stderr}")
    
    return False

def try_jetpack_opencv():
    """Try installing OpenCV from JetPack repositories"""
    print("=" * 60)
    print("TRYING JETPACK OPENCV")
    print("=" * 60)
    
    # Check if JetPack repositories are available
    ret, stdout, stderr = run_command("apt search jetpack")
    if ret == 0 and "jetpack" in stdout.lower():
        print("JetPack repositories found")
        
        # Try installing JetPack OpenCV
        ret, stdout, stderr = run_command("sudo apt install -y nvidia-jetpack")
        if ret == 0:
            print("JetPack installed successfully")
            if test_opencv_gstreamer():
                print("SUCCESS! JetPack provides OpenCV with GStreamer support")
                return True
        else:
            print(f"JetPack installation failed: {stderr}")
    else:
        print("JetPack repositories not found")
    
    return False

def try_compile_opencv():
    """Try compiling OpenCV with GStreamer support"""
    print("=" * 60)
    print("COMPILING OPENCV WITH GSTREAMER")
    print("=" * 60)
    print("WARNING: This will take 2-4 hours on Jetson Nano")
    print("Make sure you have at least 4GB free space and stable power supply")
    
    response = input("Do you want to continue with compilation? (y/N): ")
    if response.lower() != 'y':
        print("Compilation cancelled")
        return False
    
    # Install build dependencies
    print("Installing build dependencies...")
    deps = [
        "build-essential", "cmake", "git", "pkg-config",
        "libjpeg-dev", "libtiff5-dev", "libpng-dev",
        "libavcodec-dev", "libavformat-dev", "libswscale-dev",
        "libgtk2.0-dev", "libcanberra-gtk-module", "libcanberra-gtk3-module",
        "libgstreamer1.0-dev", "libgstreamer-plugins-base1.0-dev",
        "python3-dev", "python3-numpy", "python3-pip"
    ]
    
    for dep in deps:
        print(f"Installing {dep}...")
        ret, stdout, stderr = run_command(f"sudo apt install -y {dep}")
        if ret != 0:
            print(f"Warning: Failed to install {dep}")
    
    # Create build directory
    build_dir = "/tmp/opencv_build"
    run_command(f"rm -rf {build_dir}")
    run_command(f"mkdir -p {build_dir}")
    
    # Download OpenCV source
    print("Downloading OpenCV source...")
    os.chdir(build_dir)
    
    ret, stdout, stderr = run_command("git clone --depth 1 --branch 4.5.5 https://github.com/opencv/opencv.git")
    if ret != 0:
        print(f"Failed to download OpenCV: {stderr}")
        return False
    
    ret, stdout, stderr = run_command("git clone --depth 1 --branch 4.5.5 https://github.com/opencv/opencv_contrib.git")
    if ret != 0:
        print("Warning: Failed to download OpenCV contrib")
    
    # Configure build
    print("Configuring OpenCV build...")
    os.chdir(f"{build_dir}/opencv")
    run_command("mkdir -p build")
    os.chdir(f"{build_dir}/opencv/build")
    
    cmake_cmd = """
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_GSTREAMER=ON \
          -D WITH_LIBV4L=ON \
          -D BUILD_opencv_python3=ON \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -D PYTHON3_EXECUTABLE=$(which python3) \
          -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
          -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
          ..
    """
    
    ret, stdout, stderr = run_command(cmake_cmd)
    if ret != 0:
        print(f"CMake configuration failed: {stderr}")
        return False
    
    # Check if GStreamer was found
    if "GStreamer" in stdout and "YES" in stdout:
        print("GStreamer support will be included in build")
    else:
        print("Warning: GStreamer support may not be included")
    
    # Compile (this takes a long time)
    print("Compiling OpenCV (this will take 2-4 hours)...")
    print("You can monitor progress in another terminal with: htop")
    
    # Use single thread to avoid memory issues on Jetson Nano
    ret, stdout, stderr = run_command("make -j1", capture_output=False)
    if ret != 0:
        print("Compilation failed")
        return False
    
    # Install
    print("Installing compiled OpenCV...")
    ret, stdout, stderr = run_command("sudo make install")
    if ret != 0:
        print(f"Installation failed: {stderr}")
        return False
    
    # Update library cache
    run_command("sudo ldconfig")
    
    print("OpenCV compilation and installation completed")
    return test_opencv_gstreamer()

def test_opencv_gstreamer():
    """Test if current OpenCV has GStreamer support"""
    print("\nTesting OpenCV GStreamer support...")
    
    test_script = '''
import cv2
import sys

try:
    # Check build information
    build_info = cv2.getBuildInformation()
    has_gstreamer_build = "GStreamer:" in build_info and "YES" in [line for line in build_info.split("\\n") if "GStreamer:" in line][0]
    
    # Test practical GStreamer support
    test_cap = cv2.VideoCapture("videotestsrc num-buffers=1 ! appsink")
    has_gstreamer_test = test_cap.isOpened()
    test_cap.release()
    
    print(f"GStreamer in build: {has_gstreamer_build}")
    print(f"GStreamer test: {has_gstreamer_test}")
    
    if has_gstreamer_build and has_gstreamer_test:
        print("SUCCESS: OpenCV has working GStreamer support")
        sys.exit(0)
    else:
        print("FAILED: OpenCV lacks GStreamer support")
        sys.exit(1)
        
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
'''
    
    ret, stdout, stderr = run_command(f'python3 -c "{test_script}"')
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")
    
    return ret == 0

def main():
    print("OpenCV with GStreamer Installation Script for Jetson Nano")
    print("This script will try multiple approaches to install working OpenCV")
    print()
    
    # Test current OpenCV first
    print("Testing current OpenCV...")
    if test_opencv_gstreamer():
        print("Current OpenCV already has GStreamer support!")
        return 0
    
    print("Current OpenCV lacks GStreamer support. Trying installation methods...")
    print()
    
    # Backup current installation
    backup_current_opencv()
    
    # Try different installation methods
    methods = [
        ("NVIDIA OpenCV packages", try_nvidia_opencv),
        ("JetPack OpenCV", try_jetpack_opencv),
        ("Compile from source", try_compile_opencv)
    ]
    
    for method_name, method_func in methods:
        print(f"\n{'='*60}")
        print(f"TRYING: {method_name}")
        print(f"{'='*60}")
        
        try:
            if method_func():
                print(f"\nSUCCESS! {method_name} worked!")
                print("OpenCV now has GStreamer support")
                print("\nNext steps:")
                print("1. Test camera: python run_camera_detection.py --source 0")
                print("2. If camera works, you're all set!")
                return 0
        except Exception as e:
            print(f"ERROR in {method_name}: {e}")
        
        print(f"{method_name} failed, trying next method...")
    
    print("\nAll installation methods failed.")
    print("You may need to:")
    print("1. Check your JetPack version")
    print("2. Ensure you have adequate storage space")
    print("3. Try manual compilation with different settings")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
