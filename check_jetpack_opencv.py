#!/usr/bin/env python3
"""
Check JetPack version and available OpenCV packages with GStreamer support
"""

import subprocess
import sys
import os
import re

def run_command(cmd, capture_output=True):
    """Run command and return output"""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True)
            return result.returncode, "", ""
    except Exception as e:
        return 1, "", str(e)

def check_jetpack_version():
    """Check JetPack version"""
    print("=" * 60)
    print("CHECKING JETPACK VERSION")
    print("=" * 60)
    
    # Check nvidia-jetpack package
    ret, stdout, stderr = run_command("sudo apt show nvidia-jetpack 2>/dev/null")
    if ret == 0 and stdout:
        print("JetPack package info:")
        for line in stdout.split('\n'):
            if any(keyword in line.lower() for keyword in ['version', 'description']):
                print(f"  {line}")
    else:
        print("nvidia-jetpack package not found")
    
    # Check L4T version
    if os.path.exists('/etc/nv_tegra_release'):
        with open('/etc/nv_tegra_release', 'r') as f:
            l4t_info = f.read().strip()
            print(f"L4T Release: {l4t_info}")
    
    # Check CUDA version
    ret, stdout, stderr = run_command("nvcc --version 2>/dev/null")
    if ret == 0 and stdout:
        cuda_line = [line for line in stdout.split('\n') if 'release' in line.lower()]
        if cuda_line:
            print(f"CUDA: {cuda_line[0].strip()}")
    
    print()

def check_opencv_packages():
    """Check available OpenCV packages"""
    print("=" * 60)
    print("CHECKING AVAILABLE OPENCV PACKAGES")
    print("=" * 60)
    
    # Search for OpenCV packages
    ret, stdout, stderr = run_command("apt search libopencv 2>/dev/null")
    if ret == 0 and stdout:
        opencv_packages = []
        for line in stdout.split('\n'):
            if 'libopencv' in line and ('dev' in line or 'python' in line):
                opencv_packages.append(line.strip())
        
        if opencv_packages:
            print("Available OpenCV packages:")
            for pkg in opencv_packages[:10]:  # Show first 10
                print(f"  {pkg}")
        else:
            print("No OpenCV packages found")
    
    # Check installed OpenCV packages
    ret, stdout, stderr = run_command("dpkg -l | grep opencv")
    if ret == 0 and stdout:
        print("\nInstalled OpenCV packages:")
        for line in stdout.split('\n'):
            if line.strip():
                print(f"  {line}")
    
    print()

def check_nvidia_repositories():
    """Check NVIDIA repositories"""
    print("=" * 60)
    print("CHECKING NVIDIA REPOSITORIES")
    print("=" * 60)
    
    # Check apt sources
    ret, stdout, stderr = run_command("grep -r nvidia /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null")
    if ret == 0 and stdout:
        print("NVIDIA repositories found:")
        for line in stdout.split('\n'):
            if line.strip() and not line.startswith('#'):
                print(f"  {line}")
    else:
        print("No NVIDIA repositories found")
    
    print()

def check_conda_opencv():
    """Check if conda is available and has OpenCV with GStreamer"""
    print("=" * 60)
    print("CHECKING CONDA OPENCV OPTIONS")
    print("=" * 60)
    
    # Check if conda is installed
    ret, stdout, stderr = run_command("which conda")
    if ret == 0:
        print("Conda is available")
        
        # Check conda-forge opencv
        ret, stdout, stderr = run_command("conda search -c conda-forge opencv 2>/dev/null")
        if ret == 0 and stdout:
            print("Conda-forge OpenCV packages available:")
            opencv_lines = [line for line in stdout.split('\n') if 'opencv' in line][:5]
            for line in opencv_lines:
                print(f"  {line}")
        else:
            print("Could not search conda-forge opencv")
    else:
        print("Conda not available")
        print("Consider installing Miniconda for more OpenCV options")
    
    print()

def check_pip_opencv_options():
    """Check pip OpenCV packages with GStreamer support"""
    print("=" * 60)
    print("CHECKING PIP OPENCV OPTIONS")
    print("=" * 60)
    
    # Check available pip opencv packages
    ret, stdout, stderr = run_command("pip search opencv 2>/dev/null || echo 'pip search not available'")
    if "not available" not in stdout:
        print("Pip OpenCV packages:")
        opencv_lines = [line for line in stdout.split('\n') if 'opencv' in line.lower()][:5]
        for line in opencv_lines:
            print(f"  {line}")
    else:
        print("pip search not available")
    
    # Check for specialized packages
    print("\nSpecialized OpenCV packages to consider:")
    print("  - opencv-contrib-python (may have more features)")
    print("  - opencv-python-headless (for servers)")
    print("  - Pre-compiled wheels from NVIDIA or community")
    
    print()

def provide_recommendations():
    """Provide recommendations based on findings"""
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    print("Based on the analysis above, here are the recommended approaches:")
    print()
    
    print("OPTION 1 - NVIDIA Pre-built (Recommended):")
    print("   sudo apt update")
    print("   sudo apt install nvidia-opencv")
    print("   # OR")
    print("   sudo apt install libopencv-dev libopencv-contrib-dev")
    print()
    
    print("OPTION 2 - Conda-forge (If conda available):")
    print("   conda install -c conda-forge opencv")
    print()
    
    print("OPTION 3 - Rebuild from source (Last resort):")
    print("   # This will take 2-4 hours on Jetson Nano")
    print("   # Script will be provided if needed")
    print()
    
    print("NEXT STEPS:")
    print("   1. Try Option 1 first (fastest)")
    print("   2. Test camera after each attempt")
    print("   3. If all fail, we'll rebuild from source")
    print()

def main():
    print("JetPack and OpenCV Analysis for Jetson Nano")
    print("This script will analyze your system and recommend the best approach")
    print("to install OpenCV with GStreamer support.")
    print()
    
    check_jetpack_version()
    check_opencv_packages()
    check_nvidia_repositories()
    check_conda_opencv()
    check_pip_opencv_options()
    provide_recommendations()
    
    print("Analysis complete!")
    print("Please review the recommendations above and let me know which option to try first.")

if __name__ == "__main__":
    main()
