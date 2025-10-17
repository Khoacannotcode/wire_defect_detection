#!/bin/bash

echo "=============================================="
echo "  Wire Defect Detection - Raspberry Pi Setup"
echo "  Using NCNN Framework (Optimized for ARM)"
echo "=============================================="
echo ""

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo "⚠ Warning: This script is designed for Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "[1/7] Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo "[2/7] Installing system dependencies..."

# Check OS version and install appropriate packages
if grep -q "bullseye\|bookworm" /etc/os-release; then
    echo "Detected modern Raspberry Pi OS (Bullseye/Bookworm)"
    sudo apt install -y \
        python3-pip \
        python3-venv \
        python3-opencv \
        libopencv-dev \
        libopenblas-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        cmake \
        build-essential \
        pkg-config \
        libhdf5-dev \
        libhdf5-serial-dev
else
    echo "Detected older Raspberry Pi OS"
    sudo apt install -y \
        python3-pip \
        python3-venv \
        python3-opencv \
        libopencv-dev \
        libatlas-base-dev \
        libopenblas-dev \
        libjpeg-dev \
        cmake \
        build-essential
fi

# Install camera libraries
echo "[3/7] Installing camera libraries..."
sudo apt install -y python3-picamera2 python3-libcamera

# Create virtual environment
echo "[4/7] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "[5/7] Upgrading pip..."
pip install --upgrade pip

# Install Python packages
echo "[6/7] Installing Python packages..."
pip install -r requirements.txt

# Install NCNN Python binding
echo "[7/7] Installing NCNN (this may take a few minutes)..."

# Try multiple installation methods
echo "Attempting NCNN installation..."

# Method 1: Try pre-built wheel
if [ -f "ncnn-python-*.whl" ]; then
    echo "Method 1: Installing from local wheel file..."
    pip install ncnn-python-*.whl
elif pip install ncnn-python --timeout 300 2>/dev/null; then
    echo "Method 1: Successfully installed ncnn-python from PyPI"
else
    echo "Method 1 failed, trying alternative approaches..."
    
    # Method 2: Install with specific flags
    echo "Method 2: Installing with build flags..."
    pip install ncnn-python --no-cache-dir --verbose --timeout 600 2>/dev/null || {
        
        # Method 3: Build from source with specific settings
        echo "Method 2 failed, trying Method 3: Build from source..."
        export CMAKE_ARGS="-DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=OFF"
        pip install ncnn-python --no-binary :all: --timeout 900 2>/dev/null || {
            
            # Method 4: Last resort - minimal build
            echo "Method 3 failed, trying Method 4: Minimal build..."
            pip install --upgrade pip setuptools wheel
            pip install ncnn-python --no-deps --force-reinstall 2>/dev/null || {
                echo "❌ All NCNN installation methods failed!"
                echo "You may need to install manually or use alternative inference engine."
                echo "See TROUBLESHOOTING.md for manual installation steps."
            }
        }
    }
fi

# Enable camera interface
echo ""
echo "Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Test imports
echo ""
echo "Testing imports..."
python3 -c "import cv2; import numpy; import ncnn; from picamera2 import Picamera2; print('✓ All modules imported successfully')"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To run inference:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run inference: python rpi_inference_ncnn.py"
echo ""
echo "Expected performance on Raspberry Pi 3B:"
echo "  - FPS: 3-6 FPS"
echo "  - Inference time: 170-330ms per frame"
echo ""
