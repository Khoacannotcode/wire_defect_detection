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
sudo apt install -y python3-picamera2 python3-libcamera libcamera-apps

# Install picamera2 via system package (more reliable than pip)
echo "Installing picamera2 via system package..."
if ! python3 -c "import picamera2" 2>/dev/null; then
    echo "System picamera2 not working, will install via pip later..."
fi

# Create virtual environment
echo "[4/7] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "[5/7] Upgrading pip..."
pip install --upgrade pip

# Install Python packages
echo "[6/7] Installing Python packages..."

# Install basic packages first
pip install --upgrade pip setuptools wheel

# Install packages one by one with error handling
echo "Installing numpy..."
pip install "numpy>=1.21.0,<1.25.0"

echo "Installing OpenCV..."
pip install "opencv-python-headless>=4.5.0,<4.9.0"

echo "Installing Pillow..."
pip install "pillow>=8.0.0,<10.0.0"

echo "Installing tqdm..."
pip install "tqdm>=4.60.0"

# Install picamera2 if system version doesn't work
echo "Checking picamera2..."
if ! python3 -c "from picamera2 import Picamera2" 2>/dev/null; then
    echo "Installing picamera2 via pip..."
    pip install picamera2 || echo "Warning: picamera2 installation failed, using system version"
fi

# Install NCNN Python binding
echo "[7/7] Installing NCNN (this may take a few minutes)..."

# Check Python version first
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PYTHON_VERSION"

# Try multiple installation methods
echo "Attempting NCNN installation..."

# Method 1: Try pre-built wheel if available
if [ -f "ncnn-python-*.whl" ]; then
    echo "Method 1: Installing from local wheel file..."
    pip install ncnn-python-*.whl && echo "✅ NCNN installed from local wheel"
else
    echo "Method 1: Trying PyPI installation..."
    
    # Check if we can reach PyPI and if ncnn-python exists for this Python version
    if pip install ncnn-python --dry-run 2>/dev/null; then
        pip install ncnn-python --timeout 300 && echo "✅ NCNN installed from PyPI"
    else
        echo "Method 1 failed: No compatible ncnn-python wheel found"
        echo "Method 2: Trying to build from source..."
        
        # Install build dependencies
        sudo apt install -y git cmake ninja-build
        
        # Method 2: Build from source
        if command -v git >/dev/null 2>&1; then
            echo "Building NCNN from source (this may take 20-30 minutes)..."
            
            # Clone and build NCNN
            cd /tmp
            git clone --depth 1 https://github.com/Tencent/ncnn.git || {
                echo "Failed to clone NCNN repository"
                cd - && return 1
            }
            
            cd ncnn
            mkdir -p build && cd build
            
            # Configure build for Raspberry Pi
            cmake -GNinja \
                -DCMAKE_BUILD_TYPE=Release \
                -DNCNN_VULKAN=OFF \
                -DNCNN_OPENMP=ON \
                -DNCNN_BUILD_EXAMPLES=OFF \
                -DNCNN_BUILD_TOOLS=OFF \
                -DNCNN_BUILD_BENCHMARK=OFF \
                -DNCNN_PYTHON=ON \
                .. || {
                echo "CMake configuration failed"
                cd - && return 1
            }
            
            # Build (use all available cores)
            ninja -j$(nproc) || {
                echo "Build failed, trying with single thread..."
                ninja -j1 || {
                    echo "❌ NCNN build failed completely"
                    cd - && return 1
                }
            }
            
            # Install Python bindings
            cd ../python
            pip install . && echo "✅ NCNN built and installed from source"
            
            # Clean up
            cd /tmp && rm -rf ncnn
            cd -
        else
            echo "❌ Git not available, cannot build from source"
            echo ""
            echo "NCNN installation failed. You have these options:"
            echo "1. Use ONNX Runtime instead (slower but more compatible)"
            echo "2. Manually install ncnn-python following TROUBLESHOOTING.md"
            echo "3. Use a different inference engine"
            echo ""
            echo "The system will continue without NCNN..."
        fi
    fi
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
