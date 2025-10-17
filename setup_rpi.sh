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
# Try pre-built wheel first
if [ -f "ncnn-python-*.whl" ]; then
    echo "Installing from local wheel file..."
    pip install ncnn-python-*.whl
else
    echo "Building NCNN from source..."
    pip install ncnn-python
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
