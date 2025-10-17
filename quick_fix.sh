#!/bin/bash

echo "=============================================="
echo "  Quick Fix for libatlas-base-dev Error"
echo "=============================================="
echo ""

# Fix the immediate libatlas issue
echo "Fixing libatlas-base-dev dependency issue..."

# Update package lists
sudo apt update

# Install alternative math libraries
echo "Installing OpenBLAS as alternative to ATLAS..."
sudo apt install -y \
    libopenblas-dev \
    libopenblas-base \
    libblas-dev \
    liblapack-dev

# Install other essential dependencies
echo "Installing other essential dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    cmake \
    build-essential \
    pkg-config

# Install camera libraries
echo "Installing camera libraries..."
sudo apt install -y \
    python3-picamera2 \
    python3-libcamera

echo ""
echo "✅ Quick fix complete!"
echo ""
echo "Now you can continue with the setup:"
echo "  1. Run: ./setup_rpi.sh"
echo "  2. Or manually: python3 -m venv venv && source venv/bin/activate"
echo "  3. Then: pip install -r requirements.txt"
echo ""
