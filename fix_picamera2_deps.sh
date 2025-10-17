#!/bin/bash

echo "=============================================="
echo "  Fixing picamera2 Dependencies"
echo "=============================================="
echo ""

# Install missing system dependencies for picamera2
echo "Installing system dependencies for picamera2..."

sudo apt update

# Install libcap development headers (required for python-prctl)
echo "Installing libcap-dev..."
sudo apt install -y libcap-dev

# Install other common build dependencies
echo "Installing build dependencies..."
sudo apt install -y \
    python3-dev \
    python3-setuptools \
    build-essential \
    cmake \
    pkg-config \
    libffi-dev

# Install additional libraries that picamera2 might need
echo "Installing additional libraries..."
sudo apt install -y \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "Now try installing picamera2:"
echo "  pip install picamera2 --force-reinstall"
echo ""
echo "Or use the system version:"
echo "  deactivate"
echo "  python3 run_system_python.py"
echo ""
