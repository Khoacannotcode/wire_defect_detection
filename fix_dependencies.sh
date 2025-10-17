#!/bin/bash

echo "=============================================="
echo "  Wire Defect Detection - Dependency Fixer"
echo "  Fixes common installation issues on RPi"
echo "=============================================="
echo ""

# Function to check if package exists
package_exists() {
    apt-cache show "$1" >/dev/null 2>&1
}

# Function to install package with fallback
install_with_fallback() {
    local primary="$1"
    local fallback="$2"
    
    if package_exists "$primary"; then
        echo "Installing $primary..."
        sudo apt install -y "$primary"
    elif [ -n "$fallback" ] && package_exists "$fallback"; then
        echo "Package $primary not found, using fallback: $fallback"
        sudo apt install -y "$fallback"
    else
        echo "Warning: Neither $primary nor $fallback found, skipping..."
    fi
}

echo "Fixing common dependency issues..."

# Update package lists
echo "[1/5] Updating package lists..."
sudo apt update

# Fix broken packages
echo "[2/5] Fixing broken packages..."
sudo apt --fix-broken install -y

# Install essential build tools
echo "[3/5] Installing essential build tools..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    cmake \
    build-essential \
    pkg-config

# Install math libraries with fallbacks
echo "[4/5] Installing math libraries..."
install_with_fallback "libatlas-base-dev" "libopenblas-dev"
install_with_fallback "libopenblas-dev" "libblas-dev"
install_with_fallback "liblapack-dev" ""

# Install image libraries with fallbacks
echo "[5/5] Installing image libraries..."
install_with_fallback "libjpeg-dev" "libjpeg62-turbo-dev"
install_with_fallback "libpng-dev" "libpng12-dev"
install_with_fallback "libtiff-dev" "libtiff5-dev"

# Install OpenCV dependencies
echo "Installing OpenCV dependencies..."
sudo apt install -y \
    python3-opencv \
    libopencv-dev \
    libhdf5-dev \
    libhdf5-serial-dev

# Install camera libraries
echo "Installing camera libraries..."
sudo apt install -y \
    python3-picamera2 \
    python3-libcamera \
    libcamera-apps

# Clean up
echo "Cleaning up..."
sudo apt autoremove -y
sudo apt autoclean

echo ""
echo "=============================================="
echo "  Dependency fixing complete!"
echo "=============================================="
echo ""
echo "If you still have issues, try:"
echo "  1. Reboot: sudo reboot"
echo "  2. Update OS: sudo apt full-upgrade"
echo "  3. Check specific error messages"
echo ""
