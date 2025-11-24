#!/bin/bash
#
# Task 19: Jetson Environment Setup Script
# This script prepares the Jetson device for running the wire defect detection
# application by setting up a clean, non-virtual Python environment.
# It ensures that system-optimized libraries are used for maximum performance.
#
# Usage: sudo ./setup_environment.sh
#

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Safety Check ---
# Ensure the script is run with root privileges, as it installs system-wide packages.
if [ "$EUID" -ne 0 ]; then 
  echo "❌ ERROR: Please run this script with sudo."
  exit 1
fi

echo "--- Starting Jetson Environment Setup ---"

# --- 2. Remove Existing Virtual Environment ---
# Detect and remove any existing 'venv' to prevent conflicts.
echo "[INFO] Checking for and removing existing virtual environment..."
if [ -d "venv" ]; then
    echo "[INFO] Found 'venv' directory. Removing it..."
    rm -rf venv
    echo "[INFO] 'venv' directory removed successfully."
else
    echo "[INFO] No 'venv' directory found. Skipping removal."
fi

# --- 3. Install System Dependencies ---
# Update the package list and install python3-pip, which is required to manage Python packages.
echo
echo "[INFO] Updating apt package list..."
apt-get update

echo "[INFO] Installing python3-pip..."
apt-get install -y python3-pip

# --- 4. Install Python Packages (The Critical Step) ---
# Use pip to install required packages globally for the system's Python 3 interpreter.
# First, uninstall any conflicting generic OpenCV packages to ensure the system's 
# hardware-accelerated version is used.
echo
echo "[INFO] Uninstalling potentially conflicting OpenCV packages..."
python3 -m pip uninstall -y opencv-python opencv-python-headless

echo "[INFO] Installing required Python packages (numpy, pycuda)..."
python3 -m pip install numpy pycuda

# --- 5. Finalization ---
echo
echo "✅ Environment setup complete."
echo "The system is now ready for running the detection scripts."
echo "----------------------------------------"
