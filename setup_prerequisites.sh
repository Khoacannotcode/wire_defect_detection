#!/bin/bash
# Simplified environment setup for Jetson
# This script installs all necessary build tools and Python packages EXCEPT OpenCV.
# OpenCV will be installed by a dedicated build script.

set -e # Exit immediately if a command exits with a non-zero status.

LOG_DIR=$(dirname "$0")
LOG_FILE="$LOG_DIR/setup_prerequisites.log"
exec > >(tee -i "$LOG_FILE") 2>&1

echo "=============================================================="
echo "Jetson Prerequisite Setup Log"
echo "Starting setup at: $(date)"
echo "=============================================================="

# --- Sudo Privilege Check ---
if [ "$EUID" -ne 0 ]; then
  echo "[ERROR] This script must be run with sudo privileges."
  exit 1
fi
echo "[INFO] Sudo privileges confirmed."

# --- Install System Dependencies and Build Tools ---
echo "[INFO] Updating package list and installing all required build tools..."
apt-get update
apt-get install -y python3-pip cmake build-essential pkg-config ninja-build
echo "[INFO] System dependencies are up to date."

# --- Upgrade Core Python Build Tools ---
echo "[INFO] Upgrading pip, setuptools, and wheel..."
python3 -m pip install --upgrade pip setuptools wheel
echo "[INFO] Successfully upgraded core Python build tools."

# --- Install Python Dependencies (excluding OpenCV) ---
echo "[INFO] Installing Python dependencies (numpy, pycuda)..."
pip3 install numpy pycuda
echo "[INFO] Successfully installed numpy and pycuda."

echo "[INFO] Prerequisite setup script completed successfully."
exit 0
