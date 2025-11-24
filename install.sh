#!/bin/bash
# ==============================================================================
# Main Installation Script for Wire Defect Detection on Jetson
# ==============================================================================
#
# This script orchestrates the entire setup process by executing two sub-scripts:
#
#   1. setup_prerequisites.sh: Installs all necessary build tools and Python
#      packages, preparing the system for the main installation.
#
#   2. build_opencv.sh: Downloads, compiles, and installs a full, CUDA-enabled
#      version of OpenCV from source. This is a very long process.
#
# Usage:
#   cd into this directory and run:
#   ./install.sh
#
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status.

echo "======================================================="
echo "[STEP 1/3] Setting up prerequisites..."
echo "======================================================="
# Grant execute permissions to the prerequisite script
chmod +x ./setup_prerequisites.sh
# Run the prerequisite script with sudo
sudo ./setup_prerequisites.sh

echo "======================================================="
echo "[STEP 2/3] Preparing OpenCV build script..."
echo "======================================================="
# Grant execute permissions to the build script
chmod +x ./build_opencv.sh

echo "======================================================="
echo "[STEP 3/3] Starting OpenCV build..."
echo "WARNING: This process will take a very long time (potentially hours)."
echo "Please ensure the Jetson has adequate power and cooling."
echo "======================================================="
# The build script handles its own sudo commands where necessary
./build_opencv.sh

echo "======================================================="
echo "INSTALLATION COMPLETE!"
echo "A full system reboot is highly recommended before running the application."
echo "To reboot now, run: sudo reboot"
echo "======================================================="

exit 0
