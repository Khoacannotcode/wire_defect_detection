#!/bin/bash

# ==============================================================================
# Jetson Environment Setup Script
# ==============================================================================
#
# Objective:
#   Automate the setup of a clean, non-virtual Python environment on the Jetson.
#   This script ensures that system-optimized libraries (OpenCV, TensorRT) are
#   correctly utilized by removing conflicting virtual environments and Python
#   packages. It also verifies the resulting environment and logs all output.
#
# Key Actions:
#   1.  Sets up logging to 'shipping/setup_environment.log'.
#   2.  Verifies sudo privileges.
#   3.  Checks for CUDA Toolkit (a critical prerequisite).
#   4.  Removes any conflicting 'venv/' directory from the project root.
#   5.  Installs essential system packages and ALL BUILD TOOLS (pip, cmake, build-essential, ninja-build, etc.).
#   6.  UPGRADES core Python build tools (pip, setuptools, wheel).
#   7.  Installs Python build dependencies (scikit-build).
#   8.  Installs 'opencv-contrib-python-headless' to provide a complete OpenCV.
#   9.  Sets up necessary environment variables for CUDA.
#   10. Installs other required Python packages (numpy, pycuda).
#   11. Performs a final verification to ensure 'cv2.dnn' is available.
#
# Usage:
#   Navigate to the 'shipping' directory and run:
#   sudo ./setup_environment.sh
#
# ==============================================================================

# --- 1. Logging Setup ---
# All output from this script will be logged to the specified file.
LOG_DIR=$(dirname "$0") # Get the directory where the script is located
LOG_FILE="$LOG_DIR/setup_environment.log"
# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"
# Overwrite the log file on each run and redirect stdout/stderr
# Using 'tee' allows output to be visible on screen and saved to the log
exec > >(tee -i "$LOG_FILE") 2>&1

echo "=============================================================="
echo "Jetson Environment Setup Log"
echo "Starting setup at: $(date)"
echo "=============================================================="


# --- Helper Functions ---
echo_info() {
    echo "[INFO] $1"
}

echo_error() {
    echo "[ERROR] $1" >&2
}

# --- 2. Sudo Privilege Check ---
if [ "$EUID" -ne 0 ]; then
  echo_error "This script must be run with sudo privileges."
  echo "Please run as: sudo $0"
  exit 1
fi
echo_info "Sudo privileges confirmed."

# --- 3. CUDA Toolkit Prerequisite Check ---
CUDA_PATH="/usr/local/cuda"
if [ ! -d "$CUDA_PATH" ] || [ ! -f "$CUDA_PATH/bin/nvcc" ]; then
    echo_error "CUDA Toolkit not found at $CUDA_PATH."
    echo_error "Please ensure the CUDA Toolkit is installed before running this script."
    echo_error "You can typically install it via the NVIDIA SDK Manager."
    exit 1
fi
echo_info "CUDA Toolkit found at $CUDA_PATH."

# --- 4. Remove Conflicting Virtual Environment ---
# Get the directory of the currently running script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Go one level up to the project root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/venv"

if [ -d "$VENV_PATH" ]; then
    echo_info "Removing existing virtual environment at '$VENV_PATH'..."
    rm -rf "$VENV_PATH"
    if [ $? -eq 0 ]; then
        echo_info "Successfully removed '$VENV_PATH'."
    else
        echo_error "Failed to remove '$VENV_PATH'. Please remove it manually."
        exit 1
    fi
else
    echo_info "No existing '$VENV_PATH' directory found. Skipping removal."
fi

# --- 5. Install System Dependencies and Build Tools ---
echo_info "Updating package list and installing all required build tools..."
apt-get update
apt-get install -y python3-pip cmake build-essential pkg-config ninja-build
if [ $? -ne 0 ]; then
    echo_error "Failed to install system dependencies. Aborting."
    exit 1
fi
echo_info "System dependencies are up to date."

# --- 6. Upgrade Core Python Build Tools ---
echo_info "Upgrading pip, setuptools, and wheel to the latest versions..."
python3 -m pip install --upgrade pip setuptools wheel
if [ $? -ne 0 ]; then
    echo_error "Failed to upgrade core Python build tools. Aborting."
    exit 1
fi
echo_info "Successfully upgraded core Python build tools."

# --- 7. Install Python Build Dependencies ---
echo_info "Installing Python build dependencies (scikit-build)..."
pip3 install scikit-build
if [ $? -ne 0 ]; then
    echo_error "Failed to install Python build dependencies. Aborting."
    exit 1
fi
echo_info "Successfully installed Python build dependencies."

# --- 8. Install OpenCV with Contrib Modules ---
# The system's OpenCV is unreliable. We will install a complete version from pip.
# 'headless' is used to avoid installing GUI dependencies on a server.
echo_info "Installing 'opencv-contrib-python-headless' to ensure 'dnn' module is available..."
pip3 install opencv-contrib-python-headless
if [ $? -ne 0 ]; then
    echo_error "Failed to install opencv-contrib-python-headless. Aborting."
    exit 1
fi
echo_info "Successfully installed opencv-contrib-python-headless."

# --- 9. Set CUDA Environment Variables ---
echo_info "Exporting CUDA environment variables for the current session..."
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
echo_info "CUDA_HOME set to $CUDA_HOME"
echo_info "PATH updated."
echo_info "LD_LIBRARY_PATH updated."


# --- 10. Install Other Required Python Packages ---
echo_info "Installing other required Python packages (numpy, pycuda)..."
pip3 install numpy
if [ $? -ne 0 ]; then
    echo_error "Failed to install numpy. Aborting."
    exit 1
fi

# PyCUDA installation requires CUDA paths
pip3 install pycuda
if [ $? -ne 0 ]; then
    echo_error "Failed to install pycuda. Please check CUDA installation and paths."
    exit 1
fi
echo_info "Successfully installed numpy and pycuda."


# --- 11. Final Verification ---
echo_info "Performing final verification to ensure 'cv2.dnn' module is now available..."
OPENCV_CHECK_RESULT=$(python3 -c "import cv2; print(hasattr(cv2, 'dnn'))" 2>/dev/null)

if [ "$OPENCV_CHECK_RESULT" != "True" ]; then
    echo_error "FINAL VERIFICATION FAILED. The 'cv2.dnn' module is still not available after installation."
    echo_error "This indicates a critical problem with the Python environment or pip installation."
    exit 1
fi
echo_info "Final verification successful. 'cv2.dnn' module is available."

# --- 12. Make Script Executable and Final Steps ---
chmod +x "$0"
echo_info "Environment setup script completed successfully."
echo_info "A detailed log has been saved to: $LOG_FILE"
echo_info "You are now using the system's Python 3 environment, configured for CUDA."

exit 0
