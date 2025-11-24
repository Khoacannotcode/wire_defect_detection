#!/bin/bash

# ==============================================================================
# Jetson Environment Setup Script
# ==============================================================================
#
# Objective:
#   Automate the setup of a clean, non-virtual Python environment on the Jetson.
#   This script ensures that system-optimized libraries (OpenCV, TensorRT) are
#   correctly utilized by removing conflicting virtual environments and Python
#   packages.
#
# Key Actions:
#   1.  Verifies sudo privileges.
#   2.  Checks for CUDA Toolkit (a critical prerequisite).
#   3.  Removes any conflicting 'venv/' directory.
#   4.  Installs essential system packages (python3-pip).
#   5.  Uninstalls pip packages that conflict with system versions (opencv-python).
#   6.  Sets up necessary environment variables for CUDA.
#   7.  Installs required Python packages (numpy, pycuda) into the system env.
#
# Usage:
#   ./setup_environment.sh
#
# Prerequisites:
#   - NVIDIA Jetson device.
#   - JetPack/L4T installed.
#   - CUDA Toolkit installed (typically at /usr/local/cuda).
#
# ==============================================================================

# --- Helper Functions ---
echo_info() {
    echo "[INFO] $1"
}

echo_error() {
    echo "[ERROR] $1" >&2
}

# --- 1. Sudo Privilege Check ---
if [ "$EUID" -ne 0 ]; then
  echo_error "This script must be run with sudo privileges."
  echo "Please run as: sudo $0"
  exit 1
fi
echo_info "Sudo privileges confirmed."

# --- 2. CUDA Toolkit Prerequisite Check ---
CUDA_PATH="/usr/local/cuda"
if [ ! -d "$CUDA_PATH" ] || [ ! -f "$CUDA_PATH/bin/nvcc" ]; then
    echo_error "CUDA Toolkit not found at $CUDA_PATH."
    echo_error "Please ensure the CUDA Toolkit is installed before running this script."
    echo_error "You can typically install it via the NVIDIA SDK Manager."
    exit 1
fi
echo_info "CUDA Toolkit found at $CUDA_PATH."

# --- 3. Remove Conflicting Virtual Environment ---
VENV_PATH="venv"
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

# --- 4. Install System Dependencies ---
echo_info "Updating package list and installing python3-pip..."
apt-get update
apt-get install -y python3-pip
if [ $? -ne 0 ]; then
    echo_error "Failed to install python3-pip. Aborting."
    exit 1
fi
echo_info "System dependencies are up to date."

# --- 5. Uninstall Conflicting Pip Packages ---
# The pre-installed OpenCV on Jetson is system-optimized.
# The pip 'opencv-python' package conflicts with it and lacks CUDA support.
echo_info "Checking for and uninstalling 'opencv-python' to avoid conflicts..."
pip3 uninstall -y opencv-python
echo_info "'opencv-python' check/uninstallation complete."

# --- 6. Set CUDA Environment Variables ---
echo_info "Exporting CUDA environment variables for the current session..."
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
echo_info "CUDA_HOME set to $CUDA_HOME"
echo_info "PATH updated."
echo_info "LD_LIBRARY_PATH updated."


# --- 7. Install Required Python Packages ---
echo_info "Installing required Python packages (numpy, pycuda)..."
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


# --- 8. Make Script Executable and Final Steps ---
chmod +x "$0"
echo_info "Environment setup script completed successfully."
echo_info "You are now using the system's Python 3 environment, configured for CUDA."

exit 0
