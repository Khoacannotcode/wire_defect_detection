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
#   5.  Installs essential system packages (python3-pip).
#   6.  Uninstalls pip packages that conflict with system versions (opencv-python).
#   7.  VERIFIES that the system's OpenCV has the required 'dnn' module.
#   8.  Sets up necessary environment variables for CUDA.
#   9.  Installs required Python packages (numpy, pycuda) into the system env.
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

# --- 5. Install System Dependencies ---
echo_info "Updating package list and installing python3-pip..."
apt-get update
apt-get install -y python3-pip
if [ $? -ne 0 ]; then
    echo_error "Failed to install python3-pip. Aborting."
    exit 1
fi
echo_info "System dependencies are up to date."

# --- 6. Uninstall Conflicting Pip Packages ---
# The pre-installed OpenCV on Jetson is system-optimized.
# The pip 'opencv-python' package conflicts with it and lacks CUDA support.
echo_info "Checking for and uninstalling 'opencv-python' to avoid conflicts..."
pip3 uninstall -y opencv-python
echo_info "'opencv-python' check/uninstallation complete."

# --- 7. Verify System OpenCV Installation ---
echo_info "Verifying the system's OpenCV installation for the 'dnn' module..."
OPENCV_CHECK_RESULT=$(python3 -c "import cv2; print(hasattr(cv2, 'dnn'))" 2>/dev/null)

if [ "$OPENCV_CHECK_RESULT" != "True" ]; then
    echo_error "OpenCV verification FAILED. The required 'cv2.dnn' module is missing."
    echo_error "This usually means the system's OpenCV installation is incomplete or not correctly linked."
    echo_error "Setup cannot continue. Please ensure a full version of OpenCV is installed via JetPack/SDK Manager."
    exit 1
fi
echo_info "System OpenCV verification successful. 'cv2.dnn' module found."

# --- 8. Set CUDA Environment Variables ---
echo_info "Exporting CUDA environment variables for the current session..."
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
echo_info "CUDA_HOME set to $CUDA_HOME"
echo_info "PATH updated."
echo_info "LD_LIBRARY_PATH updated."


# --- 9. Install Required Python Packages ---
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


# --- 10. Make Script Executable and Final Steps ---
chmod +x "$0"
echo_info "Environment setup script completed successfully."
echo_info "A detailed log has been saved to: $LOG_FILE"
echo_info "You are now using the system's Python 3 environment, configured for CUDA."

exit 0
