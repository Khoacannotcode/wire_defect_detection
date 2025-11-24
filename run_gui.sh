#!/bin/bash

# ==============================================================================
# GUI Runner for Wire Defect Detection
# ==============================================================================
# This script launches the main GUI application.
# It assumes that the environment has been correctly set up by the
# `setup_environment.sh` script, meaning all required Python packages are
# available in the system's Python environment.
# ==============================================================================

set -e # Exit immediately if a command exits

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Navigate to the script's directory to ensure relative paths work correctly
cd "$SCRIPT_DIR"

echo "[INFO] Launching the GUI application..."
echo "[INFO] Using system Python environment."

# Run the main GUI application
python3 gui_detection_runner.py

echo "[INFO] Application finished."

