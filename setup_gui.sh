#!/bin/bash

# ==============================================================================
# GUI Setup Script for Wire Defect Detection
# ==============================================================================
# This script checks and sets up the environment for running the GUI.
# It verifies:
# 1. Required Python packages are installed
# 2. Model files exist (ONNX and/or engine)
# 3. Auto-builds TensorRT engine if needed
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

MODELS_DIR="$SCRIPT_DIR/models"
ONNX_PATH="$MODELS_DIR/best_cropped.onnx"
ENGINE_PATH="$MODELS_DIR/best_cropped.engine"
CLASS_NAMES_PATH="$MODELS_DIR/class_names.txt"

echo "=============================================================="
echo "GUI Setup - Wire Defect Detection"
echo "=============================================================="

# --- Check Python version ---
echo "[1/5] Checking Python version..."
python3 --version || { echo "[ERROR] Python3 not found!"; exit 1; }
echo "  ✅ Python3 found"

# --- Check required Python packages ---
echo "[2/5] Checking required Python packages..."
MISSING_PACKAGES=()

check_package() {
    python3 -c "import $1" 2>/dev/null || MISSING_PACKAGES+=("$1")
}

check_package "cv2"
check_package "numpy"
check_package "PIL"
check_package "tkinter"

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "  ⚠️  Missing packages: ${MISSING_PACKAGES[*]}"
    echo "  [INFO] Install missing packages with:"
    echo "    pip3 install ${MISSING_PACKAGES[*]}"
    echo "  [INFO] Or run setup_prerequisites.sh for system-wide installation"
    read -p "  Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "  ✅ All required packages found"
fi

# --- Check TensorRT (optional, only needed for engine building) ---
echo "[3/5] Checking TensorRT..."
if python3 -c "import tensorrt" 2>/dev/null; then
    echo "  ✅ TensorRT found (can build engines)"
else
    echo "  ⚠️  TensorRT not found (engine building will fail)"
    echo "  [INFO] TensorRT is usually pre-installed on Jetson devices"
fi

# --- Check model files ---
echo "[4/5] Checking model files..."
if [ ! -d "$MODELS_DIR" ]; then
    echo "  ❌ Models directory not found: $MODELS_DIR"
    exit 1
fi

if [ ! -f "$CLASS_NAMES_PATH" ]; then
    echo "  ❌ Class names file not found: $CLASS_NAMES_PATH"
    exit 1
fi
echo "  ✅ Class names file found"

if [ ! -f "$ONNX_PATH" ] && [ ! -f "$ENGINE_PATH" ]; then
    echo "  ❌ No model files found (neither ONNX nor engine)"
    echo "  [INFO] Expected ONNX: $ONNX_PATH"
    echo "  [INFO] Expected engine: $ENGINE_PATH"
    exit 1
fi

if [ -f "$ONNX_PATH" ]; then
    echo "  ✅ ONNX model found: $ONNX_PATH"
else
    echo "  ⚠️  ONNX model not found (will use existing engine)"
fi

if [ -f "$ENGINE_PATH" ]; then
    echo "  ✅ TensorRT engine found: $ENGINE_PATH"
else
    echo "  ⚠️  TensorRT engine not found"
    if [ -f "$ONNX_PATH" ]; then
        echo "  [INFO] Engine will be auto-built on first GUI launch"
    else
        echo "  ❌ Cannot build engine: ONNX model not found"
        exit 1
    fi
fi

# --- Auto-build engine if needed (optional) ---
echo "[5/5] Engine preparation..."
if [ ! -f "$ENGINE_PATH" ] && [ -f "$ONNX_PATH" ]; then
    if python3 -c "import tensorrt" 2>/dev/null; then
        echo "  [INFO] Building TensorRT engine from ONNX model..."
        echo "  [INFO] This may take a few minutes..."
        if python3 -c "
from trt_converter import build_engine
from pathlib import Path
onnx_path = Path('$ONNX_PATH')
engine_path = Path('$ENGINE_PATH')
if build_engine(onnx_path, engine_path):
    print('✅ Engine built successfully')
else:
    print('❌ Engine build failed')
    exit(1)
"; then
            echo "  ✅ Engine built successfully"
        else
            echo "  ⚠️  Engine build failed (will retry on GUI launch)"
        fi
    else
        echo "  ⚠️  Cannot build engine: TensorRT not available"
        echo "  [INFO] Engine will be built on first GUI launch (if TensorRT is available)"
    fi
else
    echo "  ✅ Engine ready (or will be auto-built on launch)"
fi

echo ""
echo "=============================================================="
echo "✅ Setup check complete!"
echo "=============================================================="
echo ""
echo "To launch the GUI, run:"
echo "  ./run_gui.sh"
echo ""

