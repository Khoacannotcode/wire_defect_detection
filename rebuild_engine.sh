#!/bin/bash

# ==============================================================================
# Rebuild TensorRT Engine Script
# ==============================================================================
# This script rebuilds the TensorRT engine from ONNX model.
# Use this when you get "invalid resource handle" errors, which usually
# means the engine was built on a different device.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

MODELS_DIR="$SCRIPT_DIR/models"
ONNX_PATH="$MODELS_DIR/best_cropped.onnx"
ENGINE_PATH="$MODELS_DIR/best_cropped.engine"

echo "=============================================================="
echo "Rebuild TensorRT Engine"
echo "=============================================================="

# Check if ONNX model exists
if [ ! -f "$ONNX_PATH" ]; then
    echo "[ERROR] ONNX model not found: $ONNX_PATH"
    echo "[INFO] Please ensure the ONNX model exists before rebuilding engine"
    exit 1
fi

# Backup existing engine if it exists
if [ -f "$ENGINE_PATH" ]; then
    BACKUP_PATH="${ENGINE_PATH}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "[INFO] Backing up existing engine to: $BACKUP_PATH"
    cp "$ENGINE_PATH" "$BACKUP_PATH"
    echo "[INFO] Removing old engine..."
    rm "$ENGINE_PATH"
fi

# Rebuild engine
echo "[INFO] Rebuilding TensorRT engine from: $ONNX_PATH"
echo "[INFO] This may take a few minutes..."
echo ""

if python3 trt_converter.py; then
    echo ""
    echo "=============================================================="
    echo "✅ Engine rebuilt successfully!"
    echo "=============================================================="
    echo "[INFO] New engine saved to: $ENGINE_PATH"
    if [ -f "$BACKUP_PATH" ]; then
        echo "[INFO] Old engine backed up to: $BACKUP_PATH"
    fi
    echo ""
    echo "You can now run the GUI again:"
    echo "  ./run_gui.sh"
else
    echo ""
    echo "=============================================================="
    echo "❌ Engine rebuild failed!"
    echo "=============================================================="
    echo "[ERROR] Check the error messages above for details"
    if [ -f "$BACKUP_PATH" ]; then
        echo "[INFO] Restoring backup engine..."
        mv "$BACKUP_PATH" "$ENGINE_PATH"
    fi
    exit 1
fi

