#!/bin/bash

# ==============================================================================
# GUI Runner for Wire Defect Detection
# ==============================================================================
# This script automatically handles setup and launches the GUI application.
# It performs:
# 1. Sync model_config.json with actual model files
# 2. Cleanup old config.json (remove model_path if present)
# 3. Verify model files exist
# 4. Launch GUI application
# ==============================================================================

# Don't exit on error for setup steps (they may not be needed)
set +e

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Navigate to the script's directory to ensure relative paths work correctly
cd "$SCRIPT_DIR"

echo "=============================================================="
echo "Wire Defect Detection - GUI Launcher"
echo "=============================================================="
echo ""

# Step 1: Sync model_config.json with actual files
echo "[1/4] Syncing model_config.json with actual model files..."
if python3 sync_model_config.py --quiet 2>/dev/null; then
    echo "  [OK] model_config.json synced"
else
    echo "  [WARN] Failed to sync model_config.json (may not exist yet)"
fi
echo ""

# Step 2: Cleanup old config.json (remove model_path if present)
echo "[2/4] Cleaning up config.json..."
if python3 cleanup_config.py --quiet 2>/dev/null; then
    echo "  [OK] config.json cleaned"
else
    echo "  [INFO] config.json cleanup not needed or failed (non-critical)"
fi
echo ""

# Step 3: Verify model files exist
echo "[3/4] Verifying model files..."
MODEL_CONFIG="$SCRIPT_DIR/model_config.json"
if [ -f "$MODEL_CONFIG" ]; then
    ONNX_PATH=$(python3 -c "import json; print(json.load(open('$MODEL_CONFIG')).get('onnx_model_path', ''))" 2>/dev/null | sed "s|^models/|$SCRIPT_DIR/models/|")
    ENGINE_PATH=$(python3 -c "import json; print(json.load(open('$MODEL_CONFIG')).get('tensorrt_engine_path', ''))" 2>/dev/null | sed "s|^models/|$SCRIPT_DIR/models/|")
    
    if [ -n "$ONNX_PATH" ] && [ -f "$ONNX_PATH" ]; then
        echo "  [OK] ONNX model found: $(basename "$ONNX_PATH")"
    else
        echo "  [WARN] ONNX model not found: $ONNX_PATH"
    fi
    
    if [ -n "$ENGINE_PATH" ] && [ -f "$ENGINE_PATH" ]; then
        echo "  [OK] TensorRT engine found: $(basename "$ENGINE_PATH")"
    else
        echo "  [INFO] TensorRT engine not found (will be auto-built if needed): $(basename "$ENGINE_PATH" 2>/dev/null || echo 'N/A')"
    fi
else
    echo "  [WARN] model_config.json not found"
fi
echo ""

# Step 4: Launch GUI
echo "[4/4] Launching GUI application..."
echo "=============================================================="
echo ""

# Now exit on error for GUI launch
set -e

# Run the main GUI application
python3 gui_detection_runner.py

echo ""
echo "[INFO] Application finished."

