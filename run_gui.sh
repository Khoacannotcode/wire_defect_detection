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

# Parse command-line arguments
DEBUG_MODE=0
SHOW_HELP=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug|-d)
            DEBUG_MODE=1
            shift
            ;;
        --help|-h)
            SHOW_HELP=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help or -h for usage information"
            exit 1
            ;;
    esac
done

# Show help and exit if requested
if [ $SHOW_HELP -eq 1 ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --debug, -d    Enable debug mode (verbose logging)"
    echo "  --help, -h     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Run in production mode (minimal logging)"
    echo "  $0 --debug      # Run in debug mode (verbose logging)"
    exit 0
fi

# Don't exit on error for setup steps (they may not be needed)
set +e

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Navigate to the script's directory to ensure relative paths work correctly
cd "$SCRIPT_DIR"

# Production mode: minimal output
# Debug mode: verbose output
if [ $DEBUG_MODE -eq 1 ]; then
    echo "=============================================================="
    echo "Wire Defect Detection - GUI Launcher (DEBUG MODE)"
    echo "=============================================================="
    echo ""
else
    # Production mode: only show errors/warnings
    # Silent startup unless there are issues
    :
fi

# Step 1: Sync model_config.json with actual files
if [ $DEBUG_MODE -eq 1 ]; then
    echo "[1/4] Syncing model_config.json with actual model files..."
fi
if python3 sync_model_config.py --quiet 2>/dev/null; then
    [ $DEBUG_MODE -eq 1 ] && echo "  [OK] model_config.json synced"
else
    [ $DEBUG_MODE -eq 1 ] && echo "  [WARN] Failed to sync model_config.json (may not exist yet)"
    # In production mode, only show warnings/errors
    if [ $DEBUG_MODE -eq 0 ]; then
        echo "[WARN] Failed to sync model_config.json" >&2
    fi
fi
[ $DEBUG_MODE -eq 1 ] && echo ""

# Step 2: Cleanup old config.json (remove model_path if present)
if [ $DEBUG_MODE -eq 1 ]; then
    echo "[2/4] Cleaning up config.json..."
fi
if python3 cleanup_config.py --quiet 2>/dev/null; then
    [ $DEBUG_MODE -eq 1 ] && echo "  [OK] config.json cleaned"
else
    [ $DEBUG_MODE -eq 1 ] && echo "  [INFO] config.json cleanup not needed or failed (non-critical)"
fi
[ $DEBUG_MODE -eq 1 ] && echo ""

# Step 3: Verify model files exist
if [ $DEBUG_MODE -eq 1 ]; then
    echo "[3/4] Verifying model files..."
fi
MODEL_CONFIG="$SCRIPT_DIR/model_config.json"
if [ -f "$MODEL_CONFIG" ]; then
    ONNX_PATH=$(python3 -c "import json; print(json.load(open('$MODEL_CONFIG')).get('onnx_model_path', ''))" 2>/dev/null | sed "s|^models/|$SCRIPT_DIR/models/|")
    ENGINE_PATH=$(python3 -c "import json; print(json.load(open('$MODEL_CONFIG')).get('tensorrt_engine_path', ''))" 2>/dev/null | sed "s|^models/|$SCRIPT_DIR/models/|")
    
    if [ -n "$ONNX_PATH" ] && [ -f "$ONNX_PATH" ]; then
        [ $DEBUG_MODE -eq 1 ] && echo "  [OK] ONNX model found: $(basename "$ONNX_PATH")"
    else
        [ $DEBUG_MODE -eq 1 ] && echo "  [WARN] ONNX model not found: $ONNX_PATH"
        # In production mode, show warnings
        [ $DEBUG_MODE -eq 0 ] && echo "[WARN] ONNX model not found: $(basename "$ONNX_PATH" 2>/dev/null || echo 'N/A')" >&2
    fi
    
    if [ -n "$ENGINE_PATH" ] && [ -f "$ENGINE_PATH" ]; then
        [ $DEBUG_MODE -eq 1 ] && echo "  [OK] TensorRT engine found: $(basename "$ENGINE_PATH")"
    else
        [ $DEBUG_MODE -eq 1 ] && echo "  [INFO] TensorRT engine not found (will be auto-built if needed): $(basename "$ENGINE_PATH" 2>/dev/null || echo 'N/A')"
    fi
else
    [ $DEBUG_MODE -eq 1 ] && echo "  [WARN] model_config.json not found"
    [ $DEBUG_MODE -eq 0 ] && echo "[WARN] model_config.json not found" >&2
fi
[ $DEBUG_MODE -eq 1 ] && echo ""

# Step 4: Launch GUI
if [ $DEBUG_MODE -eq 1 ]; then
    echo "[4/4] Launching GUI application..."
    echo "=============================================================="
    echo ""
fi

# Now exit on error for GUI launch
set -e

# Run the main GUI application with debug flag passed as environment variable
if [ $DEBUG_MODE -eq 1 ]; then
    DEBUG=1 python3 gui_detection_runner.py
else
    DEBUG=0 python3 gui_detection_runner.py
fi

[ $DEBUG_MODE -eq 1 ] && echo ""
[ $DEBUG_MODE -eq 1 ] && echo "[INFO] Application finished."

