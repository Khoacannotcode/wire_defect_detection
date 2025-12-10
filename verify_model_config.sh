#!/bin/bash
# Verify and sync model_config.json with actual model files on Jetson

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

MODELS_DIR="$SCRIPT_DIR/models"
MODEL_CONFIG="$SCRIPT_DIR/model_config.json"

echo "=============================================================="
echo "Model Config Verification"
echo "=============================================================="

# Check if model_config.json exists
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "[WARN] model_config.json not found, creating default..."
    cat > "$MODEL_CONFIG" << EOF
{
  "onnx_model_path": "models/best_v3_416x256.onnx",
  "tensorrt_engine_path": "models/best_v3_416x256.engine",
  "class_names_path": "models/class_names.txt"
}
EOF
    echo "[OK] Created default model_config.json"
fi

# Check actual model files
echo ""
echo "[1/3] Checking actual model files..."
ONNX_FILES=$(find "$MODELS_DIR" -name "*.onnx" -type f 2>/dev/null | xargs -n1 basename | sort)
ENGINE_FILES=$(find "$MODELS_DIR" -name "*.engine" -type f 2>/dev/null | xargs -n1 basename | sort)

if [ -z "$ONNX_FILES" ]; then
    echo "  [WARN] No ONNX files found in $MODELS_DIR"
else
    echo "  Found ONNX files:"
    echo "$ONNX_FILES" | sed 's/^/    - /'
fi

if [ -z "$ENGINE_FILES" ]; then
    echo "  [WARN] No engine files found in $MODELS_DIR"
else
    echo "  Found engine files:"
    echo "$ENGINE_FILES" | sed 's/^/    - /'
fi

# Check config paths
echo ""
echo "[2/3] Checking model_config.json paths..."
if [ -f "$MODEL_CONFIG" ]; then
    CONFIG_ONNX=$(python3 -c "import json; print(json.load(open('$MODEL_CONFIG')).get('onnx_model_path', ''))" 2>/dev/null | sed "s|^models/||")
    CONFIG_ENGINE=$(python3 -c "import json; print(json.load(open('$MODEL_CONFIG')).get('tensorrt_engine_path', ''))" 2>/dev/null | sed "s|^models/||")
    
    echo "  Config ONNX: $CONFIG_ONNX"
    echo "  Config Engine: $CONFIG_ENGINE"
    
    # Check if config paths exist
    if [ -n "$CONFIG_ONNX" ] && [ -f "$MODELS_DIR/$CONFIG_ONNX" ]; then
        echo "  [OK] ONNX file exists: $CONFIG_ONNX"
    elif [ -n "$CONFIG_ONNX" ]; then
        echo "  [WARN] ONNX file NOT found: $CONFIG_ONNX"
    fi
    
    if [ -n "$CONFIG_ENGINE" ] && [ -f "$MODELS_DIR/$CONFIG_ENGINE" ]; then
        echo "  [OK] Engine file exists: $CONFIG_ENGINE"
    elif [ -n "$CONFIG_ENGINE" ]; then
        echo "  [WARN] Engine file NOT found: $CONFIG_ENGINE"
    fi
fi

# Suggest update if mismatch
echo ""
echo "[3/3] Recommendations..."
if [ -n "$ONNX_FILES" ] && [ -n "$CONFIG_ONNX" ]; then
    FIRST_ONNX=$(echo "$ONNX_FILES" | head -1)
    if [ "$CONFIG_ONNX" != "$FIRST_ONNX" ]; then
        echo "  [INFO] Config ONNX ($CONFIG_ONNX) doesn't match found file ($FIRST_ONNX)"
        echo "  [INFO] To update config, run:"
        echo "    python3 -c \""
        echo "import json"
        echo "config = json.load(open('$MODEL_CONFIG'))"
        echo "config['onnx_model_path'] = 'models/$FIRST_ONNX'"
        echo "if '$FIRST_ONNX' in '$ENGINE_FILES':"
        echo "    config['tensorrt_engine_path'] = 'models/${FIRST_ONNX%.onnx}.engine'"
        echo "json.dump(config, open('$MODEL_CONFIG', 'w'), indent=2)"
        echo "\""
    fi
fi

echo ""
echo "=============================================================="
echo "[OK] Verification complete"
echo "=============================================================="

