#!/bin/bash

set -e

JETSON_INFO=""
if [[ -f /etc/nv_tegra_release ]]; then
    JETSON_INFO=$(head -n 1 /etc/nv_tegra_release 2>/dev/null)
fi

echo "=============================================="
echo "  Wire Defect Detection - Jetson Nano Setup"
echo "=============================================="
echo ""

if [[ -n "$JETSON_INFO" ]]; then
    echo "✅ Detected Jetson platform: $JETSON_INFO"
else
    echo "⚠️  Could not detect Jetson Nano (nv_tegra_release missing)."
    echo "    The script will continue, but please ensure you are on Jetson Nano."
fi

echo ""
echo "This script will:"
echo "  1. Install system dependencies (Python, OpenCV, GStreamer, build tools)"
echo "  2. Create a Python virtual environment (shipping/venv)"
echo "  3. Install Python packages including onnxruntime-gpu"
echo "  4. Run smoke tests for Python modules, model loading, and camera access"
echo ""

# Step 1: System dependencies
print_section() {
    echo "=============================================="
    echo "[${1}] ${2}"
    echo "=============================================="
}

print_section "1/4" "Installing system dependencies..."

sudo apt update
sudo apt install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-opencv \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    v4l-utils \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav

echo "✅ System dependencies installed"

# Step 2: Python environment
print_section "2/4" "Setting up Python virtual environment..."

cd "$(dirname "$0")"

if [[ ! -d venv ]]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "✅ Virtual environment ready (shipping/venv)"

# Step 3: Python packages
print_section "3/4" "Installing Python packages..."

if [[ -f requirements_simple.txt ]]; then
    pip install -r requirements_simple.txt
else
    pip install numpy opencv-python-headless pillow tqdm
fi

if ! pip install onnxruntime-gpu; then
    echo "⚠️  onnxruntime-gpu install failed, falling back to CPU build"
    pip install onnxruntime
fi

echo "✅ Python packages installed"

# Step 4: Validation
print_section "4/4" "Running validation checks..."

python - <<'PYCODE'
import os
import cv2
import numpy as np
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  ✅ onnxruntime available, providers: {providers}")
except Exception as exc:
    print(f"  ❌ onnxruntime import failed: {exc}")

model_path = os.path.join('models', 'best_cropped.onnx')
if os.path.isfile(model_path):
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("  ✅ Model loads successfully")
    except Exception as exc:
        print(f"  ❌ Model loading failed: {exc}")
else:
    print(f"  ❌ Model file missing: {model_path}")

try:
    cam = cv2.VideoCapture(0)
    if cam.isOpened():
        ret, frame = cam.read()
        if ret:
            print(f"  ✅ Camera detected via /dev/video0 (frame size: {frame.shape[1]}x{frame.shape[0]})")
        else:
            print("  ⚠️  Camera opened but no frame returned")
    else:
        print("  ⚠️  Unable to open /dev/video0 (USB/CSI camera not detected)")
    cam.release()
except Exception as exc:
    print(f"  ⚠️  Camera check failed: {exc}")
PYCODE

echo ""
echo "Setup complete!"
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. python test_with_images.py"
echo "  3. python run_camera_detection.py --source 0 --width 1280 --height 720 --fps 30"
echo ""
