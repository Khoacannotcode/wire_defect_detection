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
    echo "Detected Jetson platform: $JETSON_INFO"
else
    echo "WARNING: Could not detect Jetson Nano (nv_tegra_release missing)."
    echo "         The script will continue, but please ensure you are on Jetson Nano."
fi

echo ""
echo "This script will:"
echo "  1. Install system dependencies (Python, OpenCV, GStreamer, build tools)"
echo "  2. Create a Python virtual environment (shipping/venv)"
echo "  3. Install Python packages including onnxruntime-gpu"
echo "  4. Run smoke tests for Python modules, model loading, and camera access"
echo ""

print_section() {
    echo "=============================================="
    echo "[${1}] ${2}"
    echo "=============================================="
}

print_section "1/4" "Installing system dependencies"

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
    libjpeg-dev \
    zlib1g-dev \
    v4l-utils \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav

SYSTEM_CV2_PATH=$(python3 - <<'PYCODE'
import importlib
try:
    import cv2  # noqa: F401
except ImportError:
    print("", end="")
else:
    import cv2
    print(cv2.__file__, end="")
PYCODE
)

if [[ -z "$SYSTEM_CV2_PATH" ]]; then
    echo "ERROR: System OpenCV installation not detected. Ensure python3-opencv is installed."
    exit 1
fi

SYSTEM_CV2_DIR=$(dirname "$SYSTEM_CV2_PATH")
echo "Detected system OpenCV module at $SYSTEM_CV2_PATH"

echo "System dependencies installed"

print_section "2/4" "Setting up Python virtual environment"

cd "$(dirname "$0")"

echo "Removing existing virtual environment if it exists..."
rm -rf venv

echo "Creating Python virtual environment..."
python3 -m venv --system-site-packages venv

source venv/bin/activate

echo "Upgrading pip, setuptools, and wheel..."
python -m pip install --no-cache-dir --upgrade pip setuptools wheel

VENV_SITE_PACKAGES=$(python - <<'PYCODE'
import site
print(site.getsitepackages()[0], end="")
PYCODE
)

echo "$SYSTEM_CV2_DIR" > "$VENV_SITE_PACKAGES/opencv-system.pth"
export PYTHONPATH="$SYSTEM_CV2_DIR:${PYTHONPATH:-}"

print_section "3/4" "Installing Python packages"

if [[ -f requirements_simple.txt ]]; then
    python -m pip install --no-cache-dir -r requirements_simple.txt
else
    python -m pip install --no-cache-dir numpy pillow tqdm
fi

if ! python - <<'PYCODE'
try:
    import onnxruntime as ort
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        raise SystemExit(0)
    raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PYCODE
then
    python -m pip uninstall -y onnxruntime onnxruntime_gpu >/dev/null 2>&1 || true

    PYTHON_VERSION=$(python - <<'PYCODE'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}", end="")
PYCODE
)

    case "$PYTHON_VERSION" in
        3.6)
            DEFAULT_ONNXRUNTIME_GPU_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v46/onnxruntime/onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl"
            ;;
        3.8)
            DEFAULT_ONNXRUNTIME_GPU_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v502/onnxruntime/onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl"
            ;;
        3.10)
            DEFAULT_ONNXRUNTIME_GPU_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v60/onnxruntime/onnxruntime_gpu-1.16.1-cp310-cp310-linux_aarch64.whl"
            ;;
        *)
            echo "ERROR: Unsupported Python version $PYTHON_VERSION for automatic onnxruntime-gpu installation."
            echo "       Set ONNXRUNTIME_GPU_WHEEL to a compatible wheel URL or local path and rerun."
            exit 1
            ;;
    esac

    ONNXRUNTIME_GPU_WHEEL=${ONNXRUNTIME_GPU_WHEEL:-$DEFAULT_ONNXRUNTIME_GPU_WHEEL}

    if [[ -z "$ONNXRUNTIME_GPU_WHEEL" ]]; then
        echo "ERROR: No onnxruntime-gpu wheel specified."
        exit 1
    fi

    echo "Installing onnxruntime-gpu from $ONNXRUNTIME_GPU_WHEEL"
    python -m pip install --no-cache-dir "$ONNXRUNTIME_GPU_WHEEL"
fi

python - <<'PYCODE'
import onnxruntime as ort
providers = ort.get_available_providers()
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("CUDAExecutionProvider not available after onnxruntime-gpu installation")
print(f"  onnxruntime GPU providers: {providers}")
PYCODE

print_section "4/4" "Running validation checks"

python - <<'PYCODE'
import os
import cv2
import numpy as np
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  onnxruntime available, providers: {providers}")
except Exception as exc:
    print(f"  onnxruntime import failed: {exc}")

model_path = os.path.join('models', 'best_cropped.onnx')
if os.path.isfile(model_path):
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("  Model loads successfully")
    except Exception as exc:
        print(f"  Model loading failed: {exc}")
else:
    print(f"  Model file missing: {model_path}")

try:
    cam = cv2.VideoCapture(0)
    if cam.isOpened():
        ret, frame = cam.read()
        if ret:
            print(f"  Camera detected via /dev/video0 (frame size: {frame.shape[1]}x{frame.shape[0]})")
        else:
            print("  Camera opened but no frame returned")
    else:
        print("  Unable to open /dev/video0 (USB/CSI camera not detected)")
    cam.release()
except Exception as exc:
    print(f"  Camera check failed: {exc}")
PYCODE

import cv2
print(f"  OpenCV version: {cv2.__version__}")

echo ""
echo "Setup complete!"
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. python test_with_images.py"
echo "  3. python run_camera_detection.py --source 0 --width 1280 --height 720 --fps 30"
echo ""
