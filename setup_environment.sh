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
    python3-gi \
    python3-opencv \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    wget \
    libprotobuf-dev \
    protobuf-compiler \
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
            DEFAULT_ONNXRUNTIME_GPU_WHEEL="https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.whl"
            ONNXRUNTIME_FILENAME="onnxruntime_gpu-1.11.0-cp36-cp36m-linux_aarch64.whl"
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

    if [[ -n "$ONNXRUNTIME_FILENAME" ]]; then
        echo "Downloading onnxruntime-gpu from $ONNXRUNTIME_GPU_WHEEL"
        wget -q --show-progress -O "$ONNXRUNTIME_FILENAME" "$ONNXRUNTIME_GPU_WHEEL"
        echo "Installing onnxruntime-gpu from local file $ONNXRUNTIME_FILENAME"
        python -m pip install --no-cache-dir "$ONNXRUNTIME_FILENAME"
        rm "$ONNXRUNTIME_FILENAME"
    else
        echo "Installing onnxruntime-gpu from $ONNXRUNTIME_GPU_WHEEL"
        python -m pip install --no-cache-dir "$ONNXRUNTIME_GPU_WHEEL"
    fi
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
import sys
import cv2
import numpy as np

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    if not Gst.is_initialized():
        Gst.init(None)
    print("  PyGObject + GStreamer bindings detected")
except Exception as exc:
    print(f"  [ERROR] PyGObject/GStreamer import failed: {exc}")
    sys.exit(1)

# --- System Checks ---
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  onnxruntime available, providers: {providers}")
except Exception as exc:
    print(f"  [ERROR] onnxruntime import failed: {exc}")
    sys.exit(1)

print(f'  OpenCV version: {cv2.__version__}')

# --- Model Check ---
model_dir = 'models'
model_path_opset16 = os.path.join(model_dir, "best_cropped_opset16.onnx")
model_path_original = os.path.join(model_dir, "best_cropped.onnx")

model_to_load = None
if os.path.exists(model_path_opset16):
    model_to_load = model_path_opset16
    print(f"  Found pre-converted model, attempting to load: {model_to_load}")
elif os.path.exists(model_path_original):
    model_to_load = model_path_original
    print(f"  Found original model, attempting to load: {model_to_load}")

if model_to_load:
    try:
        session = ort.InferenceSession(model_to_load, providers=['CPUExecutionProvider'])
        print("  [SUCCESS] Model loads successfully with ONNX Runtime.")
    except Exception as exc:
        print(f"\n  [ERROR] Model loading failed: {exc}\n")
        print("  ================================[ TROUBLESHOOTING ]================================")
        print("  This error often means the ONNX model's 'opset version' is too new.")
        print("  The model must be opset 16 or lower to run on this device's environment.")
        print("  Please convert the model on your development machine using the provided script")
        print("  and place the converted 'best_cropped_opset16.onnx' file in the 'models' directory.")
        print("  ====================================================================================\n")

else:
    print(f"  [ERROR] Model file not found. Searched for:")
    print(f"    - {model_path_opset16}")
    print(f"    - {model_path_original}")

# --- Camera Check ---
def get_csi_pipeline(capture_width=1280, capture_height=720, framerate=30, display_width=1280, display_height=720):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        "nvvidconv flip-method=0 ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

print("  --- Camera Check ---")
print(f"  OpenCV version: {cv2.__version__}")

try:
    # First, try CSI camera with GStreamer pipeline (preferred for Jetson)
    print("  Attempting CSI camera with GStreamer pipeline...")
    csi_pipeline = get_csi_pipeline()
    cam = cv2.VideoCapture(csi_pipeline)
    
    if cam.isOpened():
        ret, frame = cam.read()
        if ret and frame is not None:
            print(f"  [SUCCESS] CSI Camera detected via GStreamer (frame size: {frame.shape[1]}x{frame.shape[0]})")
        else:
            print("  [WARN] CSI Camera opened but failed to capture a frame.")
        cam.release()
    else:
        print("  CSI camera not available via GStreamer. Trying USB camera...")
        cam.release()
        
        # Fallback to USB camera
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            ret, frame = cam.read()
            if ret and frame is not None:
                print(f"  [SUCCESS] USB Camera detected (frame size: {frame.shape[1]}x{frame.shape[0]})")
            else:
                print("  [WARN] USB Camera opened but failed to capture a frame.")
        else:
            print("  [ERROR] Unable to open camera via GStreamer or USB.")
            print("          Please check camera connection and kernel driver support.")
        cam.release()

except Exception as exc:
    print(f"  [ERROR] Camera check failed with exception: {exc}")
    try:
        cam.release()
    except:
        pass
PYCODE

echo ""
echo "Setup complete!"
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. python test_with_images.py"
echo "  3. python run_camera_detection.py --source 0 --width 1280 --height 720 --fps 30"
echo ""
