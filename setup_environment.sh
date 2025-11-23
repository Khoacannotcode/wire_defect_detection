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
model_path = os.path.join(model_dir, "best_cropped.onnx")

if os.path.exists(model_path):
    print(f"  Found model, attempting to load: {model_path}")
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("  [SUCCESS] Model loads successfully with ONNX Runtime.")
    except Exception as exc:
        print(f"\n  [ERROR] Model loading failed: {exc}\n")
        print("  ================================[ TROUBLESHOOTING ]================================")
        print("  If the model fails to load, check:")
        print("  1. Model file exists and is not corrupted")
        print("  2. ONNX Runtime version is compatible")
        print("  3. Model opset version is supported by ONNX Runtime")
        print("  ====================================================================================\n")
else:
    print(f"  [ERROR] Model file not found: {model_path}")
    print("  Please ensure the ONNX model is in the models/ directory")

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

def try_opencv_csi():
    print("  Attempting CSI camera with OpenCV + GStreamer pipeline...")
    pipeline = get_csi_pipeline()
    cam = cv2.VideoCapture(pipeline)
    try:
        if not cam.isOpened():
            print("  [INFO] OpenCV could not open the CSI pipeline (expected on 3.2.0 without GStreamer).")
            return False
        ret, frame = cam.read()
        if ret and frame is not None:
            print(f"  [SUCCESS] CSI camera detected via OpenCV (frame size: {frame.shape[1]}x{frame.shape[0]})")
            return True
        print("  [WARN] CSI pipeline opened but returned empty frame.")
        return False
    finally:
        cam.release()

def try_pygobject_csi():
    print("  Attempting CSI camera with PyGObject fallback pipeline...")
    pipeline_desc = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink name=setupappsink emit-signals=false max-buffers=1 drop=true sync=false"
    )
    pipeline = None
    try:
        pipeline = Gst.parse_launch(pipeline_desc)
        appsink = pipeline.get_by_name("setupappsink")
        if appsink is None:
            print("  [WARN] PyGObject pipeline missing appsink element.")
            return False

        pipeline.set_state(Gst.State.PLAYING)
        bus = pipeline.get_bus()
        if bus:
            msg = bus.timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg:
                err, debug = msg.parse_error()
                print(f"  [WARN] PyGObject pipeline error: {err} ({debug})")
                return False

        sample = appsink.emit("try-pull-sample", 2 * Gst.SECOND)
        if sample is None:
            print("  [WARN] PyGObject pipeline started but produced no frame.")
            return False

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')

        ok, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            print("  [WARN] Failed to map buffer from PyGObject pipeline.")
            return False
        try:
            np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
        finally:
            buffer.unmap(map_info)

        print(f"  [SUCCESS] CSI camera detected via PyGObject fallback (frame size: {width}x{height})")
        return True
    except Exception as exc:
        print(f"  [WARN] PyGObject fallback failed: {exc}")
        return False
    finally:
        if pipeline:
            pipeline.set_state(Gst.State.NULL)

def try_usb_camera():
    print("  Attempting USB camera on /dev/video0 ...")
    cam = cv2.VideoCapture(0)
    try:
        if not cam.isOpened():
            print("  [WARN] USB camera (/dev/video0) not detected.")
            return False
        ret, frame = cam.read()
        if ret and frame is not None:
            print(f"  [SUCCESS] USB camera detected (frame size: {frame.shape[1]}x{frame.shape[0]})")
            return True
        print("  [WARN] USB camera opened but returned empty frame.")
        return False
    finally:
        cam.release()

if not (try_opencv_csi() or try_pygobject_csi() or try_usb_camera()):
    print("  [ERROR] Automated camera checks failed. Please validate hardware and drivers manually.")
    print("          Test with: gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! xvimagesink")
PYCODE

echo ""
echo "Setup complete!"
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. python test_with_images.py"
echo "  3. python run_camera_detection.py --source 0 --width 1280 --height 720 --fps 30"
echo ""
