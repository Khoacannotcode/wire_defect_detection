#!/bin/bash

echo "=============================================="
echo "  Wire Defect Detection - Simple Setup"
echo "  3-Step Deployment Process"
echo "=============================================="
echo ""

# Check if running on Raspberry Pi
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo "✅ Detected Raspberry Pi"
else
    echo "⚠️  Not running on Raspberry Pi, continuing anyway..."
fi

echo ""
echo "This setup will:"
echo "  1. Install system dependencies"
echo "  2. Setup Python environment"
echo "  3. Install required packages"
echo "  4. Enable camera"
echo "  5. Validate installation"
echo ""

# Step 1: System dependencies
echo "=============================================="
echo "[1/5] Installing system dependencies..."
echo "=============================================="

sudo apt update

# Install essential packages with fallbacks
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-opencv \
    python3-numpy \
    python3-picamera2 \
    python3-libcamera \
    libcamera-apps \
    cmake \
    build-essential \
    pkg-config

# Install math libraries (with fallback for libatlas issue)
if ! sudo apt install -y libatlas-base-dev 2>/dev/null; then
    echo "libatlas-base-dev not available, using OpenBLAS..."
    sudo apt install -y libopenblas-dev libblas-dev liblapack-dev
fi

# Install additional dependencies for picamera2
sudo apt install -y libcap-dev libffi-dev

echo "✅ System dependencies installed"

# Step 2: Python environment
echo ""
echo "=============================================="
echo "[2/5] Setting up Python environment..."
echo "=============================================="

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

echo "✅ Python environment ready"

# Step 3: Install packages
echo ""
echo "=============================================="
echo "[3/5] Installing Python packages..."
echo "=============================================="

# Install packages one by one with error handling
echo "Installing core packages..."
pip install numpy opencv-python-headless pillow tqdm

echo "Installing ONNX Runtime..."
pip install onnxruntime || {
    echo "ONNX Runtime installation failed, trying alternative..."
    pip install onnxruntime-tools || echo "⚠️  ONNX Runtime not installed"
}

echo "Setting up picamera2..."
# Try to install picamera2 in venv, fallback to system version
if ! pip install picamera2 --no-deps 2>/dev/null; then
    echo "Using system picamera2 (this is normal and preferred)"
    
    # Create link to system picamera2
    VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
    if [ -n "$VENV_SITE" ] && [ -d "/usr/lib/python3/dist-packages/picamera2" ]; then
        ln -sf /usr/lib/python3/dist-packages/picamera2 "$VENV_SITE/picamera2" 2>/dev/null
        echo "✅ Linked system picamera2 to virtual environment"
    fi
fi

echo "✅ Python packages installed"

# Step 4: Enable camera
echo ""
echo "=============================================="
echo "[4/5] Enabling camera interface..."
echo "=============================================="

sudo raspi-config nonint do_camera 0
echo "✅ Camera interface enabled"

# Step 5: Validation
echo ""
echo "=============================================="
echo "[5/5] Validating installation..."
echo "=============================================="

# Test imports
echo "Testing imports..."
python3 -c "
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')

try:
    import numpy as np
    print('  ✅ numpy')
except ImportError as e:
    print(f'  ❌ numpy: {e}')

try:
    import cv2
    print('  ✅ opencv')
except ImportError as e:
    print(f'  ❌ opencv: {e}')

try:
    import onnxruntime as ort
    print('  ✅ onnxruntime')
except ImportError as e:
    print(f'  ❌ onnxruntime: {e}')

try:
    from picamera2 import Picamera2
    print('  ✅ picamera2')
except ImportError as e:
    print(f'  ❌ picamera2: {e}')
"

# Test model loading
echo ""
echo "Testing model loading..."
if [ -f "models/best_cropped.onnx" ]; then
    python3 -c "
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')
import onnxruntime as ort
try:
    session = ort.InferenceSession('models/best_cropped.onnx', providers=['CPUExecutionProvider'])
    print('  ✅ Model loads successfully')
except Exception as e:
    print(f'  ❌ Model loading failed: {e}')
"
else
    echo "  ❌ Model file not found: models/best_cropped.onnx"
fi

# Test camera
echo ""
echo "Testing camera..."
python3 -c "
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')
try:
    from picamera2 import Picamera2
    picam2 = Picamera2()
    info = picam2.global_camera_info()
    if info:
        print(f'  ✅ Camera detected: {len(info)} camera(s)')
    else:
        print('  ❌ No camera detected')
except Exception as e:
    print(f'  ❌ Camera test failed: {e}')
"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Test with images: python test_with_images.py"
echo "  2. Run live detection: python run_camera_detection.py"
echo ""
echo "If you encounter issues:"
echo "  - Use system Python: deactivate && python3 <script>"
echo "  - Check camera: libcamera-hello --timeout 5000"
echo ""
