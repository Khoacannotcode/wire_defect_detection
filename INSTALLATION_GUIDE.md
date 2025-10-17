# Wire Defect Detection - Installation Guide for RPi

## 🚨 Quick Fix for Current Error

You encountered the `libatlas-base-dev` error. Here's the immediate solution:

### Step 1: Run Quick Fix
```bash
# On your Raspberry Pi, run:
chmod +x quick_fix.sh
./quick_fix.sh
```

### Step 2: Continue Setup
```bash
# After quick fix completes:
chmod +x setup_rpi.sh
./setup_rpi.sh
```

### Step 3: Test Installation
```bash
source venv/bin/activate
python test_deployment.py
```

## 🔧 Alternative Installation Methods

### Method 1: Manual Step-by-Step
```bash
# Fix dependencies
sudo apt update
sudo apt install -y libopenblas-dev libblas-dev liblapack-dev
sudo apt install -y python3-pip python3-venv python3-dev cmake build-essential

# Install camera
sudo apt install -y python3-picamera2 python3-libcamera

# Create environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install numpy>=1.21.0,<1.25.0
pip install opencv-python-headless>=4.5.0,<4.9.0
pip install pillow>=8.0.0,<10.0.0
pip install tqdm>=4.60.0
pip install onnxruntime>=1.12.0

# Enable camera
sudo raspi-config nonint do_camera 0
```

### Method 2: Using ONNX Runtime Only (Skip NCNN)
```bash
# If NCNN installation keeps failing, use ONNX Runtime only:
source venv/bin/activate
pip install onnxruntime

# Test with ONNX
python rpi_inference_onnx.py
```

### Method 3: System Packages Only
```bash
# Use system packages (no virtual environment)
sudo apt install python3-numpy python3-opencv python3-picamera2
sudo apt install python3-pip
pip3 install onnxruntime --user

# Run with system Python
python3 rpi_inference_onnx.py
```

## 🎯 Recommended Approach

**For most users:**
1. Run `./quick_fix.sh` first
2. Then run `./setup_rpi.sh`
3. Use `python start_detection.py` (auto-selects best engine)

**If you keep having issues:**
1. Use Method 2 (ONNX Runtime only)
2. Skip NCNN completely
3. Performance will be 50% slower but more reliable

## 🔍 Troubleshooting Specific Errors

### `libatlas-base-dev` not found
```bash
# This package was removed in newer RPi OS
sudo apt install libopenblas-dev libblas-dev liblapack-dev
```

### `ncnn-python` no matching distribution
```bash
# NCNN wheels may not be available for your Python version
# Use ONNX Runtime instead:
pip install onnxruntime
python rpi_inference_onnx.py
```

### `picamera2` version conflicts
```bash
# Use system package instead of pip
sudo apt install python3-picamera2
# Don't install via pip
```

### Python version too old/new
```bash
# Check version
python3 --version

# If < 3.7 or > 3.11, install compatible version:
sudo apt install python3.9 python3.9-venv python3.9-dev
python3.9 -m venv venv
```

## ✅ Success Indicators

After successful installation, you should see:
```bash
python test_deployment.py
# Expected output:
✅ PASS Import Test
✅ PASS Model Files Test  
✅ PASS NCNN Model Test (or ONNX Runtime Test)
✅ PASS Camera Test
✅ PASS Performance Test

📊 Results: 5/5 tests passed
🎉 All tests passed! Deployment is ready.
```

## 🚀 Quick Start After Installation

```bash
# Activate environment
source venv/bin/activate

# Start detection (auto-selects best engine)
python start_detection.py

# Or manually choose:
python rpi_inference_ncnn.py    # NCNN (faster)
python rpi_inference_onnx.py    # ONNX Runtime (more compatible)
```

## 📞 Still Having Issues?

1. **Check TROUBLESHOOTING.md** - Comprehensive solutions
2. **Run diagnostic**: `python test_deployment.py`
3. **Check system**: `vcgencmd measure_temp && free -h`
4. **Try fallback**: Use ONNX Runtime instead of NCNN

**The system is designed to work even if NCNN fails - ONNX Runtime provides a reliable fallback!**
