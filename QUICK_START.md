# 🚀 QUICK START - Wire Defect Detection on RPi

## ⚡ Immediate Solution for Your Current Error

You're getting `libcap development headers` error. Here's the **fastest fix**:

### Option 1: Fix Dependencies (Recommended)
```bash
# On your Raspberry Pi:
chmod +x fix_picamera2_deps.sh
./fix_picamera2_deps.sh

# Then install picamera2:
pip install picamera2 --force-reinstall

# Test:
python rpi_inference_onnx.py
```

### Option 2: Use System Python (Fastest)
```bash
# Exit virtual environment
deactivate

# Install ONNX Runtime for system Python
pip3 install onnxruntime --user

# Run with system Python (bypasses all venv issues)
python3 run_system_python.py
```

### Option 3: Skip picamera2 pip install
```bash
# The system already has picamera2 installed
# Just run the fixed script:
python rpi_inference_onnx.py
# (It will automatically use system picamera2)
```

## 🎯 Which Option to Choose?

- **Option 2** is **FASTEST** - will work immediately
- **Option 1** is **CLEANEST** - fixes the root cause
- **Option 3** is **SIMPLEST** - uses existing fixes

## ⚡ Super Quick Test

Want to test if everything works? Run this one command:

```bash
deactivate && python3 run_system_python.py
```

This bypasses ALL virtual environment issues and uses system packages directly.

## 🔍 What's Happening?

The error you're seeing:
```
You need to install libcap development headers to build this module
```

This means `python-prctl` (a dependency of picamera2) needs `libcap-dev` to compile. This is a common issue on Raspberry Pi.

## ✅ Expected Results

After fixing, you should see:
```bash
=== Wire Defect Detection with System Python ===
Using system packages (no virtual environment)

✅ Using system picamera2
✅ Using ONNX Runtime
Loading ONNX model from: models/best_cropped.onnx
✅ ONNX model loaded successfully
Initializing camera...
✅ Camera ready

Starting inference... (Press Ctrl+C to stop)

[SYSTEM] FPS: 2.1 | Total: 45 | Fail: 32 | Pagan: 8 | Valid: 5
```

## 🚨 Emergency Fallback

If NOTHING works, you can still test the model:

```bash
# Test with a static image instead of camera
python3 -c "
import cv2
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession('models/best_cropped.onnx', providers=['CPUExecutionProvider'])
print('✅ Model loads successfully!')

# Test with dummy image
dummy = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
dummy = dummy.astype(np.float32) / 255.0
dummy = np.transpose(dummy, (2, 0, 1))
dummy = np.expand_dims(dummy, axis=0)

# Run inference
outputs = session.run(None, {'images': dummy})
print('✅ Inference works!')
print(f'Output shape: {outputs[0].shape}')
"
```

**Choose Option 2 for immediate results!** 🎉
