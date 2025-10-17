# Wire Defect Detection - Deployment Checklist

## 📋 Pre-Deployment Checklist

### Hardware Requirements ✅
- [ ] Raspberry Pi 3B or newer (4GB RAM recommended)
- [ ] Raspberry Pi Camera Module v2 or v3
- [ ] MicroSD card 32GB+ (Class 10 or better)
- [ ] Power supply 5V/3A (official RPi adapter recommended)
- [ ] Ethernet cable or WiFi setup
- [ ] Heatsink/fan for continuous operation (recommended)

### Software Requirements ✅
- [ ] Raspberry Pi OS (64-bit recommended)
- [ ] SSH enabled for remote access
- [ ] Camera interface enabled
- [ ] Internet connection for package installation
- [ ] 2GB+ free disk space

### File Transfer ✅
- [ ] Copy entire `shipping/` folder to Raspberry Pi
- [ ] Verify all files transferred correctly
- [ ] Check file permissions (scripts should be executable)

```bash
# Verify file transfer
ls -la ~/wire_defect/
# Should see: README.md, setup_rpi.sh, rpi_inference_ncnn.py, etc.
```

## 🚀 Deployment Steps

### Step 1: Initial Setup ✅
```bash
# Connect to Raspberry Pi
ssh pi@raspberrypi.local

# Navigate to project directory
cd ~/wire_defect/

# Make setup script executable
chmod +x setup_rpi.sh

# Run setup (this may take 15-30 minutes)
./setup_rpi.sh
```

### Step 2: Model Preparation ✅
```bash
# Activate virtual environment
source venv/bin/activate

# Convert ONNX to NCNN (if needed)
python convert_to_ncnn.py

# Verify model files
ls -lh models/
# Should see: best_cropped.param (~10KB), best_cropped.bin (~6MB)
```

### Step 3: System Validation ✅
```bash
# Run comprehensive test
python test_deployment.py

# Expected output: All tests should PASS
# If any test fails, check TROUBLESHOOTING.md
```

### Step 4: Performance Benchmark ✅
```bash
# Run performance benchmark
python benchmark.py

# Expected results:
# - Preprocessing: 20-50ms
# - NCNN Inference: 150-300ms  
# - End-to-End: 200-400ms
# - FPS: 3-6 (RPi 3B), 6-12 (RPi 4)
```

### Step 5: Camera Test ✅
```bash
# Test camera functionality
libcamera-hello --timeout 5000

# Should display camera preview for 5 seconds
# If fails, check camera connection and enable camera interface
```

### Step 6: Live Inference ✅
```bash
# Start real-time detection
python rpi_inference_ncnn.py

# Expected output:
# [NCNN] FPS: 4.2 | Total: 1523 | Fail: 1234 | Pagan: 156 | Valid: 133

# Press Ctrl+C to stop
```

## ✅ Validation Tests

### Test 1: Import Validation
```bash
python3 -c "
import numpy as np
import cv2
import ncnn
from picamera2 import Picamera2
print('✅ All imports successful')
"
```

### Test 2: Model Loading
```bash
python3 -c "
import ncnn
net = ncnn.Net()
ret1 = net.load_param('models/best_cropped.param')
ret2 = net.load_model('models/best_cropped.bin')
print(f'Model loading: param={ret1}, bin={ret2}')
print('✅ Model loaded' if ret1==0 and ret2==0 else '❌ Model failed')
"
```

### Test 3: Camera Access
```bash
python3 -c "
from picamera2 import Picamera2
picam2 = Picamera2()
info = picam2.global_camera_info()
print(f'Cameras detected: {len(info)}')
print('✅ Camera OK' if info else '❌ No camera')
"
```

### Test 4: Performance Check
```bash
# Should complete in < 30 seconds
timeout 30 python benchmark.py --runs 5
```

## 🔧 Post-Deployment Configuration

### Optimize for Production ✅
```bash
# Edit configuration in rpi_inference_ncnn.py
nano rpi_inference_ncnn.py

# Key settings to adjust:
# - conf_threshold: 0.25 (lower = more detections)
# - input_size: 416 (lower = faster, less accurate)
# - num_threads: 4 (adjust based on CPU load)
```

### Setup Auto-Start (Optional) ✅
```bash
# Create systemd service
sudo nano /etc/systemd/system/wire-defect.service

# Add service content:
[Unit]
Description=Wire Defect Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/wire_defect
ExecStart=/home/pi/wire_defect/venv/bin/python rpi_inference_ncnn.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# Enable service
sudo systemctl enable wire-defect.service
sudo systemctl start wire-defect.service

# Check status
sudo systemctl status wire-defect.service
```

### Monitor Performance ✅
```bash
# Check system resources
htop

# Monitor temperature
watch -n 5 vcgencmd measure_temp

# Check service logs
sudo journalctl -u wire-defect.service -f
```

## 🚨 Troubleshooting Quick Fixes

### If FPS < 3:
1. Reduce input size: `self.input_size = 320`
2. Increase confidence: `conf_threshold = 0.4`
3. Skip frames: `if frame_count % 2 == 0: continue`

### If Temperature > 70°C:
1. Add heatsink/fan
2. Reduce threads: `num_threads = 2`
3. Lower input resolution
4. Increase frame skip interval

### If Memory Issues:
1. Increase swap: Edit `/etc/dphys-swapfile`
2. Close other applications
3. Reboot system

### If Camera Fails:
1. Check connections
2. Enable camera: `sudo raspi-config nonint do_camera 0`
3. Reboot system
4. Test with: `libcamera-hello`

## 📊 Success Criteria

### Minimum Performance Requirements ✅
- [ ] FPS ≥ 3 (Raspberry Pi 3B) or ≥ 6 (Raspberry Pi 4)
- [ ] Inference time ≤ 400ms per frame
- [ ] Memory usage ≤ 80% of available RAM
- [ ] CPU temperature ≤ 70°C under load
- [ ] No camera errors for 10+ minutes continuous operation

### Functional Requirements ✅
- [ ] Detects all 3 classes: fail, pagan, valid
- [ ] Real-time statistics display
- [ ] Graceful shutdown with Ctrl+C
- [ ] Automatic recovery from temporary errors
- [ ] Stable operation for 1+ hour continuous use

### Quality Requirements ✅
- [ ] Detection confidence appropriate for use case
- [ ] False positive rate acceptable
- [ ] Response time suitable for production line speed
- [ ] System remains responsive during operation

## 🎯 Deployment Complete!

Once all checklist items are verified:

1. **Document the setup** - Record any custom configurations
2. **Train operators** - Show how to start/stop the system
3. **Setup monitoring** - Configure alerts for failures
4. **Plan maintenance** - Schedule regular system checks

**System is ready for production use! 🎉**
