# Wire Defect Detection - Simple Deployment Guide

## 🎯 3-Step Deployment Process

### Step 1: Setup Environment ⚙️
```bash
# Copy this folder to Raspberry Pi, then:
chmod +x setup_environment.sh
./setup_environment.sh
```
**Time**: 15-30 minutes  
**What it does**: Installs all dependencies, sets up Python environment, enables camera

### Step 2: Test with Images 🧪
```bash
source venv/bin/activate
python test_with_images.py
```
**Time**: 2-3 minutes  
**What it does**: Tests model with 10 sample images, validates performance, saves results

### Step 3: Run Live Detection 📹
```bash
python run_camera_detection.py
```
**Time**: Continuous operation  
**What it does**: Real-time detection with camera, displays statistics, press Ctrl+C to stop

## 📋 Requirements

- **Hardware**: Raspberry Pi 3B+ with camera module
- **OS**: Raspberry Pi OS (any recent version)
- **Storage**: 2GB free space
- **Network**: Internet connection for setup

## 📊 Expected Performance

- **Raspberry Pi 3B**: 2-4 FPS
- **Raspberry Pi 4**: 4-8 FPS
- **Detection Classes**: fail (red), pagan (blue), valid (green)

## 🚨 If Something Goes Wrong

### Setup fails?
```bash
# Try system Python instead:
deactivate
pip3 install onnxruntime --user
python3 run_camera_detection.py
```

### Camera not working?
```bash
# Test camera:
libcamera-hello --timeout 5000

# Enable camera:
sudo raspi-config nonint do_camera 0
sudo reboot
```

### Performance too slow?
```bash
# Edit run_camera_detection.py:
# Change: self.input_size = 320  (instead of 416)
# Change: self.conf_threshold = 0.4  (instead of 0.25)
```

## ✅ Success Indicators

**Step 1 Complete**: See "Setup Complete!" message  
**Step 2 Complete**: See test results with FPS > 1  
**Step 3 Complete**: See live detection statistics  

## 📁 Package Contents

```
shipping/
├── setup_environment.sh        # Step 1: Setup
├── test_with_images.py         # Step 2: Image testing  
├── run_camera_detection.py     # Step 3: Live detection
├── models/best_cropped.onnx    # AI model
├── test_images/                # Sample images (10 files)
└── README_SIMPLE.md            # This guide
```

**That's it! Simple 3-step deployment.** 🚀
