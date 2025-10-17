# Wire Defect Detection - Troubleshooting Guide

## 🚨 Common Issues & Solutions

### 1. Installation Issues

#### ❌ `ncnn-python` installation fails
```bash
# Solution 1: Use pre-built wheel
pip install ncnn-python --no-binary :all:

# Solution 2: Install dependencies first
sudo apt install cmake build-essential
pip install ncnn-python

# Solution 3: Use system package (if available)
sudo apt install python3-ncnn
```

#### ❌ `picamera2` not found
```bash
# Install system package
sudo apt install python3-picamera2 python3-libcamera

# Enable camera interface
sudo raspi-config nonint do_camera 0
sudo reboot
```

#### ❌ OpenCV import error
```bash
# Install system dependencies
sudo apt install libopencv-dev python3-opencv

# Or reinstall
pip uninstall opencv-python opencv-python-headless
pip install opencv-python-headless==4.8.0.74
```

### 2. Camera Issues

#### ❌ No camera detected
```bash
# Check camera connection
libcamera-hello --list-cameras

# Enable camera interface
sudo raspi-config nonint do_camera 0

# Check camera module
vcgencmd get_camera

# Expected output: supported=1 detected=1
```

#### ❌ Camera permission denied
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again
logout
```

#### ❌ Camera already in use
```bash
# Kill existing camera processes
sudo pkill -f libcamera
sudo pkill -f picamera

# Check running processes
ps aux | grep camera
```

### 3. Performance Issues

#### 🐌 FPS too low (< 3 FPS)

**Solution 1: Reduce input resolution**
```python
# In rpi_inference_ncnn.py, change:
self.input_size = 320  # instead of 416
```

**Solution 2: Increase confidence threshold**
```python
# In rpi_inference_ncnn.py, change:
conf_threshold=0.4  # instead of 0.25
```

**Solution 3: Skip frames**
```python
# Process every 2nd frame
if frame_count % 2 == 0:
    continue
```

**Solution 4: Optimize NCNN settings**
```python
# Reduce threads if overheating
self.net.opt.num_threads = 2  # instead of 4

# Disable some optimizations
self.net.opt.use_fp16_arithmetic = False
self.net.opt.use_int8_arithmetic = False
```

#### 🔥 Overheating issues
```bash
# Check temperature
vcgencmd measure_temp

# If > 70°C, add cooling or reduce performance:
# 1. Add heatsink/fan
# 2. Reduce CPU threads
# 3. Lower input resolution
# 4. Increase frame skip
```

#### 💾 Out of memory
```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change: CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Check memory usage
free -h
```

### 4. Model Issues

#### ❌ NCNN model loading fails
```bash
# Check model files exist
ls -lh models/
# Should see: best_cropped.param, best_cropped.bin

# Verify file sizes
# param: ~10KB, bin: ~6MB

# Regenerate NCNN model
python convert_to_ncnn.py
```

#### ❌ ONNX model missing
```bash
# Copy from learning_based directory
cp ../learning_based/models/best_cropped.onnx models/

# Or download from training environment
scp user@training-pc:~/wire_defect/models/best_cropped.onnx models/
```

### 5. Network/Connectivity Issues

#### ❌ SSH connection fails
```bash
# Enable SSH
sudo systemctl enable ssh
sudo systemctl start ssh

# Check SSH status
sudo systemctl status ssh

# Find Pi IP address
hostname -I
```

#### ❌ File transfer fails
```bash
# Using SCP
scp -r shipping/* pi@raspberrypi.local:~/wire_defect/

# Using rsync (faster for large files)
rsync -avz shipping/ pi@raspberrypi.local:~/wire_defect/

# Using USB drive
sudo mount /dev/sda1 /mnt/usb
cp -r /mnt/usb/shipping/* ~/wire_defect/
```

### 6. System Issues

#### ❌ Raspberry Pi won't boot
1. Check SD card connection
2. Verify power supply (5V/2.5A minimum)
3. Check SD card health
4. Re-flash Raspberry Pi OS if needed

#### ❌ Package installation fails
```bash
# Update package lists
sudo apt update

# Fix broken packages
sudo apt --fix-broken install

# Clean package cache
sudo apt clean
sudo apt autoremove
```

#### ❌ Python version issues
```bash
# Check Python version (should be 3.7+)
python3 --version

# Install specific Python version if needed
sudo apt install python3.9 python3.9-venv python3.9-dev
```

## 🔧 Diagnostic Commands

### System Health Check
```bash
# Temperature
vcgencmd measure_temp

# Memory
free -h

# CPU usage
top -n 1

# Disk space
df -h

# System info
cat /proc/device-tree/model
```

### Camera Diagnostics
```bash
# List cameras
libcamera-hello --list-cameras

# Test camera
libcamera-hello --timeout 5000

# Camera info
vcgencmd get_camera
```

### Network Diagnostics
```bash
# IP address
hostname -I

# Network interfaces
ip addr show

# Connectivity test
ping google.com -c 3
```

## 📊 Performance Optimization

### Raspberry Pi 3B Optimizations
```python
# Optimal settings for RPi 3B
self.input_size = 320           # Reduce from 416
self.conf_threshold = 0.35      # Increase from 0.25
self.net.opt.num_threads = 2    # Reduce from 4
process_every_nth_frame = 2     # Skip frames
```

### Raspberry Pi 4 Optimizations
```python
# Optimal settings for RPi 4
self.input_size = 416           # Keep at 416
self.conf_threshold = 0.25      # Keep at 0.25
self.net.opt.num_threads = 4    # Use all cores
process_every_nth_frame = 1     # Process all frames
```

## 🆘 Getting Help

### Before Asking for Help
1. Run the test script: `python test_deployment.py`
2. Check system temperature: `vcgencmd measure_temp`
3. Verify model files: `ls -lh models/`
4. Test camera: `libcamera-hello --timeout 5000`

### Collecting Debug Information
```bash
# System info
uname -a
cat /proc/device-tree/model
python3 --version

# Package versions
pip list | grep -E "(opencv|numpy|ncnn|picamera)"

# Hardware info
vcgencmd get_camera
vcgencmd measure_temp
free -h

# Error logs
journalctl -u your-service --since "1 hour ago"
```

### Performance Baseline
- **Raspberry Pi 3B**: 3-6 FPS expected
- **Raspberry Pi 4**: 6-12 FPS expected  
- **Input 416x416**: ~200-400ms per frame
- **Input 320x320**: ~100-250ms per frame

If performance is significantly below these numbers, check for:
- Overheating (>70°C)
- Insufficient power supply
- SD card bottleneck
- Background processes consuming CPU
