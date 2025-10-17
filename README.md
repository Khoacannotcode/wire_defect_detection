# Wire Defect Detection - Raspberry Pi Deployment

## 📦 Complete Deployment Package

This package contains everything needed to deploy wire defect detection on Raspberry Pi 3B using NCNN framework.

## 🚀 Quick Start

### 1. Hardware Requirements
- Raspberry Pi 3B (1GB RAM)
- Raspberry Pi Camera Module v2/v3
- MicroSD card 16GB+ (recommended 32GB)
- Power supply 5V/2.5A

### 2. Transfer Files
Copy this entire folder to Raspberry Pi:
```bash
# Option 1: SCP
scp -r shipping/* pi@raspberrypi.local:~/wire_defect/

# Option 2: USB drive
cp -r /media/usb/shipping/* ~/wire_defect/
```

### 3. Setup
```bash
cd ~/wire_defect
chmod +x setup_rpi.sh
./setup_rpi.sh
```

### 4. Convert Model (if needed)
```bash
python convert_to_ncnn.py
```

### 5. Validate Deployment
```bash
source venv/bin/activate
python test_deployment.py
```

### 6. Run Performance Benchmark
```bash
python benchmark.py
```

### 7. Start Real-time Detection
```bash
python rpi_inference_ncnn.py
```

## 📊 Performance

### Raspberry Pi 3B + NCNN:
- **FPS**: 3-6 FPS
- **Inference time**: 170-330ms per frame
- **Memory usage**: ~150MB
- **Model size**: ~4MB

### Output Format:
```
[NCNN] FPS: 4.2 | Total: 1523 | Fail: 1234 | Pagan: 156 | Valid: 133
```

## 🔧 Optimization Tips

### If FPS is too low:

1. **Reduce resolution:**
```python
# In rpi_inference_ncnn.py
self.input_size = 320  # instead of 416
```

2. **Increase confidence threshold:**
```python
conf_threshold=0.4  # instead of 0.25
```

3. **Skip frames:**
```python
if frame_count % 2 == 0:  # Process every 2nd frame
    continue
```

## 🛠️ Troubleshooting

### Camera not working:
```bash
libcamera-hello --list-cameras
sudo raspi-config nonint do_camera 0
sudo reboot
```

### NCNN import error:
```bash
pip uninstall ncnn-python
pip install ncnn-python --no-binary :all:
```

### Out of memory:
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## 📁 File Structure

```
shipping/
├── README.md                    # This guide
├── TROUBLESHOOTING.md          # Detailed troubleshooting guide
├── DEPLOYMENT_CHECKLIST.md     # Step-by-step deployment checklist
├── setup_rpi.sh                # Automated setup script
├── rpi_inference_ncnn.py       # Main inference script
├── convert_to_ncnn.py          # ONNX to NCNN converter
├── test_deployment.py          # Deployment validation script
├── benchmark.py                # Performance benchmark tool
├── requirements.txt            # Python dependencies
├── models/
│   ├── best_cropped.onnx       # ONNX model (intermediate)
│   ├── best_cropped.param      # NCNN model architecture
│   ├── best_cropped.bin        # NCNN model weights
│   └── cropped_info.txt        # Model documentation
└── test_images/                # Sample images for testing
    ├── 018476.jpg              # Sample fail detection
    ├── 0bef58.jpg              # Sample multi-class
    └── ... (50 test images)
```

## 🎯 Why NCNN?

NCNN (Tencent) is optimized for mobile/edge devices:
- ✅ ARM NEON instructions support
- ✅ Efficient INT8 quantization
- ✅ Memory-efficient layout
- ✅ No GPU required
- ✅ 2-4x faster than ONNX Runtime on ARM

## 📞 Support & Documentation

### Quick Diagnostics
1. **Run validation**: `python test_deployment.py`
2. **Check performance**: `python benchmark.py --runs 10`
3. **Test camera**: `libcamera-hello --timeout 5000`
4. **Monitor system**: `vcgencmd measure_temp && free -h`

### Detailed Guides
- **TROUBLESHOOTING.md** - Complete troubleshooting guide
- **DEPLOYMENT_CHECKLIST.md** - Step-by-step deployment checklist
- **README.md** - This overview guide

### Performance Expectations
- **Raspberry Pi 3B**: 3-6 FPS, 200-400ms per frame
- **Raspberry Pi 4**: 6-12 FPS, 100-250ms per frame
- **Memory usage**: ~150MB
- **Model size**: ~6MB total

---
**Package Version**: 1.0  
**Compatible with**: Raspberry Pi 3B, Pi 4, Pi 5
