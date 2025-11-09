# Wire Defect Detection – Jetson Nano Deployment

## 1. Prerequisites
- **Hardware:** NVIDIA Jetson Nano (4 GB recommended) running JetPack 4.6+ (CUDA & cuDNN preinstalled)
- **Camera:**
  - USB/UVC camera (default `/dev/video0`), or
  - CSI camera via ribbon cable (use the provided GStreamer pipeline)
- **Network:** Internet access (for APT + pip installs)
- **Repo:** `shipping/` folder copied to the device (contains model + scripts)

> Raspberry Pi support has been removed. All scripts and docs now target Jetson Nano.

## 2. One-Time Environment Setup
Automated helper (installs system packages, creates `shipping/venv`, installs Python deps, validation checks):
```bash
cd shipping
chmod +x setup_environment.sh
./setup_environment.sh
```
The script will:
1. Install Jetson-friendly APT packages (Python, OpenCV, GStreamer, build tools)
2. Create a virtual environment **with system site-packages** (`python3 -m venv --system-site-packages venv`)
3. Install Python packages from `requirements_simple.txt`
4. Install a Jetson-specific `onnxruntime-gpu` wheel (defaults to NVIDIA’s download for the detected Python version—override with `ONNXRUNTIME_GPU_WHEEL=<path-or-url>`)
5. Run smoke tests (imports, model loading, `/dev/video0` capture)

Manual steps (if you prefer full control):
```bash
sudo apt update
sudo apt install python3-venv python3-pip python3-dev python3-opencv python3-numpy \
                 python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
                 build-essential cmake pkg-config libopenblas-dev liblapack-dev \
                 v4l-utils gstreamer1.0-tools gstreamer1.0-plugins-base \
                 gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav

cd shipping
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements_simple.txt
# install NVIDIA's onnxruntime-gpu wheel (pick the version matching your JetPack)
export ONNXRUNTIME_GPU_WHEEL="<path-or-url-to-wheel>"
python -m pip install --no-cache-dir "$ONNXRUNTIME_GPU_WHEEL"
```

After the script (or manual steps) complete, verify that the GStreamer Python bindings are visible inside the virtual environment:
```bash
source venv/bin/activate
python -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; print('GStreamer bindings OK')"
```

## 3. Validate With Sample Images
```bash
source venv/bin/activate
python test_with_images.py
```
- Loads `models/best_cropped.onnx`
- Saves annotated outputs to `shipping/test_results/`
- Expected summary: 5 images → 19 detections (`fail/pagan/valid` counts match console output)

## 4. Run Live Detection
### 4.1 USB / Webcam (V4L2 backend)
```bash
python run_camera_detection.py --source 0 --width 1280 --height 720 --fps 30 --display
```

### 4.2 CSI Camera (GStreamer pipeline)
```bash
python run_camera_detection.py \
  --use-gstreamer \
  --source "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink" \
  --display
```

### 4.3 When OpenCV reports `GStreamer: NO`
JetPack 4.x ships with the legacy OpenCV 3.2.0 build which is **compiled without GStreamer bindings**.  
Symptoms inside `run_camera_detection.py`:
- `cv2.getBuildInformation()` prints `GStreamer: NO`
- `VideoCapture("nvarguscamerasrc ! ...")` always fails even though `gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! xvimagesink` works
- The script falls back to USB/V4L2 paths and ends up reading zero frames

**Workaround provided here**
- The setup script installs PyGObject (`python3-gi`) plus the GStreamer GIR bindings
- The live detection script detects the missing OpenCV support and switches to a pure-GStreamer capture path (using `gi.repository.Gst` + `AppSink`) so CSI cameras stream reliably
- Just run the usual command (`python run_camera_detection.py --source 0 ...`) and the fallback engages automatically when OpenCV lacks GStreamer
- If you later rebuild or replace OpenCV (e.g. install OpenCV 4.x with GStreamer), re-run `./setup_environment.sh` so the virtualenv picks up the new system package

**Quick camera validation after setup**
```bash
source venv/bin/activate
python run_camera_detection.py --source 0 --width 1280 --height 720 --fps 30 --warmup 0 --display
```
You should see the FPS counter increasing and annotated frames in the preview window. Press **Ctrl+C** or **q** to stop.

**Alternative** — upgrade OpenCV: rebuild or install an OpenCV 4.x package that enables GStreamer (`WITH_GSTREAMER=ON`). After replacing OpenCV, re-run `setup_environment.sh` so the virtualenv sees the new install.

Optional flags:
- `--warmup <n>`: skip the first *n* frames before collecting stats (default 5)
- `--display`: show annotated frames (press **q** to stop)

Console output prints rolling FPS and class counts every ~10 analysed frames.

## 5. Troubleshooting
| Issue | Fix |
|-------|------|
| `onnxruntime-gpu` wheel unavailable | Download the NVIDIA Jetson wheel matching your Python version and set `ONNXRUNTIME_GPU_WHEEL=/path/to/onnxruntime_gpu.whl` before running the setup script |
| Cannot open `/dev/video0` | Check `v4l2-ctl --list-devices`, ensure user in `video` group (`sudo usermod -aG video $USER && sudo reboot`) |
| CSI pipeline errors | Restart drivers: `sudo systemctl restart nvargus-daemon` |
| Low FPS | Lower `self.input_size` (e.g. 320) in both scripts, or reduce resolution/FPS flags |
| `ModuleNotFoundError` after reboot | Re-activate env: `source shipping/venv/bin/activate`

## 6. Repository Layout
```
shipping/
├── models/best_cropped.onnx       # YOLOv8 ONNX model
├── setup_environment.sh           # Jetson Nano setup helper (APT + pip)
├── requirements_simple.txt        # Python dependencies
├── test_with_images.py            # Batch validation
├── run_camera_detection.py        # Live detection entry point (USB / CSI)
├── test_images/                   # Sample frames
├── test_results/                  # Output directory (auto-created)
└── README_SIMPLE.md               # This guide (Jetson Nano)
```

You’re ready: run the setup script once, validate with `test_with_images.py`, then start live detection using the command that matches your camera. 🚀
