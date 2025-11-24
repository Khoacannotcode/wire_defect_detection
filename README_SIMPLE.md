# Wire Defect Detection – Jetson Nano Deployment

This document provides a simplified, step-by-step guide to set up and run the wire defect detection project on a Jetson Nano.

## 1. Environment Setup (First Time Only)

The first step is to prepare the Jetson's system environment. We provide a script to automate this process.

**Run the Setup Script:**

Navigate to the `shipping` directory and run the setup script with `sudo`. This will install all necessary system and Python packages, and configure the environment correctly.

```bash
cd /path/to/your/project/shipping
sudo ./setup_prerequisites.sh
```

This script will:
- Install `python3-pip` and build tools.
- Install required Python libraries (`numpy`, `pycuda`, `pillow`) into the system's Python environment.
- Ensure that the system's hardware-accelerated OpenCV is used (if OpenCV build script is run separately).

After this script completes, your environment is ready. You do not need to run it again unless the dependencies change.

## 2. Running the Application

There are two main ways to run the detection application.

### Option A: Run with the GUI

To launch the graphical user interface for live detection:

```bash
./run_gui.sh
```

### Option B: Run from the Command Line (No GUI)

To run detection on a live camera feed and see the output in a simple window:

```bash
python3 run_camera_detection.py
```

To run detection on a set of test images:

```bash
python3 test_with_images.py
```

## 3. Troubleshooting

| Problem | Solution |
|---|---|
| `python3: command not found` | Your system may be missing Python 3. This is unlikely on JetPack. |
| `ModuleNotFoundError` (e.g., `No module named 'PIL'`) | Run the setup script to install missing dependencies: `sudo ./setup_prerequisites.sh`. This will install numpy, pycuda, and pillow required for GUI. |
| Camera doesn't open | Ensure the camera is connected properly. Check the camera source index in the script if you have multiple cameras. |
