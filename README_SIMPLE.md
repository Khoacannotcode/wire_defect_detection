# Wire Defect Detection – Jetson Nano Deployment

This document provides a simplified, step-by-step guide to set up and run the wire defect detection project on a Jetson Nano.

## 1. Environment Setup (First Time Only)

The first step is to prepare the Jetson's system environment. We provide a script to automate this process.

**Run the Setup Script:**

Navigate to the `shipping` directory and run the setup script with `sudo`. This will install all necessary system and Python packages, and configure the environment correctly.

```bash
cd /path/to/your/project/shipping
sudo ./setup_environment.sh
```

This script will:
- Remove any old virtual environments to prevent conflicts.
- Install `python3-pip`.
- Install required Python libraries like `numpy` and `pycuda` into the system's Python environment.
- Ensure that the system's hardware-accelerated OpenCV is used.

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
| `ModuleNotFoundError` | Run the setup script again to ensure all dependencies are installed: `sudo ./setup_environment.sh` |
| Camera doesn't open | Ensure the camera is connected properly. Check the camera source index in the script if you have multiple cameras. |
