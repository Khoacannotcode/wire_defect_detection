#!/bin/bash
# Wire Defect Detection - GUI Launcher for Linux/Jetson Nano
# Double-click this file (or run from terminal) to run the GUI application
# Script automatically activates virtual environment and runs GUI

echo "========================================"
echo "Wire Defect Detection - GUI Launcher"
echo "========================================"
echo ""

# Get the directory where this script is located
cd "$(dirname "$0")"

# Check if venv exists
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo ""
    echo "Please run setup_environment.sh first to create the virtual environment."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "[ERROR] Python not found in virtual environment!"
    echo ""
    echo "Please check your virtual environment setup."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if GUI script exists
if [ ! -f "gui_detection_runner.py" ]; then
    echo "[ERROR] gui_detection_runner.py not found!"
    echo ""
    echo "Please ensure you are in the shipping directory."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Run GUI
echo "[INFO] Starting GUI application..."
echo ""
python gui_detection_runner.py

# If GUI exits, keep terminal open to show any errors
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] GUI application exited with an error."
    echo ""
    read -p "Press Enter to exit..."
fi

