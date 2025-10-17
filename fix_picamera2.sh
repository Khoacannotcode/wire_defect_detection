#!/bin/bash

echo "=============================================="
echo "  Fixing picamera2 Import Issue"
echo "=============================================="
echo ""

# Check if we're in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "❌ No virtual environment detected"
    echo "Please activate first: source venv/bin/activate"
    exit 1
fi

# Method 1: Install picamera2 in virtual environment
echo "Method 1: Installing picamera2 in virtual environment..."
pip install picamera2 --no-deps || {
    echo "Method 1 failed, trying Method 2..."
    
    # Method 2: Link system picamera2 to virtual environment
    echo "Method 2: Linking system picamera2..."
    
    # Find system picamera2 location
    SYSTEM_PICAMERA=$(python3 -c "
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')
try:
    import picamera2
    print(picamera2.__file__)
except:
    print('NOT_FOUND')
" 2>/dev/null)
    
    if [ "$SYSTEM_PICAMERA" != "NOT_FOUND" ] && [ -n "$SYSTEM_PICAMERA" ]; then
        SYSTEM_DIR=$(dirname "$SYSTEM_PICAMERA")
        VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
        
        echo "System picamera2: $SYSTEM_DIR"
        echo "Venv site-packages: $VENV_SITE"
        
        # Create symlink
        ln -sf "$SYSTEM_DIR" "$VENV_SITE/picamera2" 2>/dev/null || {
            echo "Symlink failed, trying copy..."
            cp -r "$SYSTEM_DIR" "$VENV_SITE/picamera2"
        }
        
        echo "✅ Linked system picamera2 to virtual environment"
    else
        echo "Method 2 failed, trying Method 3..."
        
        # Method 3: Add system path to Python
        echo "Method 3: Adding system path to Python..."
        
        VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
        
        # Create pth file to add system packages
        echo "/usr/lib/python3/dist-packages" > "$VENV_SITE/system-packages.pth"
        echo "✅ Added system packages path to virtual environment"
    fi
}

# Test the fix
echo ""
echo "Testing picamera2 import..."
python3 -c "
try:
    from picamera2 import Picamera2
    print('✅ picamera2 import successful!')
except ImportError as e:
    print(f'❌ picamera2 import still failed: {e}')
    print()
    print('Manual solutions:')
    print('1. Deactivate venv and use system Python:')
    print('   deactivate && python3 rpi_inference_onnx.py')
    print()
    print('2. Install picamera2 with dependencies:')
    print('   pip install picamera2 --force-reinstall')
    print()
    print('3. Use system Python entirely:')
    print('   sudo apt install python3-onnxruntime')
    print('   python3 rpi_inference_onnx.py')
"

echo ""
echo "=============================================="
echo "  Fix complete!"
echo "=============================================="
echo ""
echo "Now try running:"
echo "  python rpi_inference_onnx.py"
echo ""
