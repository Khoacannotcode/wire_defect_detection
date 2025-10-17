#!/usr/bin/env python3
"""
Smart launcher for Wire Defect Detection
Automatically chooses the best available inference engine
"""

import sys
import os
import subprocess

def check_ncnn_available():
    """Check if NCNN is available and working"""
    try:
        import ncnn
        
        # Check if model files exist
        param_path = "models/best_cropped.param"
        bin_path = "models/best_cropped.bin"
        
        if os.path.exists(param_path) and os.path.exists(bin_path):
            # Try to load model
            net = ncnn.Net()
            ret1 = net.load_param(param_path)
            ret2 = net.load_model(bin_path)
            
            if ret1 == 0 and ret2 == 0:
                return True
        
        return False
    except ImportError:
        return False
    except Exception:
        return False

def check_onnx_available():
    """Check if ONNX Runtime is available and working"""
    try:
        import onnxruntime as ort
        
        model_path = "models/best_cropped.onnx"
        if os.path.exists(model_path):
            # Try to create session
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            return True
        
        return False
    except ImportError:
        return False
    except Exception:
        return False

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🚀 Wire Defect Detection - Smart Launcher")
    print("=" * 60)
    print()
    
    # Check available inference engines
    ncnn_available = check_ncnn_available()
    onnx_available = check_onnx_available()
    
    print("🔍 Checking available inference engines...")
    print(f"  NCNN: {'✅ Available' if ncnn_available else '❌ Not available'}")
    print(f"  ONNX Runtime: {'✅ Available' if onnx_available else '❌ Not available'}")
    print()
    
    # Choose inference engine
    if ncnn_available:
        print("🎯 Using NCNN inference (optimal performance)")
        script = "rpi_inference_ncnn.py"
        expected_fps = "3-6 FPS (RPi 3B), 6-12 FPS (RPi 4)"
    elif onnx_available:
        print("🔄 Using ONNX Runtime inference (fallback)")
        script = "rpi_inference_onnx.py"
        expected_fps = "1-3 FPS (RPi 3B), 2-6 FPS (RPi 4)"
    else:
        print("❌ No inference engine available!")
        print()
        print("Please install dependencies:")
        print("  1. Run: ./setup_rpi.sh")
        print("  2. Or manually: pip install onnxruntime")
        print("  3. For NCNN: see TROUBLESHOOTING.md")
        return 1
    
    print(f"📊 Expected performance: {expected_fps}")
    print()
    print(f"🚀 Starting {script}...")
    print("Press Ctrl+C to stop detection")
    print("=" * 60)
    print()
    
    # Launch the selected inference script
    try:
        result = subprocess.run([sys.executable, script])
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nDetection stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error launching {script}: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
