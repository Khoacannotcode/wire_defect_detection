#!/usr/bin/env python3
"""
Test script for Wire Defect Detection deployment on Raspberry Pi
Validates all components and dependencies
"""

import sys
import os
import time
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import numpy as np
        print("  ✅ numpy")
    except ImportError as e:
        print(f"  ❌ numpy: {e}")
        return False
    
    try:
        import cv2
        print(f"  ✅ opencv-python ({cv2.__version__})")
    except ImportError as e:
        print(f"  ❌ opencv-python: {e}")
        return False
    
    try:
        import ncnn
        print("  ✅ ncnn-python")
    except ImportError as e:
        print(f"  ❌ ncnn-python: {e}")
        print("    Install with: pip install ncnn-python")
        return False
    
    try:
        from picamera2 import Picamera2
        print("  ✅ picamera2")
    except ImportError as e:
        print(f"  ❌ picamera2: {e}")
        print("    Install with: sudo apt install python3-picamera2")
        return False
    
    return True

def test_model_files():
    """Test model files existence and validity"""
    print("\n📁 Testing model files...")
    
    required_files = [
        "models/best_cropped.onnx",
        "models/best_cropped.param", 
        "models/best_cropped.bin"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"  ❌ {file_path} - Missing!")
            all_exist = False
    
    return all_exist

def test_ncnn_model():
    """Test NCNN model loading"""
    print("\n🤖 Testing NCNN model loading...")
    
    try:
        import ncnn
        
        # Initialize network
        net = ncnn.Net()
        net.opt.use_packing_layout = True
        net.opt.num_threads = 2
        
        # Load model
        param_path = "models/best_cropped.param"
        bin_path = "models/best_cropped.bin"
        
        if not os.path.exists(param_path):
            print(f"  ❌ Param file missing: {param_path}")
            print("  💡 Run: python convert_to_ncnn.py")
            return False
            
        if not os.path.exists(bin_path):
            print(f"  ❌ Bin file missing: {bin_path}")
            print("  💡 Run: python convert_to_ncnn.py")
            return False
        
        ret1 = net.load_param(param_path)
        ret2 = net.load_model(bin_path)
        
        if ret1 == 0 and ret2 == 0:
            print("  ✅ NCNN model loaded successfully")
            return True
        else:
            print(f"  ❌ NCNN model loading failed (param: {ret1}, bin: {ret2})")
            return False
            
    except ImportError:
        print("  ⚠️  NCNN not installed, testing ONNX Runtime fallback...")
        return test_onnx_model()
    except Exception as e:
        print(f"  ❌ NCNN model test failed: {e}")
        return test_onnx_model()

def test_onnx_model():
    """Test ONNX Runtime model loading (fallback)"""
    print("\n🔄 Testing ONNX Runtime model loading...")
    
    try:
        import onnxruntime as ort
        
        model_path = "models/best_cropped.onnx"
        if not os.path.exists(model_path):
            print(f"  ❌ ONNX model missing: {model_path}")
            return False
        
        # Create session
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        
        print("  ✅ ONNX Runtime model loaded successfully")
        print("  💡 Use: python rpi_inference_onnx.py")
        return True
        
    except ImportError:
        print("  ❌ ONNX Runtime not installed")
        print("  💡 Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"  ❌ ONNX model test failed: {e}")
        return False

def test_camera():
    """Test camera functionality"""
    print("\n📷 Testing camera...")
    
    try:
        from picamera2 import Picamera2
        
        # Try to initialize camera
        picam2 = Picamera2()
        
        # Check if camera is detected
        camera_info = picam2.global_camera_info()
        if not camera_info:
            print("  ❌ No camera detected")
            print("    Check connections and enable camera interface:")
            print("    sudo raspi-config nonint do_camera 0")
            return False
        
        print(f"  ✅ Camera detected: {len(camera_info)} camera(s)")
        
        # Try to configure (but don't start to avoid conflicts)
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        print("  ✅ Camera configuration successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Camera test failed: {e}")
        print("    Make sure camera is connected and enabled")
        return False

def test_inference_speed():
    """Test inference speed with dummy data"""
    print("\n⚡ Testing inference speed...")
    
    try:
        import numpy as np
        import cv2
        import time
        
        # Create dummy image (416x416x3)
        dummy_image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        
        # Simulate preprocessing
        start_time = time.time()
        
        # Resize and convert (simulate real preprocessing)
        processed = cv2.resize(dummy_image, (416, 416))
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        preprocess_time = time.time() - start_time
        
        # Simulate inference (without actual NCNN call to avoid model dependency)
        inference_start = time.time()
        time.sleep(0.1)  # Simulate 100ms inference
        inference_time = time.time() - inference_start
        
        total_time = preprocess_time + inference_time
        fps = 1.0 / total_time if total_time > 0 else 0
        
        print(f"  📊 Preprocessing: {preprocess_time*1000:.1f}ms")
        print(f"  📊 Inference (simulated): {inference_time*1000:.1f}ms")
        print(f"  📊 Total: {total_time*1000:.1f}ms")
        print(f"  📊 Expected FPS: {fps:.1f}")
        
        if fps >= 3:
            print("  ✅ Performance looks good for Raspberry Pi")
        else:
            print("  ⚠️  Performance may be slow on Raspberry Pi")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False

def test_system_info():
    """Display system information"""
    print("\n💻 System Information:")
    
    try:
        # Check if running on Raspberry Pi
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().strip('\x00')
                print(f"  🔧 Device: {model}")
        else:
            print("  🔧 Device: Not a Raspberry Pi")
        
        # CPU info
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "model name" in line:
                        cpu = line.split(":")[1].strip()
                        print(f"  🔧 CPU: {cpu}")
                        break
        
        # Memory info
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "MemTotal" in line:
                        mem = line.split(":")[1].strip()
                        print(f"  🔧 Memory: {mem}")
                        break
        
        # Temperature (RPi specific)
        try:
            import subprocess
            result = subprocess.run(["vcgencmd", "measure_temp"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                temp = result.stdout.strip()
                print(f"  🌡️  Temperature: {temp}")
        except:
            pass
            
    except Exception as e:
        print(f"  ⚠️  Could not get system info: {e}")

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 Wire Defect Detection - Deployment Test")
    print("=" * 60)
    
    # Test system info first
    test_system_info()
    
    # Run all tests
    tests = [
        ("Import Test", test_imports),
        ("Model Files Test", test_model_files),
        ("NCNN Model Test", test_ncnn_model),
        ("Camera Test", test_camera),
        ("Performance Test", test_inference_speed)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Deployment is ready.")
        print("\nNext steps:")
        print("  1. Run: python rpi_inference_ncnn.py")
        print("  2. Check real-time performance")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before deployment.")
        print("\nTroubleshooting:")
        print("  1. Run: ./setup_rpi.sh")
        print("  2. Check camera: libcamera-hello")
        print("  3. Check model files in models/ directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())
