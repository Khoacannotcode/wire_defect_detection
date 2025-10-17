#!/usr/bin/env python3
"""
Convert ONNX model to NCNN format
This script helps convert the ONNX model to NCNN format for Raspberry Pi deployment
"""

import os
import sys
import subprocess
from pathlib import Path

def convert_onnx_to_ncnn():
    """Convert ONNX model to NCNN format"""
    
    print("=== Converting ONNX to NCNN Format ===")
    print()
    
    # Check if ONNX model exists
    onnx_path = "models/best_cropped.onnx"
    if not os.path.exists(onnx_path):
        print(f"❌ Error: ONNX model not found at {onnx_path}")
        print("Please make sure best_cropped.onnx is in the models/ directory")
        return False
    
    # Output paths
    param_path = "models/best_cropped.param"
    bin_path = "models/best_cropped.bin"
    
    print(f"📁 Input ONNX: {onnx_path}")
    print(f"📁 Output param: {param_path}")
    print(f"📁 Output bin: {bin_path}")
    print()
    
    # Check for NCNN tools
    ncnn_tools = [
        "onnx2ncnn",
        "./onnx2ncnn",
        "ncnn-tools/onnx2ncnn",
        "../ncnn-tools/onnx2ncnn"
    ]
    
    onnx2ncnn_path = None
    for tool_path in ncnn_tools:
        try:
            result = subprocess.run([tool_path, "--help"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                onnx2ncnn_path = tool_path
                break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    if not onnx2ncnn_path:
        print("❌ onnx2ncnn tool not found!")
        print()
        print("Please download NCNN tools from:")
        print("https://github.com/Tencent/ncnn/releases")
        print()
        print("Then extract and place onnx2ncnn in one of these locations:")
        for tool_path in ncnn_tools:
            print(f"  - {tool_path}")
        print()
        print("Or run manually:")
        print(f"onnx2ncnn {onnx_path} {param_path} {bin_path}")
        return False
    
    print(f"✅ Found onnx2ncnn tool: {onnx2ncnn_path}")
    print()
    
    # Convert ONNX to NCNN
    print("🔄 Converting ONNX to NCNN...")
    try:
        result = subprocess.run([
            onnx2ncnn_path,
            onnx_path,
            param_path,
            bin_path
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Conversion successful!")
        else:
            print(f"❌ Conversion failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Conversion timed out!")
        return False
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False
    
    # Check output files
    if os.path.exists(param_path) and os.path.exists(bin_path):
        param_size = os.path.getsize(param_path) / 1024
        bin_size = os.path.getsize(bin_path) / (1024 * 1024)
        
        print()
        print("📊 Model Info:")
        print(f"  Param file: {param_size:.1f} KB")
        print(f"  Bin file: {bin_size:.1f} MB")
        print(f"  Total size: {bin_size:.1f} MB")
        
        # Try to optimize with ncnnoptimize
        print()
        print("🔧 Attempting to optimize model...")
        
        ncnnoptimize_tools = [
            "ncnnoptimize",
            "./ncnnoptimize", 
            "ncnn-tools/ncnnoptimize",
            "../ncnn-tools/ncnnoptimize"
        ]
        
        optimize_tool = None
        for tool_path in ncnnoptimize_tools:
            try:
                result = subprocess.run([tool_path, "--help"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    optimize_tool = tool_path
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        if optimize_tool:
            try:
                result = subprocess.run([
                    optimize_tool,
                    param_path,
                    bin_path,
                    param_path,
                    bin_path,
                    "65536"
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print("✅ Model optimized successfully!")
                else:
                    print("⚠️  Optimization failed, but model is still usable")
                    
            except Exception as e:
                print(f"⚠️  Optimization error: {e}")
        else:
            print("⚠️  ncnnoptimize tool not found, skipping optimization")
        
        print()
        print("🎉 NCNN model ready for Raspberry Pi deployment!")
        return True
        
    else:
        print("❌ Output files not created properly")
        return False

def main():
    """Main function"""
    print("Wire Defect Detection - ONNX to NCNN Converter")
    print("=" * 50)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("models"):
        print("❌ Error: 'models' directory not found!")
        print("Please run this script from the shipping directory")
        return 1
    
    success = convert_onnx_to_ncnn()
    
    if success:
        print()
        print("📋 Next steps:")
        print("1. Copy this entire shipping/ folder to Raspberry Pi")
        print("2. Run: ./setup_rpi.sh")
        print("3. Run: python rpi_inference_ncnn.py")
        return 0
    else:
        print()
        print("❌ Conversion failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
