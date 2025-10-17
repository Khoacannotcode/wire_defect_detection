#!/usr/bin/env python3
"""
Performance benchmark for Wire Defect Detection on Raspberry Pi
Tests inference speed with real images
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
import argparse

def benchmark_preprocessing(image_path, input_size=416, crop_ratio=0.6, num_runs=50):
    """Benchmark preprocessing performance"""
    print(f"📊 Benchmarking preprocessing ({num_runs} runs)...")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        
        # Crop to ROI (60% center width)
        h, w = image.shape[:2]
        crop_width = int(w * crop_ratio)
        start_x = (w - crop_width) // 2
        end_x = start_x + crop_width
        cropped = image[:, start_x:end_x]
        
        # Resize to input size
        resized = cv2.resize(cropped, (input_size, input_size))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb.astype(np.float32) / 255.0
        
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"  ⏱️  Preprocessing: {avg_time:.1f}ms (min: {min_time:.1f}ms, max: {max_time:.1f}ms)")
    return avg_time

def benchmark_ncnn_inference(image_path, param_path, bin_path, num_runs=20):
    """Benchmark NCNN inference performance"""
    print(f"🤖 Benchmarking NCNN inference ({num_runs} runs)...")
    
    try:
        import ncnn
    except ImportError:
        print("❌ ncnn-python not installed")
        return None
    
    # Initialize NCNN network
    net = ncnn.Net()
    net.opt.use_packing_layout = True
    net.opt.use_fp16_packed = True
    net.opt.use_fp16_storage = True
    net.opt.use_fp16_arithmetic = False
    net.opt.use_int8_storage = True
    net.opt.use_int8_arithmetic = True
    net.opt.num_threads = 4
    
    # Load model
    if not os.path.exists(param_path) or not os.path.exists(bin_path):
        print(f"❌ Model files not found: {param_path}, {bin_path}")
        return None
    
    ret1 = net.load_param(param_path)
    ret2 = net.load_model(bin_path)
    
    if ret1 != 0 or ret2 != 0:
        print(f"❌ Failed to load NCNN model (param: {ret1}, bin: {ret2})")
        return None
    
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Preprocess
    h, w = image.shape[:2]
    crop_width = int(w * 0.6)
    start_x = (w - crop_width) // 2
    end_x = start_x + crop_width
    cropped = image[:, start_x:end_x]
    resized = cv2.resize(cropped, (416, 416))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Create NCNN Mat
    mat_in = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, 416, 416)
    mat_in.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])
    
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        
        # Run inference
        ex = net.create_extractor()
        ex.input("images", mat_in)
        ret, mat_out = ex.extract("output0")
        
        end_time = time.time()
        
        if ret == 0:
            times.append((end_time - start_time) * 1000)  # Convert to ms
        else:
            print(f"⚠️  Inference failed on run {i+1}")
    
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"  ⏱️  NCNN Inference: {avg_time:.1f}ms (min: {min_time:.1f}ms, max: {max_time:.1f}ms)")
        return avg_time
    else:
        print("❌ All inference runs failed")
        return None

def benchmark_end_to_end(image_path, param_path, bin_path, num_runs=10):
    """Benchmark complete end-to-end performance"""
    print(f"🎯 Benchmarking end-to-end performance ({num_runs} runs)...")
    
    try:
        import ncnn
    except ImportError:
        print("❌ ncnn-python not installed")
        return None
    
    # Initialize NCNN network
    net = ncnn.Net()
    net.opt.use_packing_layout = True
    net.opt.use_fp16_packed = True
    net.opt.use_fp16_storage = True
    net.opt.use_fp16_arithmetic = False
    net.opt.use_int8_storage = True
    net.opt.use_int8_arithmetic = True
    net.opt.num_threads = 4
    
    # Load model
    if not os.path.exists(param_path) or not os.path.exists(bin_path):
        print(f"❌ Model files not found")
        return None
    
    ret1 = net.load_param(param_path)
    ret2 = net.load_model(bin_path)
    
    if ret1 != 0 or ret2 != 0:
        print(f"❌ Failed to load NCNN model")
        return None
    
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        
        # Load image
        image = cv2.imread(str(image_path))
        
        # Preprocess
        h, w = image.shape[:2]
        crop_width = int(w * 0.6)
        start_x = (w - crop_width) // 2
        end_x = start_x + crop_width
        cropped = image[:, start_x:end_x]
        resized = cv2.resize(cropped, (416, 416))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Create NCNN Mat
        mat_in = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, 416, 416)
        mat_in.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])
        
        # Run inference
        ex = net.create_extractor()
        ex.input("images", mat_in)
        ret, mat_out = ex.extract("output0")
        
        end_time = time.time()
        
        if ret == 0:
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        
        print(f"  ⏱️  End-to-End: {avg_time:.1f}ms (min: {min_time:.1f}ms, max: {max_time:.1f}ms)")
        print(f"  📈 Expected FPS: {fps:.1f}")
        return avg_time, fps
    else:
        print("❌ All runs failed")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Wire Defect Detection Benchmark')
    parser.add_argument('--image', type=str, default='test_images/018476.jpg',
                       help='Test image path')
    parser.add_argument('--param', type=str, default='models/best_cropped.param',
                       help='NCNN param file')
    parser.add_argument('--bin', type=str, default='models/best_cropped.bin',
                       help='NCNN bin file')
    parser.add_argument('--runs', type=int, default=20,
                       help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("⚡ Wire Defect Detection - Performance Benchmark")
    print("=" * 60)
    
    # Check files
    if not os.path.exists(args.image):
        print(f"❌ Test image not found: {args.image}")
        return 1
    
    print(f"📷 Test image: {args.image}")
    print(f"🤖 Model: {args.param}, {args.bin}")
    print(f"🔄 Runs: {args.runs}")
    print()
    
    # Run benchmarks
    preprocess_time = benchmark_preprocessing(args.image, num_runs=args.runs)
    
    # Only run NCNN benchmarks if model files exist
    if os.path.exists(args.param) and os.path.exists(args.bin):
        inference_time = benchmark_ncnn_inference(args.image, args.param, args.bin, num_runs=args.runs)
        e2e_time, fps = benchmark_end_to_end(args.image, args.param, args.bin, num_runs=args.runs//2)
    else:
        print("⚠️  NCNN model files not found, skipping inference benchmarks")
        inference_time = None
        e2e_time = None
        fps = None
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    if preprocess_time:
        print(f"Preprocessing: {preprocess_time:.1f}ms")
    
    if inference_time:
        print(f"NCNN Inference: {inference_time:.1f}ms")
    
    if e2e_time and fps:
        print(f"End-to-End: {e2e_time:.1f}ms")
        print(f"Expected FPS: {fps:.1f}")
        
        # Performance assessment
        if fps >= 6:
            print("🎉 Excellent performance for Raspberry Pi!")
        elif fps >= 3:
            print("✅ Good performance for Raspberry Pi")
        elif fps >= 1:
            print("⚠️  Acceptable performance, consider optimization")
        else:
            print("❌ Performance too low, optimization needed")
    
    print("\n💡 Optimization tips:")
    print("  - Reduce input size (320x320 instead of 416x416)")
    print("  - Increase confidence threshold")
    print("  - Process every 2nd or 3rd frame")
    print("  - Enable GPU acceleration if available")

if __name__ == "__main__":
    main()
