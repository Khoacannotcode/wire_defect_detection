#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 18, Phase 2: TensorRT Conversion Script
- Converts a given ONNX model to a TensorRT engine.
- Designed to be run directly on the target device (e.g., Jetson Nano).
- Optimizes the engine for FP16 precision to maximize performance.
- Uses relative paths to be independent of the root directory structure.
"""

import tensorrt as trt
from pathlib import Path

# --- Use relative paths based on the script's location ---
# Assumes the following structure on Jetson:
# /some_path/
# ├── trt_converter.py  (this script)
# └── models/
#     ├── best_cropped.onnx
#     └── best_cropped.engine (will be created)

SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_MODEL_PATH = SCRIPT_DIR / "models" / "best_cropped.onnx"
ENGINE_PATH = SCRIPT_DIR / "models" / "best_cropped.engine"

# Logger for TensorRT warnings, errors, and info
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, use_fp16=True):
    """Builds a TensorRT engine from an ONNX file."""
    
    # 1. Create a builder, network, and parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 2. Configure the builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GB
    
    if use_fp16 and builder.platform_has_fast_fp16:
        print("[OK] Platform supports FP16. Building engine with FP16 precision.")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("[INFO] Platform does not support FP16 or it's disabled. Building with FP32.")

    # 3. Parse the ONNX model
    print("\n[1/3] Parsing ONNX model: {}".format(onnx_path))
    if not onnx_path.exists():
        print("[ERROR] ONNX model not found at {}".format(onnx_path))
        return False

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("[ERROR] Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    print("  [OK] ONNX model parsed successfully.")

    # 4. Build the engine
    print("\n[2/3] Building TensorRT engine... (This may take a few minutes)")
    engine = builder.build_engine(network, config)
    if not engine:
        print("[ERROR] Failed to build the TensorRT engine.")
        return False
    print("  [OK] TensorRT engine built successfully.")

    # 5. Serialize and save the engine
    print("\n[3/3] Serializing and saving engine to: {}".format(engine_path))
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print("  [OK] Engine saved. Size: {:.2f} MB".format(engine_path.stat().st_size / 1e6))
    
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("TensorRT Engine Builder")
    print("=" * 60)
    
    # Check if the engine already exists
    if ENGINE_PATH.exists():
        print("[INFO] Engine file already exists at {}. Skipping build.".format(ENGINE_PATH))
        print("Delete the existing file if you want to rebuild.")
    else:
        if build_engine(ONNX_MODEL_PATH, ENGINE_PATH):
            print("\n[OK] Successfully built TensorRT engine!")
        else:
            print("\n[ERROR] Failed to build TensorRT engine.")
