#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sync model_config.json with actual model files found in models/ directory
Auto-detects ONNX and engine files and updates config accordingly
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
MODEL_CONFIG = SCRIPT_DIR / "model_config.json"

def sync_model_config():
    """Sync model_config.json with actual files"""
    print("=" * 80)
    print("Syncing model_config.json with actual model files")
    print("=" * 80)
    
    if not MODELS_DIR.exists():
        print(f"[ERROR] Models directory not found: {MODELS_DIR}")
        return False
    
    # Find ONNX files
    onnx_files = sorted(list(MODELS_DIR.glob("*.onnx")))
    if not onnx_files:
        print(f"[ERROR] No ONNX files found in {MODELS_DIR}")
        return False
    
    # Find engine files
    engine_files = sorted(list(MODELS_DIR.glob("*.engine")))
    
    print(f"\n[Found Files]")
    print(f"  ONNX files: {[f.name for f in onnx_files]}")
    print(f"  Engine files: {[f.name for f in engine_files]}")
    
    # Select best match
    # Prefer best_v3_416x256, otherwise use first ONNX found
    preferred_onnx = None
    for onnx_file in onnx_files:
        if "best_v3_416x256" in onnx_file.name:
            preferred_onnx = onnx_file
            break
    
    if not preferred_onnx:
        preferred_onnx = onnx_files[0]
    
    # Find corresponding engine
    preferred_engine = None
    engine_name = preferred_onnx.stem + ".engine"
    for engine_file in engine_files:
        if engine_file.name == engine_name:
            preferred_engine = engine_file
            break
    
    # If no exact match, use first engine or create path
    if not preferred_engine and engine_files:
        preferred_engine = engine_files[0]
    
    # Update config
    config = {
        "onnx_model_path": f"models/{preferred_onnx.name}",
        "class_names_path": "models/class_names.txt"
    }
    
    if preferred_engine:
        config["tensorrt_engine_path"] = f"models/{preferred_engine.name}"
    else:
        # Create expected engine path
        config["tensorrt_engine_path"] = f"models/{preferred_onnx.stem}.engine"
        print(f"  [INFO] Engine not found, will be built: {config['tensorrt_engine_path']}")
    
    # Save config
    with open(MODEL_CONFIG, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[Updated Config]")
    print(f"  ONNX: {config['onnx_model_path']}")
    print(f"  Engine: {config['tensorrt_engine_path']}")
    print(f"  Class names: {config['class_names_path']}")
    
    print(f"\n[OK] model_config.json updated successfully")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = sync_model_config()
    exit(0 if success else 1)

