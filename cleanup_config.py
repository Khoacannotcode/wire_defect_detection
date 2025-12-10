#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleanup config.json - Remove model_path field (should only be in model_config.json)
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "config.json"

def cleanup_config(verbose=True):
    """Remove model_path from config.json if present"""
    if not CONFIG_FILE.exists():
        if verbose:
            print(f"[INFO] config.json not found: {CONFIG_FILE}")
            print("[OK] No cleanup needed")
        return True
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if 'model_path' not in config:
            if verbose:
                print("[INFO] config.json does not contain model_path")
                print("[OK] No cleanup needed")
            return True
        
        # Remove model_path
        old_model_path = config.pop('model_path')
        if verbose:
            print(f"[INFO] Removing model_path from config.json: {old_model_path}")
        
        # Save cleaned config
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print("[OK] config.json cleaned successfully")
            print("[INFO] model_path should be in model_config.json, not config.json")
        return True
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to cleanup config.json: {e}")
        return False

if __name__ == "__main__":
    import sys
    verbose = "--quiet" not in sys.argv
    if verbose:
        print("=" * 80)
        print("Config Cleanup - Remove model_path from config.json")
        print("=" * 80)
        print()
    success = cleanup_config(verbose=verbose)
    if verbose:
        print()
        print("=" * 80)
    exit(0 if success else 1)

