#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script tests that all imports are working correctly

import sys
import os

print("Starting import test...")
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
print(f"Added project root to path: {project_root}")

# Try importing key modules one by one
try:
    print("Importing src.run...")
    from src.run import load_configuration, main_experiment
    print("Importing src.client...")
    from src.client import FLClient
    print("Importing src.models...")
    from src.models import GCN, GAT
    print("Importing src.server...")
    from src.server import Server
    print("Importing src.utils...")
    from src.utils import load_config
    
    print("[SUCCESS] All core modules imported successfully!")
    
    # Test config loading
    try:
        print("Testing configuration loading...")
        clients_num, beta, cfg = load_configuration("conf/base.yaml")
        print(f"[SUCCESS] Configuration loaded successfully!")
        print(f"  - Number of clients: {clients_num}")
        print(f"  - Beta: {beta}")
        print(f"  - Full training flag: {cfg.get('fulltraining_flag', False)}")
    except Exception as e:
        print(f"[ERROR] Error loading configuration: {str(e)}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"[ERROR] Import error: {str(e)}")
    import traceback
    traceback.print_exc()
    
print("\nTest complete!") 