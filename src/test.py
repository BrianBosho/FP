#!/usr/bin/env python
"""
Test script to verify that code refactoring doesn't break existing functionality.
This runs a miniature version of the experiments with minimal computation.
"""
import os
import sys
import torch
import numpy as np
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correct imports
from src.experiments.run_utils import (
    setup_logging, 
    log_training_results, 
    log_evaluation_results, 
    save_results_to_csv,
    compare_model_parameters,
    prepare_results_data,
    compute_experiment_statistics,
    generate_experiment_output
)

# Import from core module
from core import load_configuration, get_device, instantiate_model

# Import from run module
from run import main_experiment

# Import from dataprocessing module
from dataprocessing.loaders import (
    load_and_split_with_khop,
    load_and_split_with_feature_prop
)

def test_imports():
    """Test that all necessary imports work correctly"""
    logger.info("Testing imports...")
    try:
        # Import from core module
        from core import load_configuration, get_device, instantiate_model
        
        # Import from run module
        from run import main_experiment
        
        # Import from dataprocessing module
        from dataprocessing.loaders import (
            load_and_split_with_khop,
            load_and_split_with_feature_prop
        )
        
        logger.info("✓ All imports successful")
        return True, (load_configuration, main_experiment)
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False, None

def test_configuration(load_configuration):
    """Test that configuration loading works"""
    logger.info("Testing configuration loading...")
    try:
        config_path = "conf/base.yaml"
        if not os.path.exists(config_path):
            # Try relative path if we're in src directory
            config_path = "../conf/base.yaml"
            if not os.path.exists(config_path):
                logger.error(f"✗ Configuration file not found at: {config_path}")
                return False, None
                
        clients_num, beta, cfg = load_configuration(config_path)
        logger.info(f"✓ Configuration loaded: {clients_num} clients, beta={beta}")
        return True, (clients_num, beta, cfg)
    except Exception as e:
        logger.error(f"✗ Configuration loading error: {e}")
        return False, None

def test_device():
    """Test that device detection works correctly"""
    logger.info("Testing device detection...")
    try:
        from core import get_device
        device = get_device()
        logger.info(f"✓ Device detected: {device}")
        return True, device
    except Exception as e:
        logger.error(f"✗ Device detection error: {e}")
        return False, None

def test_model_instantiation():
    """Test that model instantiation works correctly"""
    logger.info("Testing model instantiation...")
    try:
        from core import instantiate_model, get_device
        device = get_device()
        
        # Test with dummy values
        model = instantiate_model("GCN", num_features=10, num_classes=7, device=device)
        logger.info(f"✓ Model instantiated: {type(model).__name__}")
        return True
    except Exception as e:
        logger.error(f"✗ Model instantiation error: {e}")
        return False

def test_mini_experiment(main_experiment, cfg):
    """Run a minimal experiment to test core functionality"""
    logger.info("Running minimal experiment...")
    try:
        # Reduce these parameters to make the test faster
        cfg["num_rounds"] = 1  # Just 1 round for testing
        cfg["epochs"] = 1  # Just 1 epoch per client
        
        # Test with smallest dataset and simplest settings
        results_data, result_text = main_experiment(
            clients_num=2,  # Use only 2 clients
            beta=0.5,
            data_loading_option="zero_hop",
            model_type="GCN",
            cfg=cfg,
            dataset_name="Cora",  # Smallest dataset
            hop=1,
            fulltraining_flag=False
        )
        
        # Verify the structure of the results
        if not isinstance(results_data, dict):
            logger.error("✗ Experiment result is not a dictionary")
            return False
            
        if "experiment_config" not in results_data or "rounds" not in results_data or "summary" not in results_data:
            logger.error("✗ Experiment result missing expected keys")
            return False
            
        logger.info("✓ Minimal experiment completed successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Experiment error: {e}")
        return False

def run_tests():
    """Run all tests and report results"""
    start_time = datetime.now()
    logger.info(f"Starting tests at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test imports
    imports_ok, functions = test_imports()
    if not imports_ok:
        logger.error("Import tests failed. Aborting further tests.")
        return False
    
    load_configuration, main_experiment = functions
    
    # Test configuration
    config_ok, config_data = test_configuration(load_configuration)
    if not config_ok:
        logger.error("Configuration tests failed. Aborting further tests.")
        return False
    
    # Test device detection
    device_ok, _ = test_device()
    if not device_ok:
        logger.error("Device detection test failed.")
        return False
    
    # Test model instantiation
    model_ok = test_model_instantiation()
    if not model_ok:
        logger.error("Model instantiation test failed.")
        return False
    
    clients_num, beta, cfg = config_data
    
    # Test minimal experiment
    experiment_ok = test_mini_experiment(main_experiment, cfg)
    if not experiment_ok:
        logger.error("Experiment test failed.")
        return False
    
    # All tests passed
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"All tests passed successfully in {duration:.2f} seconds.")
    return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test federated learning code integrity")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Check for CUDA using our get_device function
    if args.cuda:
        from core import get_device
        device = get_device()
    else:
        device = torch.device("cpu")
        logger.info(f"Forcing CPU device: {device}")
    
    # Run tests
    success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)