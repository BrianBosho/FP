#!/usr/bin/env python3

import yaml
import os
try:
    # Preferred: run as `python -m src.experiments.test_config`
    from src.dataprocessing.partitioning import partition_data
    from src.dataprocessing.datasets import GraphDataset
except ImportError:  # Backward compatibility: run from within `src/`
    from dataprocessing.partitioning import partition_data
    from dataprocessing.datasets import GraphDataset

def test_config_passing():
    """
    Test that positional encoding parameters from config are correctly passed through.
    """
    # Create a test config
    test_config = {
        "use_pe": True,
        "pe_r": 32,  # Different from default 64
        "pe_P": 8,   # Different from default 16
        "normalize": "qr",
        "results_dir": "runs/test_config_passing"
    }
    
    # Ensure results directory exists
    os.makedirs(test_config["results_dir"], exist_ok=True)
    
    print(f"Test config: {test_config}")
    
    # Load dataset
    d = GraphDataset()
    data, dataset = d.load_dataset('Cora', 'cpu')
    print('Dataset loaded')
    
    # Try to partition with config
    try:
        _, _, _ = partition_data(
            data, 
            2,  # num_clients 
            1.0,  # beta
            'cpu',  # device
            hop=1, 
            use_feature_prop=True, 
            config=test_config
        )
        print('Partition successful')
        
        # Check if log file was created
        expected_log_dir = os.path.join(test_config["results_dir"], "propagation_stats")
        
        if os.path.exists(expected_log_dir):
            log_files = os.listdir(expected_log_dir)
            if log_files:
                print(f"Log files created successfully in {expected_log_dir}:")
                for file in log_files:
                    print(f" - {file}")
                
                # Read the first log file to check PE parameters
                import json
                with open(os.path.join(expected_log_dir, log_files[0]), 'r') as f:
                    log_data = json.load(f)
                    
                print("\nLog file parameters:")
                print(f"PE enabled: {log_data.get('use_pe')}")
                print(f"PE dimension (r): {log_data.get('pe_r')}")
                print(f"PE propagation steps (P): {log_data.get('pe_P')}")
                print(f"PE normalization: {log_data.get('normalize')}")
                
                if log_data.get('pe_r') == test_config['pe_r'] and log_data.get('pe_P') == test_config['pe_P']:
                    print("\n✅ TEST PASSED: Configuration correctly passed through to partitioning")
                else:
                    print("\n❌ TEST FAILED: Configuration values did not match")
            else:
                print(f"No log files found in {expected_log_dir}")
        else:
            print(f"Expected log directory {expected_log_dir} not found")
            
    except Exception as e:
        print(f'Partition failed with error: {e}')
        raise

if __name__ == "__main__":
    test_config_passing() 