#!/usr/bin/env python
"""
Script for running multiple federated learning experiments with the enhanced framework
"""
import os
import yaml
import json
import argparse
import logging
import subprocess
import multiprocessing
from itertools import product
from datetime import datetime
import time
import pandas as pd
import ray

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_config(config_path):
    """Read configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_results_dir(base_dir="results"):
    """Create a timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"experiments_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def generate_experiment_commands(config, results_dir, script_path="FP/src/run_enhanced.py"):
    """Generate commands for each experiment configuration"""
    commands = []
    experiment_configs = []
    
    # Extract basic parameters
    datasets = config.get("datasets", ["Cora"])
    data_loading_options = config.get("data_loading_options", ["zero_hop"])
    num_clients_options = config.get("num_clients_options", [3])
    beta_options = config.get("beta_options", [0.5])
    rounds_options = config.get("rounds_options", [20])
    
    # Optional parameters
    use_cuda = config.get("use_cuda", False)
    memory_efficient = config.get("memory_efficient", True)
    batch_size_options = config.get("batch_size_options", {"default": 1024, "ogbn-products": 512})
    epochs_options = config.get("epochs_options", {"default": 1, "ogbn-products": 2})
    repeats = config.get("repeats", 1)
    
    # For each combination of parameters
    for dataset, data_loading, num_clients, beta, rounds in product(
        datasets, data_loading_options, num_clients_options, beta_options, rounds_options
    ):
        # Determine batch size based on dataset
        if dataset.lower() in ["ogbn-products", "products"]:
            batch_size = batch_size_options.get("ogbn-products", 512)
        else:
            batch_size = batch_size_options.get("default", 1024)
        
        # Configure epochs for this configuration
        config_file = write_experiment_config(
            dataset, epochs_options, results_dir
        )
        
        # Generate the base command
        base_cmd = [
            "python", script_path,
            "--dataset", dataset,
            "--data_loading", data_loading,
            "--num_clients", str(num_clients),
            "--beta", str(beta),
            "--rounds", str(rounds),
            "--batch_size", str(batch_size),
            "--results_dir", results_dir,
            "--config", config_file
        ]
        
        # Add optional flags
        if use_cuda:
            base_cmd.append("--cuda")
        if memory_efficient:
            base_cmd.append("--memory_efficient")
        
        # Run the experiment multiple times if specified
        for i in range(repeats):
            experiment_name = f"{dataset}_{data_loading}_c{num_clients}_b{beta}_r{rounds}_run{i}"
            cmd = base_cmd.copy()
            
            # Save the command for logging
            commands.append({
                "name": experiment_name,
                "command": " ".join(cmd)
            })
            
            # Add to experiment configs
            experiment_configs.append({
                "name": experiment_name,
                "command": cmd,
                "dataset": dataset,
                "data_loading": data_loading,
                "num_clients": num_clients,
                "beta": beta,
                "rounds": rounds,
                "batch_size": batch_size,
                "memory_efficient": memory_efficient,
                "run": i
            })
    
    # Save the experiment configurations
    with open(os.path.join(results_dir, "experiment_commands.json"), 'w') as f:
        json.dump(commands, f, indent=2)
    
    return experiment_configs

def write_experiment_config(dataset, epochs_options, results_dir):
    """Write a custom configuration file for a specific experiment"""
    config = {
        "epochs": 1,  # Default
        "global_step": 20  # Default
    }
    
    # Set epochs based on dataset
    if dataset.lower() in ["ogbn-products", "products"]:
        config["epochs"] = epochs_options.get("ogbn-products", 1)
    elif dataset.lower() in ["ogbn-arxiv", "arxiv"]:
        config["epochs"] = epochs_options.get("ogbn-arxiv", 1)
    else:
        config["epochs"] = epochs_options.get("default", 1)
    
    # Create directory if needed
    config_dir = os.path.join(results_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    # Generate a config file name
    config_file = os.path.join(config_dir, f"{dataset.lower()}_config.yaml")
    
    # Write the config to a file
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file

def run_experiment(experiment_config):
    """Run a single experiment with the given configuration"""
    try:
        logger.info(f"Starting experiment: {experiment_config['name']}")
        start_time = time.time()
        
        result = subprocess.run(
            experiment_config["command"],
            check=True,
            capture_output=True,
            text=True
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"Experiment completed: {experiment_config['name']} (Time: {execution_time:.2f}s)")
        
        # Save logs
        log_dir = os.path.join(os.path.dirname(experiment_config["command"][-1]), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        with open(os.path.join(log_dir, f"{experiment_config['name']}_stdout.log"), 'w') as f:
            f.write(result.stdout)
        
        with open(os.path.join(log_dir, f"{experiment_config['name']}_stderr.log"), 'w') as f:
            f.write(result.stderr)
        
        return {
            "name": experiment_config["name"],
            "success": True,
            "execution_time": execution_time
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment failed: {experiment_config['name']}")
        logger.error(f"Error: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        
        # Save error logs
        log_dir = os.path.join(os.path.dirname(experiment_config["command"][-1]), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        with open(os.path.join(log_dir, f"{experiment_config['name']}_error.log"), 'w') as f:
            f.write(f"Command: {' '.join(experiment_config['command'])}\n\n")
            f.write(f"Exit code: {e.returncode}\n\n")
            f.write(f"Stdout:\n{e.stdout}\n\n")
            f.write(f"Stderr:\n{e.stderr}\n\n")
        
        return {
            "name": experiment_config["name"],
            "success": False,
            "error": str(e),
            "exit_code": e.returncode
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in experiment: {experiment_config['name']}")
        logger.error(f"Error: {str(e)}")
        
        return {
            "name": experiment_config["name"],
            "success": False,
            "error": str(e)
        }

def run_experiments_parallel(experiment_configs, max_workers=None):
    """Run multiple experiments in parallel"""
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count() - 1, len(experiment_configs))
        max_workers = max(1, max_workers)  # At least 1 worker
    
    logger.info(f"Running {len(experiment_configs)} experiments with {max_workers} workers")
    
    results = []
    with multiprocessing.Pool(max_workers) as pool:
        for result in pool.imap_unordered(run_experiment, experiment_configs):
            results.append(result)
            logger.info(f"Progress: {len(results)}/{len(experiment_configs)} experiments completed")
    
    return results

def run_experiments_sequential(experiment_configs):
    """Run experiments one after another"""
    logger.info(f"Running {len(experiment_configs)} experiments sequentially")
    
    results = []
    for i, config in enumerate(experiment_configs):
        logger.info(f"Experiment {i+1}/{len(experiment_configs)}")
        result = run_experiment(config)
        results.append(result)
    
    return results

def collect_results(results_dir):
    """Collect and combine results from multiple experiments"""
    # Find all result files
    all_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json') and not file.startswith('experiment_commands'):
                all_files.append(os.path.join(root, file))
    
    combined_data = []
    for file_path in all_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract basic info
            params = data.get('params', {})
            
            # Extract experiment summary
            for result in data.get('experiment_summary', []):
                result_with_params = {
                    **params,
                    **result
                }
                combined_data.append(result_with_params)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
    
    # Convert to DataFrame and save
    if combined_data:
        df = pd.DataFrame(combined_data)
        csv_path = os.path.join(results_dir, "combined_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Combined results saved to {csv_path}")
        
        # Also save as Excel if available
        try:
            excel_path = os.path.join(results_dir, "combined_results.xlsx")
            df.to_excel(excel_path, index=False)
            logger.info(f"Results also saved to Excel: {excel_path}")
        except Exception:
            logger.warning("Could not save Excel file. Excel export requires openpyxl package.")
    
    return combined_data

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run multiple federated learning experiments')
    parser.add_argument('--config', type=str, required=True, 
                      help='Path to experiment configuration file')
    parser.add_argument('--parallel', action='store_true', default=False,
                      help='Run experiments in parallel (default: sequential)')
    parser.add_argument('--max_workers', type=int, default=None,
                      help='Maximum number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--results_dir', type=str, default=None,
                      help='Directory to store results (default: auto-generated)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Read configuration
    logger.info(f"Reading configuration from {args.config}")
    config = read_config(args.config)
    
    # Setup results directory
    results_dir = args.results_dir or setup_results_dir()
    logger.info(f"Results will be saved to {results_dir}")
    
    # Generate experiment commands
    experiment_configs = generate_experiment_commands(config, results_dir)
    logger.info(f"Generated {len(experiment_configs)} experiment configurations")
    
    # Run experiments
    if args.parallel:
        results = run_experiments_parallel(experiment_configs, args.max_workers)
    else:
        results = run_experiments_sequential(experiment_configs)
    
    # Save execution results
    with open(os.path.join(results_dir, "execution_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Collect and combine results
    logger.info("Collecting and combining results")
    combined_data = collect_results(results_dir)
    logger.info(f"Processed {len(combined_data)} result entries")
    
    logger.info("All experiments completed")
    
    # Count successes and failures
    successes = sum(1 for r in results if r.get('success', False))
    failures = len(results) - successes
    logger.info(f"Summary: {successes} successful, {failures} failed experiments")

if __name__ == "__main__":
    main() 