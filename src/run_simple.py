#!/usr/bin/env python3

from run import load_configuration, main_experiment
import os
import ray
import argparse
import json
import yaml
from tabulate import tabulate
from datetime import datetime
import time
import shutil
from utils import load_config

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run a simple federated GNN experiment from a YAML config')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--save_results', action='store_true', help='Save detailed results to files')
    return parser.parse_args()

def load_yaml_config(config_path):
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_environment_for_experiment(dataset_name, data_loading_option, model_type, results_dir, timestamp):
    """Setup environment variables to redirect CSV output to experiment directory"""
    # Create experiment directory path
    experiment_name = f"{dataset_name}_{data_loading_option}_{model_type}"
    exp_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Set environment variables that will be checked in run_utils.py
    os.environ["EXPERIMENT_RESULTS_DIR"] = exp_dir
    os.environ["EXPERIMENT_TIMESTAMP"] = timestamp
    
    return exp_dir, experiment_name

def copy_training_csv_to_experiment_dir(exp_dir, experiment_name, timestamp):
    """Copy training CSV file to experiment directory and rename it"""
    source_file = "results.csv"
    if os.path.exists(source_file):
        target_file = os.path.join(exp_dir, f"training_{experiment_name}_{timestamp}.csv")
        shutil.copy2(source_file, target_file)
        print(f"Training CSV results saved to {target_file}")

def format_time(seconds):
    """Format time in seconds to a human-readable string"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def run_simple_experiment(args):
    # Load configuration from YAML file
    if not args.config:
        print("Error: A configuration file must be specified with --config")
        return None, None
    
    yaml_config = load_yaml_config(args.config)
    
    # Set single values for experiment
    clients_num = yaml_config.get("num_clients", 10)
    # Handle case when num_clients is a list (take first value)
    if isinstance(clients_num, list) and len(clients_num) > 0:
        clients_num = clients_num[0]
        
    beta = yaml_config.get("beta", 1.0)
    # Handle case when beta is a list (take first value)
    if isinstance(beta, list) and len(beta) > 0:
        beta = beta[0]
        
    dataset_name = yaml_config.get("dataset", "Cora")
    # Handle case when dataset is a list (take first value)
    if isinstance(dataset_name, list) and len(dataset_name) > 0:
        dataset_name = dataset_name[0]
        
    data_loading_option = yaml_config.get("data_loading", "full")
    # Handle case when data_loading is a list (take first value)
    if isinstance(data_loading_option, list) and len(data_loading_option) > 0:
        data_loading_option = data_loading_option[0]
        
    model_type = yaml_config.get("model", "GCN")
    # Handle case when model is a list (take first value)
    if isinstance(model_type, list) and len(model_type) > 0:
        model_type = model_type[0]
        
    results_dir = yaml_config.get("results_dir", "results/simple_experiment")
    save_results = yaml_config.get("save_results", False) or args.save_results
    hop = yaml_config.get("hop", 1)
    fulltraining_flag = yaml_config.get("fulltraining_flag", False)
    num_rounds = yaml_config.get("num_rounds", 10)
    epochs = yaml_config.get("epochs", 3)
    lr = yaml_config.get("lr", 0.5)
    
    # Create a training configuration for the experiment
 # Use the entire YAML config instead of just a subset
    training_cfg = yaml_config.copy()  # Copy the complete config

    # Ensure critical parameters are present with defaults
    training_cfg.setdefault("num_rounds", num_rounds)
    training_cfg.setdefault("epochs", epochs)
    training_cfg.setdefault("lr", lr)
    training_cfg.setdefault("beta", beta)
    training_cfg.setdefault("fulltraining_flag", fulltraining_flag)
    training_cfg.setdefault("hop", hop)
    training_cfg.setdefault("results_dir", results_dir)
    # Ensure Ray is shut down before starting
    try:
        ray.shutdown()
    except:
        pass
    
    # Create main results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print experiment configuration
    print("\nExperiment Configuration:")
    print(f"- Clients: {clients_num}")
    print(f"- Rounds: {num_rounds}")
    print(f"- Epochs: {epochs}")
    print(f"- Learning Rate: {lr}")
    print(f"- Beta: {beta}")
    print(f"- Dataset: {dataset_name}")
    print(f"- Data Loading Option: {data_loading_option}")
    print(f"- Model Type: {model_type}")
    print(f"- Results Directory: {results_dir}")
    print(f"- Save Detailed Results: {save_results}")
    print(f"- Hop: {hop}")
    print(f"- Full Training Flag: {fulltraining_flag}")
    
    # Setup environment for this experiment
    exp_dir, experiment_name = setup_environment_for_experiment(
        dataset_name, data_loading_option, model_type, results_dir, timestamp
    )
    
    # Create a monkey patch for save_results_to_csv in run_utils
    from run_utils import save_results_to_csv as original_save_func
    
    def patched_save_func(results, filename=None):
        # Use the original function but with our custom filename
        csv_filename = os.path.join(exp_dir, f"training_{experiment_name}_{timestamp}.csv")
        return original_save_func(results, csv_filename)
    
    # Apply the monkey patch to the imported function
    import run_utils
    run_utils.save_results_to_csv = patched_save_func
    
    # Print experiment header
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*80}")
    
    # Start time measurement
    start_time = time.time()
    
    # Run the experiment
    result, output = main_experiment(
        clients_num, 
        beta, 
        data_loading_option, 
        model_type, 
        training_cfg, 
        dataset_name=dataset_name, 
        hop=hop,
        fulltraining_flag=fulltraining_flag
    )
    
    # Calculate experiment duration
    end_time = time.time()
    duration = end_time - start_time
    duration_formatted = format_time(duration)
    
    # Add duration to the result
    result["duration"] = {
        "seconds": duration,
        "formatted": duration_formatted
    }
    
    # Extract key metrics
    avg_global = result["summary"]["average_global_result"]
    avg_client = result["summary"]["average_client_result"]
    
    # Save results if requested
    if save_results:
        # Save detailed results
        filename = f"results_{experiment_name}_{timestamp}.json"
        filepath = os.path.join(exp_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Also save readable output with added duration information
        txt_filepath = os.path.join(exp_dir, f"results_{experiment_name}_{timestamp}.txt")
        with open(txt_filepath, 'w') as f:
            f.write(output)
            f.write(f"\n\nExperiment Duration: {duration_formatted} (HH:MM:SS)\n")
        
        print(f"Results saved to {filepath}")
    
    # Make sure the training CSV was copied to experiment directory
    copy_training_csv_to_experiment_dir(exp_dir, experiment_name, timestamp)
    
    # Print key results
    print("\n")
    print("=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Data Loading: {data_loading_option}")
    print(f"Model: {model_type}")
    print(f"Beta: {beta}")
    print(f"Clients: {clients_num}")
    print("-" * 40)
    print(f"Average Global Result: {avg_global:.4f}")
    print(f"Average Client Result: {avg_client:.4f}")
    print(f"Experiment Duration: {duration_formatted}")
    print("=" * 80)
    
    return result, output

def create_example_config(output_path="simple_config_example.yaml"):
    """Create an example YAML configuration file"""
    example_config = {
        "num_clients": 10,
        "num_rounds": 10,
        "epochs": 3,
        "beta": 1.0,
        "lr": 0.5,
        "dataset": "Cora",
        "data_loading": "full",
        "model": "GCN",
        "results_dir": "results/simple_experiment",
        "save_results": True,
        "hop": 1,
        "fulltraining_flag": False
    }
    
    with open(output_path, 'w') as file:
        yaml.dump(example_config, file, default_flow_style=False)
    
    print(f"Example configuration file created at: {output_path}")

if __name__ == "__main__":
    args = parse_arguments()
    
    # Generate example config file if requested
    if args.config == "generate_example":
        create_example_config()
        exit(0)
    
    run_simple_experiment(args) 