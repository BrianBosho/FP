#!/usr/bin/env python3

from run import load_configuration, main_experiment
import os
import ray
import argparse
import json
import yaml
from tabulate import tabulate
from datetime import datetime
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run federated GNN experiments and print results')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    
    # Keep command-line arguments as fallback options
    parser.add_argument('--clients', type=int, default=10, help='Number of clients (if no config file)')
    parser.add_argument('--beta', type=float, default=1, help='Beta parameter for Dirichlet distribution (if no config file)')
    parser.add_argument('--datasets', nargs='+', default=["Cora"], 
                        help='List of datasets to run experiments on (if no config file)')
    parser.add_argument('--data_loading', nargs='+', default=["full", "adjacency", "zero_hop"], 
                        help='Data loading options (if no config file)')
    parser.add_argument('--models', nargs='+', default=["GCN"], 
                        help='Model types to use (if no config file)')
    parser.add_argument('--save_results', action='store_true', help='Save detailed results to files')
    parser.add_argument('--results_dir', type=str, default="results/Planetoid_test_results", 
                        help='Directory to save results (if no config file)')
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

def run_experiments(args):
    # Load configuration from YAML if provided
    if args.config:
        yaml_config = load_yaml_config(args.config)
        clients_num = yaml_config.get('clients', args.clients)
        beta = yaml_config.get('beta', args.beta)
        datasets = yaml_config.get('datasets', args.datasets)
        data_loading_options = yaml_config.get('data_loading', args.data_loading)
        model_types = yaml_config.get('models', args.models)
        results_dir = yaml_config.get('results_dir', args.results_dir)
        save_results = yaml_config.get('save_results', args.save_results)
        hop = yaml_config.get('hop', 1)
    else:
        clients_num = args.clients
        beta = args.beta
        datasets = args.datasets
        data_loading_options = args.data_loading
        model_types = args.models
        results_dir = args.results_dir
        save_results = args.save_results
        hop = 1
    
    # Ensure Ray is shut down before starting
    try:
        ray.shutdown()
    except:
        pass
    
    # Load configuration
    _, _, cfg = load_configuration()
    
    # Create main results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Store all experiment results
    all_results = []
    
    # Create a summary table for the final output
    summary_rows = []
    
    # Print experiment configuration
    print("\nExperiment Configuration:")
    print(f"- Clients: {clients_num}")
    print(f"- Beta: {beta}")
    print(f"- Datasets: {datasets}")
    print(f"- Data Loading Options: {data_loading_options}")
    print(f"- Model Types: {model_types}")
    print(f"- Results Directory: {results_dir}")
    print(f"- Save Detailed Results: {save_results}")
    print(f"- Hop: {hop}")
    
    # Run experiments for each combination
    for dataset_name in datasets:
        for data_loading_option in data_loading_options:
            for model_type in model_types:
                # Generate timestamp for this experiment
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
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
                
                # Run the experiment
                result, output = main_experiment(
                    clients_num, 
                    beta, 
                    data_loading_option, 
                    model_type, 
                    cfg, 
                    dataset_name=dataset_name, 
                    hop=hop
                )
                
                # Extract key metrics
                avg_global = result["summary"]["average_global_result"]
                avg_client = result["summary"]["average_client_result"]
                
                # Add to summary table
                summary_rows.append([
                    dataset_name, 
                    data_loading_option, 
                    model_type, 
                    f"{avg_global:.4f}",
                    f"{avg_client:.4f}"
                ])
                
                # Save results if requested
                if save_results:
                    # Save detailed results
                    filename = f"results_{experiment_name}_{timestamp}.json"
                    filepath = os.path.join(exp_dir, filename)
                    
                    with open(filepath, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    # Also save readable output
                    txt_filepath = os.path.join(exp_dir, f"results_{experiment_name}_{timestamp}.txt")
                    with open(txt_filepath, 'w') as f:
                        f.write(output)
                    
                    print(f"Results saved to {filepath}")
                
                # Make sure the training CSV was copied to experiment directory
                # This is a fallback in case our monkey patching didn't work
                copy_training_csv_to_experiment_dir(exp_dir, experiment_name, timestamp)
                
                # Store results for final summary
                all_results.append({
                    "dataset": dataset_name,
                    "data_loading": data_loading_option,
                    "model": model_type,
                    "avg_global": avg_global,
                    "avg_client": avg_client,
                    "details": result
                })
    
    return summary_rows, all_results

def print_summary(summary_rows):
    # Print summary table
    print("\n\n")
    print("=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    
    headers = ["Dataset", "Data Loading", "Model", "Avg Global Result", "Avg Client Result"]
    print(tabulate(summary_rows, headers=headers, tablefmt="grid"))
    
    print("\n")
    
    # Print the key results separately in a clear, easy-to-read format
    print("KEY RESULTS:")
    print("-" * 40)
    for row in summary_rows:
        print(f"{row[0]} with {row[1]} using {row[2]}:")
        print(f"  - Average Global Result: {row[3]}")
        print(f"  - Average Client Result: {row[4]}")
        print("-" * 40)

def create_example_config(output_path="experiment_config_example.yaml"):
    """Create an example YAML configuration file"""
    example_config = {
        "clients": 10,
        "beta": 1,
        "datasets": ["Cora", "Citeseer"],
        "data_loading": ["full", "adjacency", "zero_hop"],
        "models": ["GCN"],
        "results_dir": "results/yaml_experiment_results",
        "save_results": True,
        "hop": 1
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
    
    summary_rows, all_results = run_experiments(args)
    print_summary(summary_rows) 