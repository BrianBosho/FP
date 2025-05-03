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
from utils import load_config

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run federated GNN experiments and print results')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    
    # Keep command-line arguments as fallback options
    parser.add_argument('--clients', type=int, help='Number of clients')
    parser.add_argument('--rounds', type=int, help='Number of communication rounds')
    parser.add_argument('--epochs', type=int, help='Number of local epochs per round')
    parser.add_argument('--beta', type=float, help='Beta parameter for Dirichlet distribution')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--datasets', nargs='+', help='List of datasets to run experiments on')
    parser.add_argument('--data_loading', nargs='+', help='Data loading options')
    parser.add_argument('--models', nargs='+', help='Model types to use')
    parser.add_argument('--hop', type=int, help='Number of hops for graph propagation')
    parser.add_argument('--save_results', action='store_true', help='Save detailed results to files')
    parser.add_argument('--results_dir', type=str, help='Directory to save results')
    parser.add_argument('--fulltraining_flag', action='store_true', help='Use full training flag')
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

def save_summary_results(summary_rows, all_results, results_dir, config):
    """Save summary results to a file in the parent results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create summary TXT file
    summary_txt_path = os.path.join(results_dir, f"summary_results_{timestamp}.txt")
    
    with open(summary_txt_path, 'w') as f:
        # Write experiment configuration
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"- Clients: {config['num_clients']}\n")
        f.write(f"- Rounds: {config['num_rounds']}\n")
        f.write(f"- Epochs: {config['epochs']}\n")
        f.write(f"- Learning Rate: {config['lr']}\n")
        f.write(f"- Beta: {config['beta']}\n")
        f.write(f"- Datasets: {config['datasets']}\n")
        f.write(f"- Data Loading Options: {config['data_loading']}\n")
        f.write(f"- Model Types: {config['models']}\n")
        f.write(f"- Results Directory: {config['results_dir']}\n")
        f.write(f"- Save Detailed Results: {config['save_results']}\n")
        f.write(f"- Hop: {config['hop']}\n")
        f.write(f"- Full Training Flag: {config['fulltraining_flag']}\n\n")
        
        # Write summary table
        f.write("\n" + "=" * 80 + "\n")
        f.write("EXPERIMENT RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        headers = ["Dataset", "Data Loading", "Model", "Avg Global Result", "Avg Client Result"]
        f.write(tabulate(summary_rows, headers=headers, tablefmt="grid") + "\n\n")
        
        # Write key results
        f.write("KEY RESULTS:\n")
        f.write("-" * 40 + "\n")
        for row in summary_rows:
            f.write(f"{row[0]} with {row[1]} using {row[2]}:\n")
            f.write(f"  - Average Global Result: {row[3]}\n")
            f.write(f"  - Average Client Result: {row[4]}\n")
            f.write("-" * 40 + "\n")
    
    # Create summary JSON file
    summary_json_path = os.path.join(results_dir, f"summary_results_{timestamp}.json")
    
    # Create a structured JSON with all results
    summary_json = {
        "timestamp": timestamp,
        "configuration": config,
        "results": [
            {
                "dataset": result["dataset"],
                "data_loading": result["data_loading"],
                "model": result["model"],
                "avg_global": result["avg_global"],
                "avg_client": result["avg_client"]
            }
            for result in all_results
        ]
    }
    
    with open(summary_json_path, 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"\nSummary results saved to:")
    print(f"- Text file: {summary_txt_path}")
    print(f"- JSON file: {summary_json_path}")
    
    return summary_txt_path, summary_json_path

def run_experiments(args):
    # Load base configuration from base.yaml
    base_config_path = "conf/base.yaml"
    base_cfg = load_config(base_config_path)
    
    # Set default values from base.yaml
    cfg = {
        "num_clients": base_cfg.get("num_clients", 10),
        "num_rounds": base_cfg.get("num_rounds", 10),
        "epochs": base_cfg.get("epochs", 3),
        "beta": base_cfg.get("beta", 1),
        "lr": base_cfg.get("lr", 0.5),
        "fulltraining_flag": base_cfg.get("fulltraining_flag", False),
        "datasets": ["Cora"],
        "data_loading": ["full", "adjacency", "zero_hop"],
        "models": ["GCN"],
        "results_dir": "results/Planetoid_test_results",
        "save_results": False,
        "hop": 1
    }
    
    # Update with YAML config if provided
    if args.config:
        yaml_config = load_yaml_config(args.config)
        # Update config with values from YAML
        for key, value in yaml_config.items():
            cfg[key] = value
    
    # Override with command-line arguments if provided
    if args.clients is not None:
        cfg["num_clients"] = args.clients
    if args.rounds is not None:
        cfg["num_rounds"] = args.rounds
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.beta is not None:
        cfg["beta"] = args.beta
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.datasets is not None:
        cfg["datasets"] = args.datasets
    if args.data_loading is not None:
        cfg["data_loading"] = args.data_loading
    if args.models is not None:
        cfg["models"] = args.models
    if args.hop is not None:
        cfg["hop"] = args.hop
    if args.save_results:
        cfg["save_results"] = args.save_results
    if args.results_dir is not None:
        cfg["results_dir"] = args.results_dir
    if args.fulltraining_flag:
        cfg["fulltraining_flag"] = args.fulltraining_flag
    
    # Extract values from the merged configuration
    clients_num = cfg["num_clients"]
    beta = cfg["beta"]
    datasets = cfg["datasets"]
    data_loading_options = cfg["data_loading"]
    model_types = cfg["models"]
    results_dir = cfg["results_dir"]
    save_results = cfg["save_results"]
    hop = cfg["hop"]
    fulltraining_flag = cfg["fulltraining_flag"]
    
    # Create a training configuration for the experiment
    training_cfg = {
        "num_rounds": cfg["num_rounds"],
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "beta": beta,
        "fulltraining_flag": fulltraining_flag
    }
    
    # Ensure Ray is shut down before starting
    try:
        ray.shutdown()
    except:
        pass
    
    # Create main results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Store all experiment results
    all_results = []
    
    # Create a summary table for the final output
    summary_rows = []
    
    # Print experiment configuration
    print("\nExperiment Configuration:")
    print(f"- Clients: {clients_num}")
    print(f"- Rounds: {cfg['num_rounds']}")
    print(f"- Epochs: {cfg['epochs']}")
    print(f"- Learning Rate: {cfg['lr']}")
    print(f"- Beta: {beta}")
    print(f"- Datasets: {datasets}")
    print(f"- Data Loading Options: {data_loading_options}")
    print(f"- Model Types: {model_types}")
    print(f"- Results Directory: {results_dir}")
    print(f"- Save Detailed Results: {save_results}")
    print(f"- Hop: {hop}")
    print(f"- Full Training Flag: {fulltraining_flag}")
    
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
                    # Save as JSON too
                    json_filename = os.path.join(exp_dir, f"training_{experiment_name}_{timestamp}.json")
                    with open(json_filename, 'w') as f:
                        # Process the data to ensure it's JSON serializable
                        processed_results = []
                        for result in results:
                            processed_result = {}
                            for key, value in result.items():
                                # Convert any numpy arrays to lists
                                if hasattr(value, 'tolist'):
                                    processed_result[key] = value.tolist()
                                else:
                                    processed_result[key] = value
                            processed_results.append(processed_result)
                        
                        json.dump(processed_results, f, indent=2)
                    print(f"Training JSON results saved to {json_filename}")
                    
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
                    training_cfg, 
                    dataset_name=dataset_name, 
                    hop=hop,
                    fulltraining_flag=fulltraining_flag
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
    
    # Save summary results to the parent directory
    save_summary_results(summary_rows, all_results, results_dir, cfg)
    
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
        "num_clients": 10,
        "num_rounds": 10,
        "epochs": 3,
        "beta": 1,
        "lr": 0.5,
        "datasets": ["Cora", "Citeseer"],
        "data_loading": ["full", "adjacency", "zero_hop"],
        "models": ["GCN"],
        "results_dir": "results/yaml_experiment_results",
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
    
    summary_rows, all_results = run_experiments(args)
    print_summary(summary_rows)