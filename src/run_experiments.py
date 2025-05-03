#!/usr/bin/env python3

from run import load_configuration, main_experiment
import os
import ray
import argparse
import json
from tabulate import tabulate
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run federated GNN experiments and print results')
    parser.add_argument('--clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--beta', type=float, default=1, help='Beta parameter for Dirichlet distribution')
    parser.add_argument('--datasets', nargs='+', default=["Cora"], 
                        help='List of datasets to run experiments on')
    parser.add_argument('--data_loading', nargs='+', default=["full", "adjacency", "zero_hop"], 
                        help='Data loading options')
    parser.add_argument('--models', nargs='+', default=["GCN"], 
                        help='Model types to use')
    parser.add_argument('--save_results', action='store_true', help='Save detailed results to files')
    parser.add_argument('--results_dir', type=str, default="results/Planetoid_test_results", 
                        help='Directory to save results')
    return parser.parse_args()

def run_experiments(args):
    # Ensure Ray is shut down before starting
    try:
        ray.shutdown()
    except:
        pass
    
    # Load configuration
    _, _, cfg = load_configuration()
    
    # Create main results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Store all experiment results
    all_results = []
    
    # Create a summary table for the final output
    summary_rows = []
    
    # Run experiments for each combination
    for dataset_name in args.datasets:
        for data_loading_option in args.data_loading:
            for model_type in args.models:
                experiment_name = f"{dataset_name}_{data_loading_option}_{model_type}"
                print(f"\n{'='*80}")
                print(f"Running experiment: {experiment_name}")
                print(f"{'='*80}")
                
                # Run the experiment
                result, output = main_experiment(
                    args.clients, 
                    args.beta, 
                    data_loading_option, 
                    model_type, 
                    cfg, 
                    dataset_name=dataset_name, 
                    hop=1
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
                if args.save_results:
                    # Create directory for this experiment
                    exp_dir = os.path.join(args.results_dir, experiment_name)
                    os.makedirs(exp_dir, exist_ok=True)
                    
                    # Save detailed results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"results_{experiment_name}_{timestamp}.json"
                    filepath = os.path.join(exp_dir, filename)
                    
                    with open(filepath, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    # Also save readable output
                    txt_filepath = os.path.join(exp_dir, f"results_{experiment_name}_{timestamp}.txt")
                    with open(txt_filepath, 'w') as f:
                        f.write(output)
                    
                    print(f"Results saved to {filepath}")
                
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

if __name__ == "__main__":
    args = parse_arguments()
    summary_rows, all_results = run_experiments(args)
    print_summary(summary_rows) 