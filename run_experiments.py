#!/usr/bin/env python

import os
import sys
import argparse
import torch
import ray
import numpy as np
from datetime import datetime
import logging
import json
import pandas as pd

# Import functions from run.py
from run import (
    load_configuration, 
    main_experiment,
    load_and_split_with_khop,
    load_and_split_with_feature_prop
)

def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run federated learning experiments")
    
    # Required arguments
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["single", "all"],
        default="all",
        help="Run a single experiment or all combinations"
    )
    
    # Arguments for single experiment mode
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="Cora",
        help="Dataset name (default: Cora)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["GCN", "GAT"],
        default="GCN",
        help="Model type (default: GCN)"
    )
    parser.add_argument(
        "--data_loading", 
        type=str,
        choices=["split_dataset", "split_dataset_with_khop", "split_dataset_with_feature_prop"],
        default="split_dataset",
        help="Data loading method (default: split_dataset)"
    )
    parser.add_argument(
        "--num_clients", 
        type=int, 
        default=None,
        help="Number of clients (default: from config)"
    )
    parser.add_argument(
        "--hop", 
        type=int, 
        default=1,
        help="Number of hops for k-hop methods (default: 1)"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="conf/base.yaml",
        help="Path to configuration file (default: conf/base.yaml)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--repetitions", 
        type=int, 
        default=5,
        help="Number of repetitions for each experiment (default: 5)"
    )
    
    return parser.parse_args()

def run_single_experiment(
    dataset_name,
    model_type,
    data_loading_option,
    num_clients,
    beta,
    cfg,
    hop,
    output_dir,
    logger
):
    """Run a single experiment with the specified parameters"""
    logger.info(f"Running experiment with the following parameters:")
    logger.info(f"  - Dataset: {dataset_name}")
    logger.info(f"  - Model: {model_type}")
    logger.info(f"  - Data loading: {data_loading_option}")
    logger.info(f"  - Number of clients: {num_clients}")
    logger.info(f"  - Hop: {hop}")
    
    # Create a structured directory for the experiment
    result_name = f"{dataset_name}_{data_loading_option}_{model_type}"
    results_dir = os.path.join(output_dir, result_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Run the experiment
    results_data, result_text = main_experiment(
        clients_num=num_clients,
        beta=beta,
        data_loading_option=data_loading_option,
        model_type=model_type,
        cfg=cfg,
        dataset_name=dataset_name,
        hop=hop
    )
    
    # Create a unique timestamp for the files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the text output for backward compatibility
    text_filename = f'results_{dataset_name}_{data_loading_option}_{model_type}_{timestamp}.txt'
    text_filepath = os.path.join(results_dir, text_filename)
    with open(text_filepath, 'w') as f:
        f.write(result_text)
    
    # Save the results as JSON
    json_filename = f'results_{dataset_name}_{data_loading_option}_{model_type}_{timestamp}.json'
    json_filepath = os.path.join(results_dir, json_filename)
    with open(json_filepath, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    # Save the results as CSV
    # For rounds data
    rounds_df = pd.DataFrame(results_data["rounds"])
    rounds_csv_filename = f'rounds_{dataset_name}_{data_loading_option}_{model_type}_{timestamp}.csv'
    rounds_csv_filepath = os.path.join(results_dir, rounds_csv_filename)
    rounds_df.to_csv(rounds_csv_filepath, index=False)
    
    # For summary data
    summary_data = results_data["summary"]
    # Flatten the lists in summary for CSV format
    flat_summary = {
        "average_global_result": summary_data["average_global_result"],
        "average_client_result": summary_data["average_client_result"],
        "std_global": summary_data["std_global"],
        "std_client": summary_data["std_client"]
    }
    # Add experiment config to summary
    for key, value in results_data["experiment_config"].items():
        flat_summary[key] = value
    
    summary_df = pd.DataFrame([flat_summary])
    summary_csv_filename = f'summary_{dataset_name}_{data_loading_option}_{model_type}_{timestamp}.csv'
    summary_csv_filepath = os.path.join(results_dir, summary_csv_filename)
    summary_df.to_csv(summary_csv_filepath, index=False)
    
    # Create a separate CSV for global and client results arrays
    results_arrays = {
        "round": list(range(1, len(summary_data["global_results"]) + 1)),
        "global_results": summary_data["global_results"],
        "client_results": summary_data["client_results"]
    }
    results_arrays_df = pd.DataFrame(results_arrays)
    arrays_csv_filename = f'results_arrays_{dataset_name}_{data_loading_option}_{model_type}_{timestamp}.csv'
    arrays_csv_filepath = os.path.join(results_dir, arrays_csv_filename)
    results_arrays_df.to_csv(arrays_csv_filepath, index=False)
    
    logger.info(f"Results saved to:")
    logger.info(f"  - Text: {text_filepath}")
    logger.info(f"  - JSON: {json_filepath}")
    logger.info(f"  - CSV (rounds): {rounds_csv_filepath}")
    logger.info(f"  - CSV (summary): {summary_csv_filepath}")
    logger.info(f"  - CSV (results arrays): {arrays_csv_filepath}")
    
    return {
        "text_file": text_filepath,
        "json_file": json_filepath,
        "rounds_csv": rounds_csv_filepath,
        "summary_csv": summary_csv_filepath,
        "arrays_csv": arrays_csv_filepath
    }

def run_all_experiments(args, logger):
    """Run experiments for all combinations of parameters"""
    # Define lists of options
    datasets = ["Cora", "Citeseer", "Pubmed"]  # Add more datasets if available
    data_loading_options = [
        "split_dataset", 
        "split_dataset_with_khop", 
        "split_dataset_with_feature_prop"
    ]
    model_types = ["GCN", "GAT"]
    
    # Load configuration
    clients_num, beta, cfg = load_configuration(args.config_path)
    
    # Override number of clients if specified
    if args.num_clients:
        clients_num = args.num_clients
    
    # Create main output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(args.output_dir, f"results_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Track all results
    results_summary = []
    
    # Loop over all combinations
    for dataset_name in datasets:
        for data_loading_option in data_loading_options:
            for model_type in model_types:
                try:
                    filepath = run_single_experiment(
                        dataset_name=dataset_name,
                        model_type=model_type,
                        data_loading_option=data_loading_option,
                        num_clients=clients_num,
                        beta=beta,
                        cfg=cfg,
                        hop=args.hop,
                        output_dir=main_output_dir,
                        logger=logger
                    )
                    results_summary.append({
                        "dataset": dataset_name,
                        "model": model_type,
                        "data_loading": data_loading_option,
                        "status": "success",
                        "file": filepath
                    })
                except Exception as e:
                    logger.error(f"Error running experiment: {e}")
                    results_summary.append({
                        "dataset": dataset_name,
                        "model": model_type,
                        "data_loading": data_loading_option,
                        "status": "failed",
                        "error": str(e)
                    })
    
    # Write summary of all experiments
    summary_path = os.path.join(main_output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Experiments Summary\n")
        f.write("==================\n\n")
        for result in results_summary:
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"Model: {result['model']}\n")
            f.write(f"Data Loading: {result['data_loading']}\n")
            f.write(f"Status: {result['status']}\n")
            if result['status'] == 'success':
                f.write(f"Results File: {result['file']}\n")
            else:
                f.write(f"Error: {result['error']}\n")
            f.write("\n")
    
    logger.info(f"All experiments completed. Summary saved to {summary_path}")

def main():
    """Main function to run the script"""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("Starting experiments")
    logger.info(f"Arguments: {args}")
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Ensure ray is shut down before starting
    if ray.is_initialized():
        ray.shutdown()
    
    if args.mode == "single":
        # Load configuration
        clients_num, beta, cfg = load_configuration(args.config_path)
        
        # Override number of clients if specified
        if args.num_clients:
            clients_num = args.num_clients
        
        # Run single experiment
        run_single_experiment(
            dataset_name=args.dataset,
            model_type=args.model,
            data_loading_option=args.data_loading,
            num_clients=clients_num,
            beta=beta,
            cfg=cfg,
            hop=args.hop,
            output_dir=args.output_dir,
            logger=logger
        )
    else:
        # Run all experiments
        run_all_experiments(args, logger)
    
    logger.info("Experiments completed")

if __name__ == "__main__":
    main() 