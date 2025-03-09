import os
import argparse
import torch
import numpy as np
import yaml
import json
import logging
import copy
import gc
import psutil
import random
import pandas as pd
from datetime import datetime
from pathlib import Path
import ray
import time

# Import FP components as needed
from utils import load_config
from dataprocessing.loaders import load_dataset, load_and_split, load_and_split_with_khop
from enhanced_training import enhanced_train, enhanced_evaluate, enhanced_test
from enhanced_models import EnhancedGCN, EnhancedSAGE, EnhancedGAT, SAGE_Products, GCN_Arxiv
from memory_efficient_models import MemoryEfficientGNN, MemoryEfficientMLP
from enhanced_client import EnhancedFLClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage for both CPU and GPU."""
    # Log CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 ** 3)  # in GB
    logger.info(f"CPU Memory Usage: {cpu_mem:.2f} GB")
    
    # Log GPU memory if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # in GB
            mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # in GB
            logger.info(f"GPU {i} Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")

def setup_logging_with_file(args):
    """Setup logging with file output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{args.dataset}_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    return log_file

def load_configuration(config_path="conf/base.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (num_clients, beta, cfg) where cfg is the full configuration dictionary
    """
    try:
        cfg = load_config(config_path)
        num_clients = cfg.get("num_clients", 3)
        beta = cfg.get("beta", 0.5)
        return num_clients, beta, cfg
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        # Return default values
        return 3, 0.5, {"num_clients": 3, "beta": 0.5, "epochs": 3, "lr": 0.01, "weight_decay": 5e-4}

def get_dataset_type(dataset_name):
    """
    Determine the type of dataset for model selection
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Dataset type (standard, arxiv, products)
    """
    # Normalize dataset name to lowercase for case-insensitive comparison
    dataset_name = dataset_name.lower()
    
    if "ogbn-arxiv" in dataset_name or dataset_name == "arxiv":
        return "arxiv"
    elif "ogbn-products" in dataset_name or dataset_name == "products":
        return "products"
    else:
        return "standard"

def get_model_for_dataset(dataset_name, num_features, num_classes, device):
    """
    Automatically select the appropriate model based on the dataset name.
    
    Args:
        dataset_name: Name of the dataset
        num_features: Number of input features
        num_classes: Number of output classes
        device: Device to place model on
    
    Returns:
        Appropriate model for the dataset
    """
    # Normalize dataset name to lowercase for case-insensitive comparison
    dataset_name = dataset_name.lower()
    
    if "ogbn-products" in dataset_name or dataset_name == "products":
        logger.info("Using specialized SAGE_Products model for OGBN-Products dataset")
        return SAGE_Products(
            nfeat=num_features, 
            nhid=256,  # Larger hidden dimension for this dataset
            nclass=num_classes,
            dropout=0.5,
            NumLayers=2
        ).to(device)
    
    elif "ogbn-arxiv" in dataset_name or dataset_name == "arxiv":
        logger.info("Using specialized GCN_Arxiv model for OGBN-Arxiv dataset")
        return GCN_Arxiv(
            nfeat=num_features, 
            nhid=256,  # Larger hidden dimension for this dataset
            nclass=num_classes,
            dropout=0.5,
            NumLayers=3
        ).to(device)
    
    else:  # Default for Cora, Citeseer, Pubmed and other small datasets
        logger.info(f"Using standard GCN model for {dataset_name}")
        return EnhancedGCN(
            nfeat=num_features, 
            nhid=16,  # Standard hidden dimension for smaller datasets
            nclass=num_classes,
            dropout=0.5,
            NumLayers=2
        ).to(device)

def load_data(data_loading_option, num_clients, beta, dataset_name, device, hop=1, fulltraining_flag=False):
    """
    Load and prepare data for federated learning
    
    Args:
        data_loading_option: Method for loading data
        num_clients: Number of clients to partition data for
        beta: Parameter for Dirichlet distribution
        dataset_name: Name of the dataset to load
        device: Device to run computation on
        hop: Number of hops for graph-based loading
        fulltraining_flag: Flag for full training mode
    
    Returns:
        Tuple of (data, dataset, clients_data, test_data)
    """
    # Normalize dataset name to handle case-insensitivity
    dataset_name_normalized = dataset_name.lower()
    
    # Apply specific modifications for known datasets to ensure proper loading
    if dataset_name_normalized == "cora":
        dataset_name_for_loading = "Cora"
    elif dataset_name_normalized == "citeseer":
        dataset_name_for_loading = "Citeseer"
    elif dataset_name_normalized == "pubmed":
        dataset_name_for_loading = "Pubmed"
    elif "ogbn-arxiv" in dataset_name_normalized:
        dataset_name_for_loading = "ogbn-arxiv"
    elif "ogbn-products" in dataset_name_normalized:
        dataset_name_for_loading = "ogbn-products"
    else:
        # If it's not a known dataset, use as-is but log a warning
        dataset_name_for_loading = dataset_name
        logger.warning(f"Unknown dataset name: {dataset_name}. Using as-is, but loading may fail.")
    
    logger.info(f"Loading dataset: {dataset_name_for_loading}")
    
    # Load data based on the data loading option
    if data_loading_option == "zero_hop":
        # Fix the parameter order: name, device, num_clients, beta
        data, dataset, clients_data, test_data = load_and_split(dataset_name_for_loading, device, num_clients, beta)
    elif data_loading_option in ["one_hop", "two_hop", "three_hop"]:
        hop_count = {"one_hop": 1, "two_hop": 2, "three_hop": 3}[data_loading_option]
        # Fix the parameter order: name, device, num_clients, beta, hop, fulltraining_flag
        data, dataset, clients_data, test_data = load_and_split_with_khop(
            dataset_name_for_loading, device, num_clients, beta, hop_count, fulltraining_flag
        )
    else:
        raise ValueError(f"Unsupported data loading option: {data_loading_option}")
        
    return data, dataset, clients_data, test_data

def initialize_clients(data, dataset, clients_data, model_type, cfg, device, dataset_type="standard", batch_size=1024, num_neighbors=None):
    """
    Initialize federated learning clients
    
    Args:
        data: PyG Data object containing the graph
        dataset: PyG Dataset object 
        clients_data: List of client data
        model_type: Type of GNN model (or model name for automatic selection)
        cfg: Configuration dictionary
        device: Device to run computation on
        dataset_type: Type of dataset - "standard", "arxiv", or "products"
        batch_size: Batch size for mini-batch training
        num_neighbors: Number of neighbors to sample for each layer
    
    Returns:
        List of initialized clients
    """
    clients = []
    
    epochs_per_round = cfg.get("epochs", cfg.get("local_step", 1))
    
    # Adjust training parameters for large datasets
    if dataset_type == "products":
        epochs_per_round = min(epochs_per_round, 2)  # Fewer epochs for large datasets
        
    # Get number of features and classes
    num_features = data.x.size(1)
    num_classes = dataset.num_classes
    
    # Configure hidden dimension based on dataset
    if dataset_type == "products":
        nhid = 256
    elif dataset_type == "arxiv":
        nhid = 256
    else:
        nhid = 16
    
    logger.info(f"Initializing {len(clients_data)} clients with {epochs_per_round} epochs per round")
    
    for client_id, client_data in enumerate(clients_data):
        # Create client reference actor
        client = EnhancedFLClient.remote(
            client_id=client_id,
            data=client_data,
            model_type=model_type,
            dataset_type=dataset_type,
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=nhid,
            device=device,
            epochs=epochs_per_round,
            batch_size=batch_size,
            num_neighbors=num_neighbors
        )
        clients.append(client)
    
    return clients

def save_results(results, args):
    """Save experiment results to CSV in a format that handles nested structures."""
    # Create results directory if it doesn't exist
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save experiment summary (this is the main per-round metrics data)
    if "experiment_summary" in results:
        # This is already a list of dictionaries with consistent structure, perfect for a DataFrame
        summary_df = pd.DataFrame(results["experiment_summary"])
        summary_filename = f"{args.dataset}_{args.data_loading}_summary_{timestamp}.csv"
        summary_filepath = os.path.join(results_dir, summary_filename)
        summary_df.to_csv(summary_filepath, index=False)
        logger.info(f"Experiment summary saved to {summary_filepath}")
    
    # Save configuration as a separate file
    if "config" in results:
        config_filename = f"{args.dataset}_{args.data_loading}_config_{timestamp}.json"
        config_filepath = os.path.join(results_dir, config_filename)
        with open(config_filepath, 'w') as f:
            import json
            # Convert any values that might not be JSON serializable
            config_dict = {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v 
                         for k, v in results["config"].items()}
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {config_filepath}")
        
    # Save client metrics
    if "client_metrics" in results and results["client_metrics"]:
        # Client metrics can be complex nested structures
        # Save each client's metrics separately
        client_metrics_dir = os.path.join(results_dir, f"client_metrics_{timestamp}")
        os.makedirs(client_metrics_dir, exist_ok=True)
        
        for i, client_metric in enumerate(results["client_metrics"]):
            client_filename = f"client_{i}_metrics.json"
            client_filepath = os.path.join(client_metrics_dir, client_filename)
            
            with open(client_filepath, 'w') as f:
                import json
                # Convert any values that might not be JSON serializable
                # For lists of values, we convert them to regular floats
                clean_metrics = {}
                for k, v in client_metric.items():
                    if isinstance(v, list):
                        # Convert list items to basic types
                        clean_metrics[k] = [float(item) if isinstance(item, (int, float)) else str(item) for item in v]
                    else:
                        clean_metrics[k] = str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v
                
                json.dump(clean_metrics, f, indent=2)
        
        logger.info(f"Client metrics saved to {client_metrics_dir}")
    
    # Save server metrics
    if "server_metrics" in results:
        server_filename = f"{args.dataset}_{args.data_loading}_server_metrics_{timestamp}.json"
        server_filepath = os.path.join(results_dir, server_filename)
        
        with open(server_filepath, 'w') as f:
            import json
            # Convert any values that might not be JSON serializable
            clean_metrics = {}
            for k, v in results["server_metrics"].items():
                if isinstance(v, list):
                    # Convert list items to basic types
                    clean_metrics[k] = [float(item) if isinstance(item, (int, float)) else str(item) for item in v]
                else:
                    clean_metrics[k] = str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v
            
            json.dump(clean_metrics, f, indent=2)
        
        logger.info(f"Server metrics saved to {server_filepath}")
    
    # Return the path to the main summary file
    if "experiment_summary" in results:
        return summary_filepath
    else:
        # If there's no summary, return a dummy filename
        return os.path.join(results_dir, f"{args.dataset}_{args.data_loading}_{timestamp}.txt")

def run_federated_learning(args):
    """
    Run the enhanced federated learning process
    
    Args:
        args: Command-line arguments
    
    Returns:
        Dictionary of results
    """
    # Setup device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("Using CPU")
    
    # Clean up memory before starting
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load configuration
    if args.config:
        cfg = load_configuration(args.config)
    else:
        num_clients, beta, cfg = load_configuration()
    
    # Determine dataset type
    dataset_type = get_dataset_type(args.dataset)
    
    # Log memory usage before loading data
    log_memory_usage()
    
    # Reduce rounds and epochs for very large datasets
    if dataset_type == "products":
        logger.info("Memory optimization for OGBN-Products dataset")
        # Adjust clients for OGBN-Products to prevent OOM
        if args.num_clients > 2:
            original_num_clients = args.num_clients
            args.num_clients = 2  # Limit to 2 clients for OGBN-Products
            logger.warning(f"Reducing number of clients from {original_num_clients} to {args.num_clients} for OGBN-Products dataset to save memory")
    
    # Load and prepare data
    data, dataset, clients_data, test_data = load_data(
        args.data_loading,
        args.num_clients,
        args.beta,
        args.dataset,
        device,
        args.hop,
        args.fulltraining
    )
    
    # Log dataset info
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Number of nodes: {data.x.size(0)}")
    logger.info(f"Number of features: {data.x.size(1)}")
    logger.info(f"Number of classes: {dataset.num_classes}")
    if hasattr(data, 'edge_index'):
        logger.info(f"Number of edges: {data.edge_index.size(1)}")
    
    # Log memory usage after loading data
    log_memory_usage()
    
    # Get number of features and classes
    num_features = data.x.size(1)
    num_classes = dataset.num_classes
    
    # Setup global model - automatically select based on dataset
    global_model = get_model_for_dataset(args.dataset, num_features, num_classes, device)
    logger.info(f"Global model type: {type(global_model).__name__}")
    
    # Log model parameters
    total_params = sum(p.numel() for p in global_model.parameters())
    logger.info(f"Total model parameters: {total_params}")
    
    # Extract training parameters from config
    epochs_per_round = cfg.get("epochs", cfg.get("local_step", 1))
    total_rounds = args.rounds or cfg.get("global_step", 20)
    
    # Reduce rounds and epochs for very large datasets
    if dataset_type == "products":
        # Limit rounds for large datasets to prevent OOM errors
        total_rounds = min(total_rounds, 10)
        epochs_per_round = min(epochs_per_round, 2)
        
    logger.info(f"Training for {total_rounds} rounds with {epochs_per_round} local epochs per round")
    
    # For mini-batch training
    batch_size = args.batch_size
    # Reduce batch size for large datasets
    if dataset_type == "products":
        batch_size = min(batch_size, 512)
        
    num_neighbors = args.num_neighbors
    # Use smaller neighborhood sampling for products
    if dataset_type == "products" and num_neighbors is None:
        num_neighbors = [5]  # Reduce neighbor sampling for products
    
    # Initialize clients with automatically selected model type
    model_type = type(global_model).__name__
    clients = initialize_clients(
        data, dataset, clients_data, model_type, cfg, device, dataset_type,
        batch_size=batch_size, num_neighbors=num_neighbors
    )
    
    logger.info(f"Initialized {len(clients)} clients")
    
    # Initialize server model (global model)
    server_model = copy.deepcopy(global_model)
    
    # Training metrics
    global_train_losses = []
    global_train_accs = []
    global_test_accs = []
    
    client_metrics = []
    server_metrics = []
    experiment_summary = []
    
    # Get the number of rounds from args or config
    rounds = min(total_rounds, 10) if dataset_type == "products" else total_rounds
    
    logger.info(f"Starting federated training for {rounds} rounds")
    
    for round_num in range(rounds):
        logger.info(f"Round {round_num+1}/{rounds}")
        
        # Log memory usage before round begins
        log_memory_usage()
        
        # Clean up memory before each round
        gc.collect()
        torch.cuda.empty_cache()
        
        # Train clients
        logger.info(f"Starting client training for global round {round_num}")
        
        train_results = []
        for i, client in enumerate(clients):
            try:
                # Train each client
                train_result = ray.get(client.train_client.remote())
                train_results.append(train_result)
                
                # Clean memory after each client to prevent accumulation
                if dataset_type == "products":
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info(f"Completed training for client {i+1}/{len(clients)}")
            except Exception as e:
                logger.error(f"Error training client: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Log training metrics - now handling 4-value tuples: (loss, acc, val_acc, test_acc)
        if train_results:
            avg_train_loss = sum(result[0] for result in train_results) / len(train_results)
            avg_train_acc = sum(result[1] for result in train_results) / len(train_results)
            avg_val_acc = sum(result[2] for result in train_results) / len(train_results)
            avg_test_acc = sum(result[3] for result in train_results) / len(train_results)
            logger.info(f"Average metrics - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f}, Test Acc: {avg_test_acc:.4f}")
        else:
            avg_train_loss, avg_train_acc, avg_val_acc, avg_test_acc = float('inf'), 0, 0, 0
        
        global_train_losses.append(avg_train_loss)
        global_train_accs.append(avg_train_acc)
        
        # Aggregate client parameters
        logger.info("Collecting client parameters for aggregation")
        try:
            # Get parameters from all clients
            client_params = [ray.get(client.get_params.remote()) for client in clients]
            
            # Perform federated averaging (basic implementation)
            with torch.no_grad():
                # Create a state dict from parameter tuples
                avg_state_dict = {}
                for name, param in client_params[0]:
                    # Initialize with zeros of the right shape and type
                    avg_state_dict[name] = torch.zeros_like(param)
                
                # Sum parameters from all clients
                for params in client_params:
                    params_dict = dict(params)
                    for name, param in params_dict.items():
                        avg_state_dict[name] += param
                
                # Divide by number of clients to get average
                num_clients = len(client_params)
                for name in avg_state_dict:
                    # Handle integer tensors differently - we need to convert them to float before division
                    if avg_state_dict[name].dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
                        # Convert to float, divide, then convert back to original type
                        original_type = avg_state_dict[name].dtype
                        avg_state_dict[name] = (avg_state_dict[name].float() / num_clients).to(original_type)
                    else:
                        # For floating point tensors, division works directly
                        avg_state_dict[name] /= num_clients
                
                # Update the server model
                server_model.load_state_dict(avg_state_dict)
            
            # Distribute updated model to clients
            for client in clients:
                client.update_params.remote(tuple(avg_state_dict.items()), round_num)
                
            logger.info(f"Completed global round {round_num}")
            
            # Clean up memory after parameter aggregation
            if dataset_type == "products":
                del avg_state_dict, client_params
                gc.collect()
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error in parameter aggregation: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Evaluate performance
        if (round_num + 1) % 1 == 0:  # Evaluate every round by default
            try:
                # Test on global model
                server_model.to(device)
                if test_data is not None:
                    # If we have specific test data, use it
                    if isinstance(test_data, list) and len(test_data) > 0:
                        # Use first test data for global evaluation
                        test_data_item = test_data[0]
                        if hasattr(test_data_item, 'to'):
                            test_data_item = test_data_item.to(device)
                        global_test_acc = enhanced_test(server_model, test_data_item, dataset_type=dataset_type)
                    else:
                        # Single test data object
                        if hasattr(test_data, 'to'):
                            test_data = test_data.to(device)
                        global_test_acc = enhanced_test(server_model, test_data, dataset_type=dataset_type)
                else:
                    # Use data.test_mask for evaluation
                    global_test_acc = enhanced_test(server_model, data, dataset_type=dataset_type)
                
                logger.info(f"Global model - Test Accuracy: {global_test_acc:.4f}")
                global_test_accs.append(global_test_acc)
        
                # We already have client test results from training, so we don't need to test again
                # Record metrics with results we already have from training
                round_results = {
                    "round": round_num + 1,
                    "avg_train_loss": avg_train_loss,
                    "avg_train_acc": avg_train_acc,
                    "avg_val_acc": avg_val_acc,
                    "avg_test_acc": avg_test_acc,
                    "global_test_acc": global_test_acc,
                    "dataset": args.dataset,
                    "model_type": getattr(global_model, "model_type", type(global_model).__name__),
                    "data_loading": args.data_loading,
                    "num_clients": args.num_clients,
                    "beta": args.beta,
                    "hop": args.hop,
                }
                experiment_summary.append(round_results)
                
                logger.info(f"Round {round_num+1} - Avg Train Loss: {avg_train_loss:.4f}, " 
                           f"Avg Train Acc: {avg_train_acc:.4f}, Avg Val Acc: {avg_val_acc:.4f}, "
                           f"Avg Test Acc: {avg_test_acc:.4f}, Global Test Acc: {global_test_acc:.4f}")
                
                # Clean up memory after evaluation
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Record partial metrics
                round_results = {
                    "round": round_num + 1,
                    "avg_train_loss": avg_train_loss,
                    "avg_train_acc": avg_train_acc,
                    "avg_val_acc": avg_val_acc,
                    "avg_test_acc": avg_test_acc,
                    "global_test_acc": None,
                    "dataset": args.dataset,
                    "model_type": getattr(global_model, "model_type", type(global_model).__name__),
                    "data_loading": args.data_loading,
                    "num_clients": args.num_clients,
                    "beta": args.beta,
                    "hop": args.hop,
                }
                experiment_summary.append(round_results)
    
    # Get final client metrics
    try:
        client_metrics = [ray.get(client.get_loss_acc.remote()) for client in clients]
    except Exception as e:
        logger.error(f"Error getting client metrics: {e}")
        client_metrics = []
    
    # Combine all results
    final_results = {
        "params": {
            "dataset": args.dataset,
            "model_type": args.model_type,
            "data_loading": args.data_loading,
            "num_clients": args.num_clients,
            "beta": args.beta,
            "hop": args.hop,
            "fulltraining": args.fulltraining,
            "total_rounds": rounds,
            "epochs_per_round": epochs_per_round,
        },
        "metrics": {
            "global_train_losses": global_train_losses,
            "global_train_accs": global_train_accs,
            "global_test_accs": global_test_accs,
            "client_metrics": client_metrics,
        },
        "experiment_summary": experiment_summary,
    }
    
    # Generate results filename
    results_file = generate_results_filename(args, timestamp=time.strftime("%Y%m%d-%H%M%S"))
    
    return final_results, results_file

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Federated Learning with GNNs')
    
    # Dataset and data loading
    parser.add_argument('--dataset', type=str, default='Cora', 
                        help='Dataset name (Cora, Citeseer, Pubmed, ogbn-arxiv, ogbn-products)')
    parser.add_argument('--data_loading', type=str, default='zero_hop', 
                        choices=['zero_hop', 'one_hop', 'two_hop', 'three_hop'],
                        help='Data loading strategy')
    parser.add_argument('--hop', type=int, default=1, 
                        help='Number of hops for data loading')
    
    # Client configurations
    parser.add_argument('--num_clients', type=int, default=5, 
                        help='Number of clients')
    parser.add_argument('--beta', type=float, default=0.5, 
                        help='Beta parameter for Dirichlet distribution')
    
    # Model configurations
    parser.add_argument('--model_type', type=str, default=None, 
                        choices=['GCN', 'GAT', 'SAGE', 'VanillaGNN', 'MLP'],
                        help='Type of GNN model (optional, auto-selected based on dataset if not provided)')
    
    # Training configurations
    parser.add_argument('--rounds', type=int, default=None, 
                        help='Number of communication rounds')
    parser.add_argument('--batch_size', type=int, default=1024, 
                        help='Batch size for mini-batch training (used for large datasets)')
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=None, 
                        help='Number of neighbors to sample for each node in mini-batch training')
    
    # Environment configurations
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='Use CUDA for training')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--memory_efficient', action='store_true', default=False, 
                        help='Use memory-efficient operations')
    parser.add_argument('--fulltraining', action='store_true', default=False, 
                        help='Use full training mode')
    
    # Logging and configuration
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to configuration file')
    parser.add_argument('--results_dir', type=str, default='results', 
                        help='Directory to save results')
    
    return parser.parse_args()

def init_ray(args=None):
    """Initialize Ray for distributed computing."""
    if not ray.is_initialized():
        try:
            # Basic initialization for most use cases
            ray.init(ignore_reinit_error=True)
            logger.info("Ray initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Ray: {e}")
            # Fallback initialization with minimal settings
            try:
                ray.init(ignore_reinit_error=True, local_mode=True)
                logger.warning("Ray initialized in local mode due to initialization error")
            except Exception as e2:
                logger.error(f"Failed to initialize Ray even in local mode: {e2}")
                raise RuntimeError("Cannot initialize Ray. Please check your Ray installation.")

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def generate_results_filename(args, timestamp=None):
    """
    Generate a filename for saving results
    
    Args:
        args: Command-line arguments
        timestamp: Optional timestamp to add to filename
    
    Returns:
        Path to save results to
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create a descriptive filename
    dataset_name = args.dataset.lower()
    data_loading = args.data_loading
    
    # Include model information - auto-selected or explicitly provided
    model_info = args.model_type if args.model_type else "auto"
    
    # Include number of clients
    clients_info = f"c{args.num_clients}"
    
    # Include beta parameter if non-default
    beta_info = f"b{args.beta}" if args.beta != 0.5 else ""
    
    # Include memory-efficient flag if used
    mem_info = "mem" if args.memory_efficient else ""
    
    # Include batch size if a large dataset
    batch_info = ""
    if dataset_name in ["ogbn-products", "ogbn-arxiv"]:
        batch_info = f"bs{args.batch_size}"
    
    # Combine all parts
    filename_parts = [
        dataset_name,
        data_loading,
        model_info,
        clients_info,
        beta_info,
        mem_info,
        batch_info,
        timestamp
    ]
    
    # Filter out empty parts
    filename_parts = [part for part in filename_parts if part]
    
    # Join with underscores
    filename = "_".join(filename_parts) + ".json"
    
    # Create results directory if it doesn't exist
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    return os.path.join(results_dir, filename)

if __name__ == '__main__':
    args = parse_args()
    
    # Configure logging
    setup_logging_with_file(args)
    
    logger.info(f"Starting enhanced federated learning with {args.dataset} dataset")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    if args.memory_efficient:
        logger.info("Using memory-efficient models")
    
    # For mini-batch training
    if args.dataset.lower() in ["ogbn-products"]:
        logger.info(f"Using mini-batch training with batch size {args.batch_size}")
        logger.info(f"Neighbor sampling: {args.num_neighbors}")
    
    # Initialize Ray
    init_ray(args)
    
    # Run federated learning
    results, results_file = run_federated_learning(args)
    
    # Save results
    save_results(results, args)
    
    logger.info("Training completed successfully") 