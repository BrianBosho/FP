# run_utils.py
import torch
import ray
import numpy as np
import pandas as pd
import logging
import torch.nn.functional as F
import os

def setup_logging(log_dir="logs"):
    """
    Set up logging configuration for experiments
    
    Args:
        log_dir (str): Directory to store log files
    
    Returns:
        logging.Logger: Configured logger
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def log_training_results(train_results):
    """
    Log training results for each round
    
    Args:
        train_results (list): List of training results from each round
    """
    print("Training done")
    for i, results in enumerate(train_results):
        print(f"Round {i+1}")
        for loss, acc in results:
            print(f"Train Loss: {loss:.3f}, Train Accuracy: {acc:.3f}")

def log_evaluation_results(eval_results):
    """
    Log evaluation results
    
    Args:
        eval_results (list): List of evaluation results
    """
    for loss, acc in eval_results:
        print(f"Validation Loss: {loss:.3f}, Validation Accuracy: {acc:.3f}")

def save_results_to_csv(results, filename="results.csv"):
    """
    Save results to a CSV file
    
    Args:
        results (list): Results to be saved
        filename (str): Name of the output CSV file
    """
    # Check if environment variables are set for experiment directory and timestamp
    exp_dir = os.environ.get("EXPERIMENT_RESULTS_DIR")
    timestamp = os.environ.get("EXPERIMENT_TIMESTAMP")
    
    if exp_dir and timestamp and filename == "results.csv":
        # Extract experiment name from directory path
        experiment_name = os.path.basename(exp_dir)
        # Create a custom filename in the experiment directory
        custom_filename = os.path.join(exp_dir, f"training_{experiment_name}_{timestamp}.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(custom_filename)
        print(f"Training CSV results saved to {custom_filename}")
        
        # Also save to the default location for backward compatibility
        results_df.to_csv(filename)
    else:
        # Original behavior
        results_df = pd.DataFrame(results)
        results_df.to_csv(filename)

def compare_model_parameters(server_model, clients, debug=False):
    """
    Compare model parameters between server and clients
    
    Args:
        server_model (torch.nn.Module): Server's model
        clients (list): List of client actors
        debug (bool): Whether to print detailed comparison info
    
    Returns:
        bool: Whether all parameters are identical
    """
    # Get server parameters
    server_params = list(server_model.parameters())
    
    # Collect client parameters (now returns dict with 'params' and 'buffers')
    client_params_dicts = ray.get([client.get_params.remote() for client in clients])
    
    # Extract just the parameters from the dictionaries
    client_params_list = [params_dict['params'] for params_dict in client_params_dicts]
    
    if debug:
        print("\n--- Model Parameter Comparison ---")
    
    all_match = True
    for layer_idx, (server_param, client_params) in enumerate(zip(server_params, zip(*client_params_list))):
        # Convert to numpy for easier comparison
        server_param_np = server_param.detach().cpu().numpy()
        client_params_np = [p.detach().cpu().numpy() for p in client_params]
        
        # Check if all client params match server params
        layer_match = all(np.array_equal(server_param_np, client_param_np) 
                         for client_param_np in client_params_np)
        
        if not layer_match:
            all_match = False
        
        if debug:
            # Check shape
            print(f"\nLayer {layer_idx}:")
            print(f"Server param shape: {server_param_np.shape}")
            print(f"Client param shapes: {[p.shape for p in client_params_np]}")
            
            # Detailed difference if not matching
            if not layer_match:
                print("❌ Parameters do NOT match!")
                for client_idx, client_param_np in enumerate(client_params_np):
                    if not np.array_equal(server_param_np, client_param_np):
                        print(f"  Client {client_idx} differs:")
                        print(f"    Max absolute difference: {np.max(np.abs(server_param_np - client_param_np))}")
                        print(f"    Mean absolute difference: {np.mean(np.abs(server_param_np - client_param_np))}")
            else:
                print("✅ Parameters match!")
    
    return all_match

def prepare_results_data(device, data_loading_option, model_type, dataset_name, 
                          clients_num, beta, hop, fulltraining_flag):
    """
    Prepare a structured dictionary to store experiment results
    
    Args:
        device (torch.device): Computation device
        data_loading_option (str): Data loading method
        model_type (str): Type of model used
        dataset_name (str): Name of the dataset
        clients_num (int): Number of clients
        beta (float): Dirichlet distribution parameter
        hop (int): Number of hops
        fulltraining_flag (bool): Full training flag
    
    Returns:
        dict: Structured results dictionary
    """
    return {
        "experiment_config": {
            "device": str(device),
            "data_loading_option": data_loading_option,
            "model_type": model_type,
            "dataset": dataset_name,
            "num_clients": clients_num,
            "beta": beta,
            "hop": hop,
            "fulltraining_flag": fulltraining_flag
        },
        "rounds": []
    }

def compute_experiment_statistics(test_results, client_test_results):
    """
    Compute statistical metrics for experiment results
    
    Args:
        test_results (list): Global test results
        client_test_results (list): Client test results
    
    Returns:
        dict: Dictionary of statistical metrics
    """
    average_global_results = np.mean(test_results)
    average_client_results = np.mean(client_test_results)

    std_global = np.std(test_results)
    std_client = np.std(client_test_results)

    return {
        "global_results": [float(x) for x in test_results],
        "client_results": [float(x) for x in client_test_results],
        "average_global_result": float(average_global_results),
        "average_client_result": float(average_client_results),
        "std_global": float(std_global),
        "std_client": float(std_client)
    }

def generate_experiment_output(device, data_loading_option, model_type, 
                                fulltraining_flag, test_results, 
                                client_test_results, average_global_results, 
                                average_client_results, std_global, std_client):
    """
    Generate a comprehensive text output of experiment results
    
    Args:
        (multiple arguments describing experiment configuration and results)
    
    Returns:
        str: Formatted text output of experiment results
    """
    output = f"DEVICE: {device}\n"
    output += f"Data loading option is {data_loading_option}\n"
    output += f"Model type is {model_type}\n"
    output += f"Full training flag is {fulltraining_flag}\n"
    output += f"\nFinal Results:\n"
    output += f"The global test results: {test_results}\n"
    output += f"The client test results: {client_test_results}\n"
    output += f"The average global test results: {average_global_results}\n"
    output += f"The average client test results: {average_client_results}\n"
    output += f"The standard deviation global is: {std_global}\n"
    output += f"The standard deviation client is: {std_client}\n"
    
    return output

def verify_model_inference_mode(model):
    print("\n--- Model Inference Mode Check ---")
    print(f"Model training mode: {model.training}")
    
    # Check BatchNorm layers specifically
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            print(f"BatchNorm layer {name}:")
            print(f"  Training mode: {module.training}")
            print(f"  Track running stats: {module.track_running_stats}")

def compare_model_predictions(server_model, client_model, data):
    server_model.eval()
    client_model.eval()
    
    with torch.no_grad():
        # Server model predictions
        server_output = server_model(data.x, data.edge_index)
        server_pred = torch.argmax(server_output[data.test_mask], dim=1)
        server_probs = F.softmax(server_output[data.test_mask], dim=1)
        
        # Client model predictions
        client_output = client_model(data.x, data.edge_index)
        client_pred = torch.argmax(client_output[data.test_mask], dim=1)
        client_probs = F.softmax(client_output[data.test_mask], dim=1)
        
        # Detailed comparison
        print("\n--- Prediction Comparison ---")
        print("Server Predictions:")
        print(server_pred)
        print("\nClient Predictions:")
        print(client_pred)
        
        # Compute accuracy
        server_correct = (server_pred == data.y[data.test_mask]).float()
        client_correct = (client_pred == data.y[data.test_mask]).float()
        
        print(f"\nServer Accuracy: {server_correct.mean().item():.4f}")
        print(f"Client Accuracy: {client_correct.mean().item():.4f}")
        
        # Identify differences
        diff_mask = server_pred != client_pred
        diff_indices = torch.where(diff_mask)[0]
        
        print("\nDifference Analysis:")
        print(f"Number of different predictions: {diff_mask.sum()}")
        
        if diff_mask.sum() > 0:
            print("\nDetailed Difference:")
            for idx in diff_indices:
                print(f"Index {idx}:")
                print(f"  True Label: {data.y[data.test_mask][idx]}")
                print(f"  Server Pred: {server_pred[idx]} (Prob: {server_probs[idx].max():.4f})")
                print(f"  Client Pred: {client_pred[idx]} (Prob: {client_probs[idx].max():.4f})")

def verify_test_mask(data):
    print("\n--- Test Mask Verification ---")
    print(f"Total nodes: {data.num_nodes}")
    print(f"Test mask sum: {data.test_mask.sum()}")
    print(f"Test mask indices: {torch.where(data.test_mask)[0]}")
    
    # Distribution of classes in test mask
    unique_classes, class_counts = torch.unique(data.y[data.test_mask], return_counts=True)
    print("\nClass Distribution in Test Mask:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"Class {cls}: {count} nodes")