import torch
import ray
from client import FLClient
from models import GCN, GAT
from server import Server
import pandas as pd
from utils import load_config
from dataprocessing.loaders import (
    load_dataset,
    load_and_split,
    load_and_split_with_khop,
    load_and_split_with_feature_prop    
)
import numpy as np

import numpy as np

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"DEVICE: {DEVICE}")

def load_configuration(config_path="conf/base.yaml"):
    cfg = load_config(config_path)
    return cfg["num_clients"], cfg["beta"], cfg

def instantiate_model(model_type, num_features, num_classes, device):
    DEVICE = device
    if model_type == "GCN":
        return GCN(num_features, 16, num_classes).to(DEVICE)
    elif model_type == "GAT":
        return GAT(num_features, 16, num_classes).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def initialize_clients(data, dataset, clients_data, model_type, cfg, device):
    DEVICE = device
    return [FLClient.remote(data.to(DEVICE), dataset, i, cfg, device, model_type) for i, data in enumerate(clients_data)]

def log_training_results(train_results):
    print("Training done")
    for i, results in enumerate(train_results):
        print(f"Round {i+1}")
        for loss, acc in results:
            print(f"Train Loss: {loss:.3f}, Train Accuracy: {acc:.3f}")

def log_evaluation_results(eval_results):
    for loss, acc in eval_results:
        print(f"Validation Loss: {loss:.3f}, Validation Accuracy: {acc:.3f}")

def save_results_to_csv(results, filename="results.csv"):
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename)

def load_data(data_loading_option, num_clients, beta, dataset_name, device, hop = 1, fulltraining_flag = False):
    """
    Args:
        dat_loading_option: full_dataset, split_dataset, split_dataset_with_khop, split_dataset_with_feature_prop
        num_clients: number of clients
        beta: beta for dirichlet distribution
        dataset_name: name of the dataset
        hop: number of hops for k-hop subgraph
        imputation_method: zero, propagation, full
        fulltraining_flag: if True, use full training
    """

    kh_options = ["page_rank", "random_walk", "diffusion", "efficient", "adjacency", "propagation", "zero", "propagation", "full"]
    if data_loading_option == "full_dataset":
        return load_dataset(dataset_name)
    elif data_loading_option == "zero_hop":
        return load_and_split(dataset_name, device, num_clients, beta)

    elif data_loading_option in kh_options:
        return load_and_split_with_khop(dataset_name, device, num_clients, beta, hop=hop, imputation_method=data_loading_option, fulltraining_flag=fulltraining_flag)
 
      
        


def run_with_server(dataset_name, num_clients, beta, data_loading_option, model_type, cfg, device, hop = 1, fulltraining_flag = False):
    """
    Run federated learning with a server coordinating multiple clients.
    Args:
        num_clients (int): Number of clients participating in federated learning.
        beta (float): Parameter controlling the degree of non-IID data distribution among clients.
        data_loading_option (str): Option for loading data, can be "feature_prop", "no_feature_prop", "ogbn-arxiv", or other dataset names.
        model_type (str): Type of model to be instantiated and used for training.
        cfg (dict): Configuration dictionary containing various settings such as number of rounds.
    Returns:
        tuple: A tuple containing the final global test results and the average client test results.
    """
    DEVICE = device
    
    print(f"data_loading_option: {data_loading_option}")

    data, dataset, clients_data, test_data = load_data(data_loading_option, num_clients, beta, dataset_name, device=DEVICE, hop=hop, fulltraining_flag=fulltraining_flag)
    test_data = clients_data
    print("Data loaded")
    data = data.to(DEVICE)
    print(f"Device is {DEVICE}")
    
    print(dataset)
    print(len(clients_data))
    
    rounds = cfg['num_rounds']
    model = instantiate_model(model_type, dataset.num_features, dataset.num_classes, DEVICE)
    clients = initialize_clients(data, dataset, clients_data, model_type, cfg, DEVICE)
    server = Server(clients, model, device)
    
    train_results = [server.train_clients(i) for i in range(rounds)]
    log_training_results(train_results)
    
    eval_results = server.evaluate_clients()
    log_evaluation_results(eval_results)
    
    training_results = ray.get([client.get_loss_acc.remote() for client in server.clients])
    save_results_to_csv(training_results)
    
    test_results = server.test_global_model(dataset)
    client_test_results = ray.get([client.test.remote(test.to(DEVICE)) for client, test in zip(server.clients, test_data)])
    
    average_results = sum(client_test_results) / len(client_test_results)
    print(f"The average client test results: {average_results}")
    print(f"The final global test results: {test_results}")

    return test_results, average_results

def main_experiment(clients_num, beta, data_loading_option, model_type, cfg, dataset_name = "Cora", hop = 1, fulltraining_flag = False):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = "cpu"
    test_results = []
    client_test_results = []
    print(f"DEVICE: {DEVICE}")
    
    # Create a dictionary to store all results
    results_data = {
        "experiment_config": {
            "device": str(DEVICE),
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
    
    print(f"Data loading option is {data_loading_option}")
    print(f"Model type is {model_type}")

    # Initialize Ray once at the beginning
    try:
        ray.init(num_gpus=1, ignore_reinit_error=True)
        
        for i in range(3):  # Change 1 to the desired number of repetitions
            try:
                global_results, client_results = run_with_server(dataset_name, clients_num, beta, data_loading_option, model_type, cfg, DEVICE, hop=1, fulltraining_flag=fulltraining_flag)
                test_results.append(global_results)
                client_test_results.append(client_results)
                print(f"Round {i+1} is complete")
                
                # Store round results in the dictionary
                results_data["rounds"].append({
                    "round": i+1,
                    "global_result": float(global_results),
                    "client_result": float(client_results)
                })
            except Exception as e:
                print(f"Error in round {i+1}: {e}")
                # Continue with the next iteration
        
        # Rest of the code remains the same
        print(f"The global test results: {test_results}")
        print(f"The client test results: {client_test_results}")

        average_global_results = np.mean(test_results)
        average_client_results = np.mean(client_test_results)

        std_global = np.std(test_results)
        std_client = np.std(client_test_results)

        print(f"The average global test results: {average_global_results}")
        print(f"The average client test results: {average_client_results}")
        print(f"The standad deviation global is: {std_global}")
        print(f"The standad deviation client is: {std_client}")

        # Add summary statistics to the results dictionary
        results_data["summary"] = {
            "global_results": [float(x) for x in test_results],
            "client_results": [float(x) for x in client_test_results],
            "average_global_result": float(average_global_results),
            "average_client_result": float(average_client_results),
            "std_global": float(std_global),
            "std_client": float(std_client)
        }

        # Output remains the same
        output = f"DEVICE: {DEVICE}\n"
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
        
    finally:
        # Make sure Ray is always shut down, even if there's an exception
        ray.shutdown()
    
    # Return both structured data and text output
    return results_data, output
    
    