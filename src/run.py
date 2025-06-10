import torch
import ray
from client import FLClient
from models import GCN, GAT, GCN_arxiv, GraphSAGEProducts
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
from run_utils import (
    setup_logging, 
    log_training_results, 
    log_evaluation_results, 
    save_results_to_csv,
    compare_model_parameters,
    prepare_results_data,
    compute_experiment_statistics,
    generate_experiment_output
)
import gc

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"DEVICE: {DEVICE}")

def load_configuration(config_path="conf/base.yaml"):
    cfg = load_config(config_path)
    return cfg["num_clients"], cfg["beta"], cfg

def instantiate_model(model_type,  num_features, num_classes, device, dataset_name="Cora"):
    DEVICE = device
    if model_type == "GCN":
        if dataset_name == "ogbn-arxiv": # 
            model = GCN_arxiv(input_dim=num_features, hidden_dim=256, output_dim=40, dropout=0.5)
            print(f"Model is {model}")
            return model.to(DEVICE)
        elif dataset_name == "ogbn-products":
            model = GraphSAGEProducts(input_dim=num_features, hidden_dim=256, output_dim=47, dropout=0.5, num_layers=3)
            print(f"Model is {model}")
            return model.to(DEVICE)
        else:
            return GCN(num_features, 16, num_classes).to(DEVICE)
    elif model_type == "GAT":
        return GAT(num_features, 16, num_classes).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def initialize_clients(data, dataset, clients_data, model_type, cfg, device):
    DEVICE = device
    return [FLClient.remote(data.to(DEVICE), dataset, i, cfg, device, model_type) for i, data in enumerate(clients_data)]

def load_data(data_loading_option, num_clients, beta, dataset_name, device, hop = 1, fulltraining_flag = False, config = None):
    """
    Args:
        dat_loading_option: full_dataset, split_dataset, split_dataset_with_khop, split_dataset_with_feature_prop
        num_clients: number of clients
        beta: beta for dirichlet distribution
        dataset_name: name of the dataset
        hop: number of hops for k-hop subgraph
        imputation_method: zero, propagation, full
        fulltraining_flag: if True, use full training
        config: Configuration dictionary from YAML file (optional)
    """

    kh_options = ["page_rank", "random_walk", "diffusion", "efficient", "adjacency", "propagation", "zero", "propagation", "full"]
    if data_loading_option == "full_dataset":
        return load_dataset(dataset_name)
    elif data_loading_option == "zero_hop":
        return load_and_split(dataset_name, device, num_clients, beta, config=config)

    elif data_loading_option in kh_options:
        return load_and_split_with_khop(
            dataset_name, 
            device, 
            num_clients, 
            beta, 
            hop=hop, 
            imputation_method=data_loading_option, 
            fulltraining_flag=fulltraining_flag, 
            config=config
        )
 

def run_with_server(dataset_name, num_clients, beta, data_loading_option, model_type, cfg, device, hop = 1, fulltraining_flag = False):
    DEVICE = device
    
    print(f"data_loading_option: {data_loading_option}")

    data, dataset, clients_data, test_data = load_data(
        data_loading_option, 
        num_clients, 
        beta, 
        dataset_name, 
        device=DEVICE, 
        hop=hop, 
        fulltraining_flag=fulltraining_flag,
        config=cfg
    )
    test_data = clients_data
    print("Data loaded")
    data = data.to(DEVICE)
    print(f"Device is {DEVICE}")
    
    print(dataset)
    print(len(clients_data))

    print(f"Shape of clients_data: {clients_data[0].x.shape}")
    input_dim = clients_data[0].x.size(1)
    
    rounds = cfg['num_rounds']
    model = instantiate_model(model_type, input_dim, dataset.num_classes, DEVICE, dataset_name)
    clients = initialize_clients(data, dataset, clients_data, model_type, cfg, DEVICE)
    server = Server(clients, model, device)

    try:
        train_results = [server.train_clients(i) for i in range(rounds)]
        log_training_results(train_results)
        
        eval_results = server.evaluate_clients()
        log_evaluation_results(eval_results)
        
        training_results = ray.get([client.get_loss_acc.remote() for client in server.clients])
        save_results_to_csv(training_results)

        are_params_identical = compare_model_parameters(server.model, server.clients)
        print(f"\nAll model parameters are identical: {are_params_identical}")

        if dataset_name == "ogbn-arxiv" or dataset_name == "ogbn-products":
            test_results = server.test_global_model(clients_data[0])
            client_test_results = ray.get([client.test.remote(test) for client, test in zip(server.clients, test_data)])
        else:
            test_results = server.test_global_model(data)
            client_test_results = ray.get([client.test.remote(test) for client, test in zip(server.clients, test_data)])
        
        average_results = sum(client_test_results) / len(client_test_results)
        print(f"The average client test results: {average_results}")
        print(f"The final global test results: {test_results}")

        return test_results, average_results
    finally:
        # Clean up resources
        for client in clients:
            ray.kill(client)
        torch.cuda.empty_cache()
        gc.collect()

def main_experiment(clients_num, beta, data_loading_option, model_type, cfg, dataset_name = "Cora", hop = 1, fulltraining_flag = False):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_results = []
    client_test_results = []
    print(f"DEVICE: {DEVICE}")
    
    # Adjust clients_num based on dataset to avoid OOM
    adjusted_clients = clients_num
    if dataset_name == "ogbn-products":
        adjusted_clients = min(5, clients_num)
        print(f"Adjusting number of clients from {clients_num} to {adjusted_clients} for {dataset_name} dataset to prevent memory issues")
    
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

    try:
        # Initialize Ray with memory management settings
        ray.init(
            num_gpus=1, 
            ignore_reinit_error=True,
            object_store_memory=10 * 1024 * 1024 * 1024,  # 10GB for object store
            _system_config={
                "object_spilling_threshold": 0.8,
                "max_io_workers": 4,
            }
        )
        
        for i in range(1):  # Change 1 to the desired number of repetitions
            try:
                # Clear CUDA cache before each iteration
                torch.cuda.empty_cache()
                gc.collect()
                
                global_results, client_results = run_with_server(
                    dataset_name, 
                    clients_num, 
                    beta, 
                    data_loading_option, 
                    model_type, 
                    cfg, 
                    DEVICE, 
                    hop=hop, 
                    fulltraining_flag=fulltraining_flag
                )
                test_results.append(global_results)
                client_test_results.append(client_results)
                print(f"Round {i+1} is complete")
                
                results_data["rounds"].append({
                    "round": i+1,
                    "global_result": float(global_results),
                    "client_result": float(client_results)
                })
            except Exception as e:
                print(f"Error in round {i+1}: {e}")
                # Clear resources even if there's an error
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        print(f"The global test results: {test_results}")
        print(f"The client test results: {client_test_results}")

        average_global_results = np.mean(test_results)
        average_client_results = np.mean(client_test_results)

        std_global = np.std(test_results)
        std_client = np.std(client_test_results)

        print(f"The average global test results: {average_global_results}")
        print(f"The average client test results: {average_client_results}")
        print(f"The standard deviation global is: {std_global}")
        print(f"The standard deviation client is: {std_client}")

        results_data["summary"] = {
            "global_results": [float(x) for x in test_results],
            "client_results": [float(x) for x in client_test_results],
            "average_global_result": float(average_global_results),
            "average_client_result": float(average_client_results),
            "std_global": float(std_global),
            "std_client": float(std_client)
        }

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
        # Make sure Ray is always shut down
        ray.shutdown()
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    return results_data, output
    

# run centralized is equal to run main_experiment with num_clients = 1, zerohop



def verify_test_masks(data):
    print("Test Mask Details:")
    print(f"Total nodes: {data.num_nodes}")
    print(f"Test mask sum: {data.test_mask.sum()}")
    print(f"Test mask indices: {torch.where(data.test_mask)[0]}")
