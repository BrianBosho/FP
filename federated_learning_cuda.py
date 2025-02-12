import torch
import ray
from client import FLClient
from models import GCN, GAT
from server import Server
import pandas as pd
from utils import load_config
from dataprocessingset import load_processed_data, load_processed_data_with_hop
from feature_propagation import load_with_feature_prop, load_with_no_feature_prop, load_ogbn_arxiv
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

def load_configuration(config_path="conf/base.yaml"):
    cfg = load_config(config_path)
    return cfg["num_clients"], cfg["beta"], cfg

def instantiate_model(model_type, num_features, num_classes):
    if model_type == "GCN":
        return GCN(num_features, 16, num_classes).to(DEVICE)
    elif model_type == "GAT":
        return GAT(num_features, 16, num_classes).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def initialize_clients(data, cora_dataset, clients_data, model_type, cfg):
    return [FLClient.remote(data.to(DEVICE), cora_dataset, i, cfg, model_type) for i, data in enumerate(clients_data)]

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

def load_data(data_loading_option, num_clients, beta):
    print(f"Loading data with option: {data_loading_option}")
    if data_loading_option == "processed_data":
        return load_processed_data(num_clients=num_clients, beta=beta)
    elif data_loading_option == "processed_data_with_hop":
        return load_processed_data_with_hop(num_clients=num_clients, beta=beta, hop=2)
    elif data_loading_option == "feature_prop":
        return load_with_feature_prop(num_clients=num_clients, beta=beta, hop=2)
    elif data_loading_option == "no_feature_prop":
        return load_with_no_feature_prop(num_clients=num_clients, beta=beta, hop=2)
    elif data_loading_option == "ogbn-arxiv":
        print("Loading ogbn-arxiv")
        return load_ogbn_arxiv(num_clients=num_clients, beta=beta, hop = 2)
    else:
        raise ValueError(f"Unsupported data loading option: {data_loading_option}")

def run_with_server(num_clients, beta, data_loading_option, model_type, cfg):
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
    
    ray.init(num_gpus=1)
    print(f"data_loading_option: {data_loading_option}")


    if data_loading_option == "feature_prop" or data_loading_option == "no_feature_prop" or data_loading_option == "ogbn-arxiv":
        data, cora_dataset, clients_data, test_data = load_data(data_loading_option, num_clients, beta)
    else:
        data, cora_dataset, clients_data = load_data(data_loading_option, num_clients, beta)
    test_data = clients_data
    data = data.to(DEVICE)
    print(f"Device is {DEVICE}")
    
    print(cora_dataset)
    print(len(clients_data))
    
    rounds = cfg['num_rounds']
    model = instantiate_model(model_type, cora_dataset.num_features, cora_dataset.num_classes)
    clients = initialize_clients(data, cora_dataset, clients_data, model_type, cfg)
    server = Server(clients, model)
    
    train_results = [server.train_clients(i) for i in range(rounds)]
    log_training_results(train_results)
    
    eval_results = server.evaluate_clients()
    log_evaluation_results(eval_results)
    
    training_results = ray.get([client.get_loss_acc.remote() for client in server.clients])
    save_results_to_csv(training_results)
    
    test_results = server.test_global_model(cora_dataset)
    client_test_results = ray.get([client.test.remote(test.to(DEVICE)) for client, test in zip(server.clients, test_data)])
    
    average_results = sum(client_test_results) / len(client_test_results)
    print(f"The average client test results: {average_results}")
    print(f"The final global test results: {test_results}")
    
    ray.shutdown()
    return test_results, average_results

def main_experiment(clients_num, beta, data_loading_option, model_type, cfg):
    test_results = []
    client_test_results = []
    print(f"DEVICE: {DEVICE}")
    output = f"DEVICE: {DEVICE}\n"
    output += f"Data loading option is {data_loading_option}\n"
    output += f"Model type is {model_type}\n"

    
    for i in range(1):  # Change 1 to the desired number of repetitions
        global_results, client_results = run_with_server(clients_num, beta, data_loading_option, model_type, cfg)
        test_results.append(global_results)
        client_test_results.append(client_results)
        print(f"Round {i+1} is complete")
    
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

    output += f"The global test results: {test_results}\n"
    output += f"The client test results: {client_test_results}\n"
    output += f"The average global test results: {average_global_results}\n"
    output += f"The average client test results: {average_client_results}\n"
    output += f"The standard deviation global is: {std_global}\n"
    output += f"The standard deviation client is: {std_client}\n"

    return output
    
    