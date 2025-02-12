import torch
import ray
import pandas as pd
from client import FLClient
from models import GCN, GAT
from server import Server
from utils import load_config
from dataprocessingset import load_processed_data
from feature_propagation import load_with_feature_prop

def initialize_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(num_clients, beta, data_type='normal', hop=None):
    if data_type == 'facebook':
        return load_facebook_data(num_clients=num_clients, beta=beta)
    elif data_type == 'hop':
        return load_processed_data_with_hop(num_clients=num_clients, beta=beta, hop=hop)
    elif data_type == 'feature_prop':
        return load_with_feature_prop(num_clients=num_clients, beta=beta, hop=hop)
    return load_processed_data(num_clients=num_clients, beta=beta)

def init_model(model_type, num_features, num_classes):
    if model_type == "GCN":
        return GCN(num_features, 16, num_classes)
    elif model_type == "GAT":
        return GAT(num_features, 16, num_classes)
    raise ValueError("Unsupported model type")

def run_experiments(num_clients, beta, cfg, model_type="GAT"):
    DEVICE = initialize_device()
    ray.init()
    data, cora_dataset, clients_data = load_data(num_clients, beta, data_type='normal')
    test_data = clients_data if 'test_data' not in locals() else test_data
    
    data = data.to(DEVICE)
    print(cora_dataset, len(clients_data))

    model = init_model(model_type, cora_dataset.num_features, cora_dataset.num_classes)
    clients = [FLClient.remote(data, cora_dataset, i, cfg, model_type) for i, data in enumerate(clients_data)]
    server = Server(clients, model)

    train_results, eval_results = run_training_rounds(server, cfg['num_rounds'])
    final_results = evaluate_models(server, cora_dataset, test_data)
    ray.shutdown()
    return final_results

def run_training_rounds(server, num_rounds):
    train_results = []
    for i in range(num_rounds):
        result = server.train_clients(i)
        print(f"Round {i+1} is complete")
        train_results.append(result)
    return train_results, server.evaluate_clients()

def evaluate_models(server, dataset, test_data):
    criterion = torch.nn.CrossEntropyLoss()
    test_results = server.test_global_model(dataset)
    client_test_results = [client.test.remote(test) for client, test in zip(server.clients, test_data)]
    client_results = ray.get(client_test_results)
    print(f"The final global test results: {test_results}")
    return test_results, sum(client_results) / len(client_results)

def main():
    cfg = load_config("conf/base.yaml")
    results = run_experiments(cfg["num_clients"], cfg["beta"], cfg)
    print(f"Global Test Results: {results[0]}, Average Client Results: {results[1]}")

if __name__ == "__main__":
    main()
