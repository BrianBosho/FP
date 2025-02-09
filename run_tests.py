from federated_learning import load_configuration, main_experiment

if __name__ == "__main__":
    data_loading_option = "processed_data"  # Options: "processed_data", "processed_data_with_hop", "feature_prop"
    model_type = "GCN"  # Options: "GCN", "GAT"
    
    clients_num, beta, cfg = load_configuration()
    main_experiment(clients_num, beta, data_loading_option, model_type, cfg)
