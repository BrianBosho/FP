import argparse
import os
import ray
from run import load_configuration, main_experiment

def run_gnn_experiments(data_options, model_types, dataset_name, clients, beta_value, hop_value):
    """Run a set of experiments with given parameters"""
    
    # Load configuration
    _, _, cfg = load_configuration()
    
    # Override with command line parameters
    clients_num = clients
    beta = beta_value
    
    # Create results directory
    main_dir = f'results_{dataset_name}_{clients_num}_clients'
    os.makedirs(main_dir, exist_ok=True)
    
    # Run all combinations
    for data_loading_option in data_options:
        for model_type in model_types:
            # Create experiment name and directory
            result_name = f"{dataset_name}_{data_loading_option}_{model_type}"
            results_dir = os.path.join(main_dir, result_name)
            os.makedirs(results_dir, exist_ok=True)
            
            print(f"Running experiment: {result_name}")
            result, output = main_experiment(
                clients_num, 
                beta, 
                data_loading_option, 
                model_type, 
                cfg, 
                dataset_name=dataset_name, 
                hop=hop_value
            )
            
            # Save results
            filename = f'results_{result_name}.txt'
            filepath = os.path.join(results_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(str(result))
                
            print(f"Results saved to {filepath}\n")

if __name__ == "__main__":
    # Make sure Ray is shut down before starting
    if ray.is_initialized():
        ray.shutdown()
        
    parser = argparse.ArgumentParser(description='Run federated learning experiments')
    parser.add_argument('--data_options', nargs='+', default=['adjacency'], 
                        help='List of data loading options')
    parser.add_argument('--model_types', nargs='+', default=['GCN'], 
                        help='List of model types')
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset name')
    parser.add_argument('--clients', type=int, default=2,
                        help='Number of clients')
    parser.add_argument('--beta', type=float, default=10000,
                        help='Beta parameter for data distribution')
    parser.add_argument('--hop', type=int, default=1,
                        help='Number of hops for k-hop methods')
    
    args = parser.parse_args()
    
    run_gnn_experiments(
        args.data_options,
        args.model_types,
        args.dataset,
        args.clients,
        args.beta,
        args.hop
    )