import torch
import ray
from src.client import FLClient
from src.models import GCN, GAT, GCN_arxiv, GraphSAGEProducts, PubmedGAT
from src.server import Server
import pandas as pd
from src.utils.utils import load_config
from src.utils.wandb_utils import initialize_wandb, log_client_training_metrics, log_test_metrics
from dotenv import load_dotenv
load_dotenv()
import wandb


from src.dataprocessing.loaders import (
    load_dataset,
    load_and_split,
    load_and_split_with_khop,
    load_and_split_with_feature_prop    
)
import numpy as np
from src.utils.run_utils import (
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

def load_configuration(config_path="/home/brian_bosho/FP/FP/federated-gnn/conf/base.yaml"):
    cfg = load_config(config_path)
    return cfg["num_clients"], cfg["beta"], cfg

def instantiate_model(model_type, num_features, num_classes, device, dataset_name="Cora", cfg=None):
    DEVICE = device
    model_params = {}
    if cfg is not None and "model_params" in cfg and model_type in cfg["model_params"]:
        model_params = cfg["model_params"][model_type]
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
        if dataset_name == "Pubmed":
            model = PubmedGAT(num_features, 8, num_classes, heads=8).to(DEVICE)
            print(f"Model is {model}")
            return model.to(DEVICE)
        else:
            hidden_dim = model_params.get("hidden_dim", 8)
            num_heads = model_params.get("num_heads", 8)
            dropout = model_params.get("dropout", 0.6)
            return GAT(num_features, hidden_dim, num_classes, heads=num_heads, dropout=dropout).to(DEVICE)
            
        # else:
            
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def initialize_clients(full_data, dataset, clients_data, model_type, cfg, device):
    """
    Initialize FL clients with their respective subgraphs.
    
    Args:
        full_data: Full graph (used for global evaluation, stays on CPU)
        dataset: Dataset object
        clients_data: List of subgraphs, one per client
        model_type: Type of model to use
        cfg: Configuration
        device: Device to use
    """
    DEVICE = device
    
    # Log what we're actually passing to clients (only in debug mode)
    debug = cfg.get("debug", False) if cfg else False
    if debug:
        print(f"\n=== CLIENT DATA LOADING ===")
        print(f"Full graph size: {full_data.num_nodes} nodes")
        print(f"Number of clients: {len(clients_data)}")
        for i, client_subgraph in enumerate(clients_data):
            print(f"Client {i} subgraph: {client_subgraph.num_nodes} nodes, {client_subgraph.x.shape[1]} features")
        print(f"===========================\n")
    
    # IMPORTANT: Pass the SUBGRAPH to each client, not the full graph
    return [FLClient.remote(client_subgraph, dataset, i, cfg, device, model_type) 
            for i, client_subgraph in enumerate(clients_data)]

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

    kh_options = ["page_rank", "random_walk", "diffusion", "efficient", "adjacency", "propagation", "zero", "propagation", "full", "chebyshev_diffusion", "chebyshev-diffusion", "chebyshev_diffusion_operator", "chebyshev-diffusion-operator"]
    if data_loading_option == "full_dataset":
        return load_dataset(dataset_name, device)
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
            config=config        )
 

def run_with_server(dataset_name, num_clients, beta, data_loading_option, model_type, cfg, device, hop = 1, fulltraining_flag = False):
    DEVICE = device
    
    # Get debug flag from config
    debug = cfg.get("debug", False)
    
    if debug:
        print(f"data_loading_option: {data_loading_option}")

    import time
    _t_partition_start = time.time()
    data, dataset, clients_data, test_data = load_data(
        data_loading_option, 
        num_clients, 
        beta, 
        dataset_name, 
        device=DEVICE, 
        hop=hop, 
        fulltraining_flag=fulltraining_flag,
        config=cfg,
    )
    _partition_secs = time.time() - _t_partition_start
    if debug:
        print(f"partition_time_s: {round(_partition_secs, 2)}")
        # Debug: device and feature dims to ensure tensors are on the expected device
        try:
            print("client[0] device:", clients_data[0].x.device, clients_data[0].edge_index.device)
            print("client[0] feature_dim:", clients_data[0].x.size(1))
            print("config.use_pe:", cfg.get("use_pe"), "num_iterations:", cfg.get("num_iterations"), "feature_prop_device:", cfg.get("feature_prop_device"))
            print("data_loading_option:", data_loading_option, "hop:", hop, "num_clients:", num_clients)
        except Exception as _dbg_e:
            print("debug_print_error:", _dbg_e)
    test_data = clients_data
    
    if debug:
        print("\n=== DATA LOADING SUMMARY ===")
        print(f"Full graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
        print(f"Number of client subgraphs: {len(clients_data)}")
        print(f"First client subgraph shape: {clients_data[0].x.shape}")
        print(f"Dataset: {dataset}")
        print(f"Device: {DEVICE}")
        
        # IMPORTANT: Keep full graph on CPU for now - only used for global testing
        # Each client already receives their subgraph
        print(f"\nFull graph device (before): {data.x.device}")
        # NOTE: Commenting this out to save GPU memory - full graph only needed for global testing
        # data = data.to(DEVICE)
        print(f"Full graph staying on CPU to save GPU memory")
        print("===========================\n")
    input_dim = clients_data[0].x.size(1)
    
    rounds = cfg['num_rounds']
    model = instantiate_model(model_type, input_dim, dataset.num_classes, DEVICE, dataset_name, cfg)
    clients = initialize_clients(data, dataset, clients_data, model_type, cfg, DEVICE)
    server = Server(clients, model, device, cfg)

    try:
        train_results = []
        best_eval_acc = 0
        patience = 0
        patience_threshold = 10
        best_eval_loss = float('inf')
        
        _t_train_start = time.time()
        for i in range(rounds):
            results = server.train_clients(i)
            train_results.append(results[0])
            avg_eval_acc = results[1]
            avg_eval_loss = results[2]
            if avg_eval_acc > best_eval_acc:
                best_eval_acc = avg_eval_acc
                patience = 0
            elif avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                patience = 0
            else:
                patience += 1
            if patience >= patience_threshold:
                print(f"Early stopping triggered at round {i}")
                break
        _train_secs = time.time() - _t_train_start
        if debug:
            print(f"training_time_s: {round(_train_secs, 2)}")
        log_training_results(train_results, debug=debug)
        
        eval_results = server.evaluate_clients()
        log_evaluation_results(eval_results, debug=debug)
        
        training_results = ray.get([client.get_loss_acc.remote() for client in server.clients])
        save_results_to_csv(training_results)

        are_params_identical = compare_model_parameters(server.model, server.clients, debug=debug)
        if debug:
            print(f"\nAll model parameters are identical: {are_params_identical}")

        # Evaluate: ensure consistency for single-client runs by testing both on the same global graph
        if len(server.clients) == 1:
            test_results = server.test_global_model(data)
            client_test_results = ray.get([client.test.remote(data) for client in server.clients])
        else:
            test_results = server.test_global_model(data)
            client_test_results = ray.get([client.test.remote(test) for client, test in zip(server.clients, test_data)])
        
        # Debug: verify masks and cross-accuracy when single client
        if debug:
            try:
                if len(server.clients) == 1:
                    print("\n=== DEBUG: Single-client cross-check ===")
                    try:
                        print("Global test nodes:", int(data.test_mask.sum()))
                        print("Client0 test nodes:", int(test_data[0].test_mask.sum()))
                    except Exception as _dbg_e0:
                        print("debug_mask_error:", _dbg_e0)
                    try:
                        server_on_client = server.test_global_model(test_data[0])
                        client_on_global = ray.get([server.clients[0].test.remote(data)])[0]
                        print(f"Server model on client graph acc: {server_on_client}")
                        print(f"Client model on global graph acc: {client_on_global}")
                    except Exception as _dbg_e1:
                        print("debug_cross_eval_error:", _dbg_e1)
                    print("=== END DEBUG ===\n")
            except Exception as _dbg_e2:
                print("debug_section_error:", _dbg_e2)

        # Log test results to wandb - use proper step value instead of -1
        final_round = rounds  # Use the total number of rounds as the step
        log_test_metrics(test_results, client_test_results, current_global_epoch=final_round)
        
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
    # DEVICE = torch.device("cpu")
    test_results = []
    client_test_results = []
    
    # Get debug flag from config
    debug = cfg.get("debug", False)
    
    if debug:
        print(f"DEVICE: {DEVICE}")
    repetitions = cfg.get("repetitions", 1),
    num_iterations = cfg.get("num_iterations", 50)
    
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
            "fulltraining_flag": fulltraining_flag,
            "use_pe": cfg.get("use_pe"),
        },
        "rounds": []
    }
    
    if debug:
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
        
        for i in range(cfg.get("repetitions", 5)):  # Change 1 to the desired number of repetitions
            try:
                # Clear CUDA cache before each iteration
                torch.cuda.empty_cache()
                gc.collect()
                
                # Force synchronization for memory-intensive operations
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                
                run_config = {
                    "dataset": dataset_name,
                    "model": model_type,
                    "data_loading": data_loading_option,
                    "num_clients": cfg.get("num_clients"),
                    "num_rounds": cfg.get("num_rounds"),
                    "epochs": cfg.get("epochs"),
                    "beta": cfg.get("beta"),
                    "lr": cfg.get("lr"),
                    "optimizer": cfg.get("optimizer"),
                    "decay": cfg.get("decay"),
                    "hop": cfg.get("hop"),
                    "fulltraining_flag": cfg.get("fulltraining_flag"),
                    "use_pe": cfg.get("use_pe"),
                    "pe_r": cfg.get("pe_r"),
                    "pe_P": cfg.get("pe_P"),
                    "normalize": cfg.get("normalize"),
                    "run_index": i+1,
                    "num_iterations": cfg.get("num_iterations")
                }
                # Get wandb configuration from cfg
                use_wandb = cfg.get("use_wandb", True)  # Default to True for backward compatibility
                wandb_project = cfg.get("wandb_project", "FGL3")
                wandb_entity = cfg.get("wandb_entity", None)
                wandb_mode = cfg.get("wandb_mode", "online")
                
                initialize_wandb(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=run_config,
                    group=f"{model_type}_{dataset_name}_{data_loading_option}",
                    mode=wandb_mode,
                    use_wandb=use_wandb
                )
                current_cfg = cfg.copy()
                
                # Only process wandb config if wandb is enabled and initialized
                if use_wandb and wandb.run is not None:
                    print(f"Wandb config is set to: {wandb.config}")
                    print(wandb.config)
                    for key in wandb.config:
                        current_cfg[key] = wandb.config[key]
                    run_name = f"{model_type}_{dataset_name}_{current_cfg.get('optimizer')}_lr{current_cfg.get('lr')}_decay{current_cfg.get('decay')}_beta{current_cfg.get('beta')}_run{i+1}"
                    wandb.run.name = run_name

                # Clear memory before each experiment
                from src.utils.memory_utils import clear_memory_aggressive, log_memory_usage
                log_memory_usage("before experiment")
                clear_memory_aggressive()
                
                global_results, client_results = run_with_server(
                    dataset_name, 
                    clients_num, 
                    beta, 
                    data_loading_option, 
                    model_type, 
                    current_cfg,  # Use current_cfg with sweep values
                    DEVICE, 
                    hop=hop, 
                    fulltraining_flag=fulltraining_flag,
                
                )
                
                # Clear memory after experiment
                log_memory_usage("after experiment")
                clear_memory_aggressive()
                
                # Check if results are valid
                if global_results is None or client_results is None:
                    print(f"Warning: Round {i+1} returned None results, skipping...")
                    continue
                
                test_results.append(global_results)
                client_test_results.append(client_results)
                if debug:
                    print(f"Round {i+1} is complete")
                
                results_data["rounds"].append({
                    "round": i+1,
                    "global_result": float(global_results),
                    "client_result": float(client_results)                })
                
                # Log individual run results before finishing (only if wandb is enabled)
                if use_wandb and wandb.run is not None:
                    wandb.log({
                        "run_global_test_result": global_results,
                        "run_client_test_result": client_results                })
                
                if use_wandb:
                    wandb.finish()
            except Exception as e:
                print(f"Error in round {i+1}: {e}")
                import traceback
                traceback.print_exc()
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

        # Note: Final summary logging moved to before wandb.finish() in the loop above
        # This prevents logging after the wandb run has already been finished
        
        


        results_data["summary"] = {
            "global_results": [float(x) for x in test_results],
            "client_results": [float(x) for x in client_test_results],
            "average_global_result": float(average_global_results),
            "average_client_result": float(average_client_results),
            "std_global": float(std_global),
            "std_client": float(std_client)        }

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
