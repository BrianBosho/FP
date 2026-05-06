#!/usr/bin/env python3

from src.fedgnn.fl.run import ensure_ray_initialized, load_configuration, main_experiment
import os
import ray
import argparse
import json
import yaml
import torch
from tabulate import tabulate
from datetime import datetime
import time
import shutil
from glob import glob
from src.fedgnn.utils.config import load_config
from src.fedgnn.utils.project_paths import resolve_results_and_summary_dirs
import wandb

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run federated GNN experiments and print results')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    
    # Keep command-line arguments as fallback options
    parser.add_argument('--clients', type=int, nargs='+', help='Number of clients (multiple values for ablation)')
    parser.add_argument('--rounds', type=int, help='Number of communication rounds')
    parser.add_argument('--epochs', type=int, help='Number of local epochs per round')
    parser.add_argument('--beta', type=float, nargs='+', help='Beta parameter(s) for Dirichlet distribution')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--datasets', nargs='+', help='List of datasets to run experiments on')
    parser.add_argument('--data_loading', nargs='+', help='Data loading options')
    parser.add_argument('--models', nargs='+', help='Model types to use')
    parser.add_argument('--hop', type=int, help='Number of hops for graph propagation')
    parser.add_argument('--save_results', action='store_true', help='Save detailed results to files')
    parser.add_argument('--results_dir', type=str, help='Directory to save results')
    parser.add_argument('--fulltraining_flag', action='store_true', help='Use full training flag')
    parser.add_argument('--optimizer', type=str, help='Optimizer to use')
    parser.add_argument('--decay', type=float, help='Weight decay')
    parser.add_argument('--repetitions', type=int, help='Number of repetitions for each experiment')
    parser.add_argument('--ray_port', type=int, help='Ray port for distributed execution')
    parser.add_argument('--use_pe', type=lambda x: x.lower() == 'true', help='Enable positional encoding (true/false)')
    return parser.parse_args()

def load_yaml_config(config_path):
    """Load experiment configuration from YAML file with base.yaml merging"""
    return load_config(config_path)

FEATURE_PROP_DATA_LOADING_OPTIONS = {
    "page_rank",
    "random_walk",
    "diffusion",
    "difussion",
    "efficient",
    "adjacency",
    "propagation",
    "chebyshev_diffusion",
    "chebyshev-diffusion",
    "chebyshev_diffusion_operator",
    "chebyshev-diffusion-operator",
    # Propagator-eval operators (V2)
    "appnp",
    "asymmetric_random_walk",
    "heat_kernel_exact",
}


def data_loading_uses_pe(data_loading_option) -> bool:
    """Return whether a data-loading mode can actually apply positional encodings."""
    return str(data_loading_option).lower() in FEATURE_PROP_DATA_LOADING_OPTIONS


def setup_environment_for_experiment(dataset_name, data_loading_option, model_type, beta_value, clients_num, results_dir, timestamp, hop=1, use_pe=False, num_iterations=None, diffusion_t=None, alpha=None):
    """Setup environment variables to redirect CSV output to experiment directory"""
    # Create experiment directory path with full config to avoid collisions
    experiment_name = f"{dataset_name}_{data_loading_option}_{model_type}_beta{beta_value}_clients{clients_num}_hop{hop}"
    
    # Add sweep parameters to name if they are part of an ablation
    if num_iterations is not None:
        experiment_name += f"_iter{num_iterations}"
    if diffusion_t is not None:
        experiment_name += f"_t{diffusion_t}"
    if alpha is not None:
        experiment_name += f"_alpha{alpha}"
        
    # Add PE indicator to experiment name when enabled
    if use_pe:
        experiment_name += "_pe"


    exp_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Set environment variables that will be checked in run_utils.py
    os.environ["EXPERIMENT_RESULTS_DIR"] = exp_dir
    os.environ["EXPERIMENT_TIMESTAMP"] = timestamp

    return exp_dir, experiment_name


def find_completed_result(exp_dir, repetitions):
    """Return the newest complete result JSON for a condition, if one exists."""
    candidates = sorted(glob(os.path.join(exp_dir, "results_*.json")), key=os.path.getmtime, reverse=True)
    for path in candidates:
        try:
            with open(path) as f:
                result = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        summary = result.get("summary")
        rounds = result.get("rounds")
        if not isinstance(summary, dict) or not isinstance(rounds, list):
            continue

        try:
            expected_reps = int(repetitions)
        except (TypeError, ValueError):
            expected_reps = 1

        if len(rounds) >= expected_reps:
            return path, result

    return None, None


def build_summary_row(result, dataset_name, data_loading_option, model_type, beta, clients_num,
                      hop, effective_use_pe, num_iterations, diffusion_t, alpha):
    summary = result["summary"]
    duration = result.get("duration", {})
    duration_formatted = duration.get("formatted", "00:00:00")
    return [
        dataset_name,
        data_loading_option,
        model_type,
        f"{float(beta):.2f}",
        f"{int(clients_num)}",
        f"{summary['average_global_result']:.4f}",
        f"{summary['average_client_result']:.4f}",
        f"{summary['std_client']:.4f}",
        f"{summary['std_global']:.4f}",
        duration_formatted,
        hop,
        effective_use_pe,
        num_iterations,
        diffusion_t,
        alpha,
    ]


def build_all_results_entry(result, dataset_name, data_loading_option, model_type, beta, clients_num,
                            hop, effective_use_pe, requested_use_pe, num_iterations, diffusion_t, alpha):
    summary = result["summary"]
    return {
        "dataset": dataset_name,
        "data_loading": data_loading_option,
        "model": model_type,
        "beta": beta,
        "clients": clients_num,
        "avg_global": summary["average_global_result"],
        "avg_client": summary["average_client_result"],
        "client_std_dev": summary["std_client"],
        "global_std_dev": summary["std_global"],
        "duration": result.get("duration", {"seconds": None, "formatted": "00:00:00"}),
        "hop": hop,
        "use_pe": effective_use_pe,
        "requested_use_pe": requested_use_pe,
        "num_iterations": num_iterations,
        "diffusion_t": diffusion_t,
        "alpha": alpha,
    }

def copy_training_csv_to_experiment_dir(exp_dir, experiment_name, timestamp):
    """Copy training CSV file to experiment directory and rename it"""
    source_file = "results.csv"
    if os.path.exists(source_file):
        target_file = os.path.join(exp_dir, f"training_{experiment_name}_{timestamp}.csv")
        shutil.copy2(source_file, target_file)
        print(f"Training CSV results saved to {target_file}")

def format_time(seconds):
    """Format time in seconds to a human-readable string"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def save_summary_results(summary_rows, all_results, results_dir, summary_dir, config):
    """Save summary results to a file in the parent results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create summary TXT file in summary directory
    summary_txt_path = os.path.join(summary_dir, f"summary_results_{timestamp}.txt")
    
    with open(summary_txt_path, 'w') as f:
        # Write experiment configuration
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"- Clients: {config['num_clients']}\n")
        f.write(f"- Rounds: {config['num_rounds']}\n")
        f.write(f"- Epochs: {config['epochs']}\n")
        f.write(f"- Learning Rate: {config['lr']}\n")
        f.write(f"- Beta values: {config['beta']}\n")
        f.write(f"- Datasets: {config['datasets']}\n")
        f.write(f"- Data Loading Options: {config['data_loading']}\n")
        f.write(f"- Model Types: {config['models']}\n")
        f.write(f"- Results Directory: {config['results_dir']}\n")
        f.write(f"- Save Detailed Results: {config['save_results']}\n")
        f.write(f"- Hop: {config['hop']}\n")
        f.write(f"- Use PE values: {config.get('use_pe')}\n")
        f.write(f"- Full Training Flag: {config['fulltraining_flag']}\n\n")
        
        # Write summary table
        f.write("\n" + "=" * 80 + "\n")
        f.write("EXPERIMENT RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        headers = [
            "Dataset",
            "Data Loading",
            "Model",
            "Beta",
            "Clients",
            "Avg Global Result",
            "Avg Client Result",
            "client_std_dev",
            "global_std_dev",
            "Duration",
            "Hop",
            "Use PE",
        ]
        f.write(tabulate(summary_rows, headers=headers, tablefmt="grid") + "\n\n")
        
        # Write key results
        f.write("KEY RESULTS:\n")
        f.write("-" * 80 + "\n")
        for row in summary_rows:
            f.write(f"{row[0]} with {row[1]} using {row[2]} (beta={row[3]}, clients={row[4]}, hop={row[10]}, use_pe={row[11]}):\n")
            f.write(f"  - Average Global Result: {row[5]}\n")
            f.write(f"  - Average Client Result: {row[6]}\n")
            f.write(f"  - Client StdDev: {row[7]}\n")
            f.write(f"  - Global StdDev: {row[8]}\n")
            f.write(f"  - Experiment Duration: {row[9]}\n")
            f.write("-" * 80 + "\n")
    
    # Create summary JSON file in summary directory
    summary_json_path = os.path.join(summary_dir, f"summary_results_{timestamp}.json")
    
    # Create a structured JSON with all results
    # Convert DictConfig to regular dict for JSON serialization
    from omegaconf import OmegaConf
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    summary_json = {
        "timestamp": timestamp,
        "configuration": config_dict,
        "results": [
            {
                "dataset": result["dataset"],
                "data_loading": result["data_loading"],
                "model": result["model"],
                "beta": result["beta"],
                "clients": result["clients"],
                "avg_global": result["avg_global"],
                "avg_client": result["avg_client"],
                "client_std_dev": result["client_std_dev"],
                "global_std_dev": result["global_std_dev"],
                "duration": result["duration"],
                "hop": result.get("hop"),
                "use_pe": result.get("use_pe"),
            }
            for result in all_results
        ]
    }
    
    with open(summary_json_path, 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"\nSummary results saved to:")
    print(f"- Text file: {summary_txt_path}")
    print(f"- JSON file: {summary_json_path}")
    print(f"- Detailed results saved to: {results_dir}")
    
    return summary_txt_path, summary_json_path

def run_experiments(args):
    # Start with a minimal default configuration
    cfg = {
        "num_clients": [10],
        "beta": [1.0],
        "datasets": ["Cora"],
        "data_loading": ["full"],
        "models": ["GCN"],
        "num_rounds": 10,
        "epochs": 3,
        "lr": 0.5,
        "optimizer": "SGD",
        "decay": 0.0005,
        # Default is repo-local to avoid polluting parent directories.
        # Users can still override via YAML `results_dir` or CLI `--results_dir`.
        "results_dir": "runs/experiments",
        "save_results": False,
        "hop": 1,
        "fulltraining_flag": False,
        "use_pe": [False],
    }
    
    # If a YAML config is provided, use it as the base configuration
    if args.config:
        yaml_config = load_yaml_config(args.config)
        
        # Replace the default config with the YAML config
        cfg = yaml_config
        
        # Ensure iteration parameters are lists (check for both list and ListConfig)
        from omegaconf import ListConfig
        for param in ["num_clients", "beta", "datasets", "data_loading", "models", "use_pe"]:
            if param in cfg and not isinstance(cfg[param], (list, ListConfig)):
                # lets print param and cfg[param]
                print(f"param: {param}, cfg[param]: {cfg[param]}")
                cfg[param] = [cfg[param]]
    
    # Override with command-line arguments if provided
    if args.clients is not None:
        cfg["num_clients"] = args.clients
    if args.rounds is not None:
        cfg["num_rounds"] = args.rounds
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.beta is not None:
        cfg["beta"] = args.beta
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.optimizer is not None:
        cfg["optimizer"] = args.optimizer
    if args.decay is not None:
        cfg["decay"] = args.decay
    if args.datasets is not None:
        cfg["datasets"] = args.datasets
    if args.data_loading is not None:
        cfg["data_loading"] = args.data_loading
    if args.models is not None:
        cfg["models"] = args.models
    if args.hop is not None:
        cfg["hop"] = args.hop
    if args.save_results:
        cfg["save_results"] = args.save_results
    if args.results_dir is not None:
        cfg["results_dir"] = args.results_dir
    if args.fulltraining_flag:
        cfg["fulltraining_flag"] = args.fulltraining_flag
    if args.repetitions is not None:
        cfg["repetitions"] = args.repetitions
    if args.use_pe is not None:
        cfg["use_pe"] = [args.use_pe]



    # Handle wandb configuration (only if wandb is enabled and running)
    use_wandb = cfg.get("use_wandb", True)  # Default to True for backward compatibility
    if use_wandb and wandb.run is not None:
        for key in wandb.config:
            cfg[key] = wandb.config[key]
    
    # Extract values for iteration and convert ListConfig to regular Python types
    from omegaconf import OmegaConf

    def as_list(value):
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    client_nums = as_list(cfg["num_clients"])
    beta_values = as_list(cfg["beta"])
    datasets = as_list(cfg["datasets"])
    data_loading_options = as_list(cfg["data_loading"])
    model_types = as_list(cfg["models"])
    num_iterations_values = as_list(cfg.get("num_iterations", [80]))
    diffusion_t_values = as_list(cfg.get("diffusion_t", [1.0]))
    alpha_values = as_list(cfg.get("alpha", [0.5]))
    # Resolve results_dir and summary_dir deterministically
    resolved_paths = resolve_results_and_summary_dirs(cfg.get("results_dir"))
    results_dir = str(resolved_paths.results_dir)
    cfg["results_dir"] = results_dir
    save_results = cfg["save_results"]
    resume_completed = bool(cfg.get("resume_completed", False))
    hop_values = as_list(cfg.get("hop", 1))
    fulltraining_flag = cfg["fulltraining_flag"]
    use_pe_values = as_list(cfg.get("use_pe", [False]))  # Default to no PE if not specified
    
    # Ensure Ray is shut down before starting
    try:
        ray.shutdown()
    except:
        pass
    
    # Create main results directory
    os.makedirs(results_dir, exist_ok=True)

    # Create summary results directory
    summary_dir = str(resolved_paths.summary_dir)
    os.makedirs(summary_dir, exist_ok=True)
    
    # Store all experiment results
    all_results = []
    
    # Create a summary table for the final output
    summary_rows = []
    
    # Print experiment configuration
    print("\nExperiment Configuration:")
    print(f"- Clients: {client_nums}")
    print(f"- Rounds: {cfg['num_rounds']}")
    print(f"- Epochs: {cfg['epochs']}")
    print(f"- Learning Rate: {cfg['lr']}")
    print(f"- Beta values: {beta_values}")
    print(f"- Datasets: {datasets}")
    print(f"- Data Loading Options: {data_loading_options}")
    print(f"- Model Types: {model_types}")
    print(f"- Results Directory: {results_dir}")
    print(f"- Save Detailed Results: {save_results}")
    print(f"- Resume Completed: {resume_completed}")
    print(f"- Hop: {hop_values}")
    print(f"- Full Training Flag: {fulltraining_flag}")
    
    device_cfg = str(cfg.get("device", "cuda")).lower()
    using_cuda_device = (
        device_cfg == "cuda" or
        (device_cfg == "auto" and torch.cuda.is_available())
    )
    ensure_ray_initialized(cfg, using_cuda_device)

    try:
        # Run experiments for each combination
        for use_pe in use_pe_values:
            for dataset_name in datasets:
                for data_loading_option in data_loading_options:
                    for model_type in model_types:
                        for beta in beta_values:
                            for hop in hop_values:
                                for clients_num in client_nums:
                                    for num_iterations in num_iterations_values:
                                        for diffusion_t in diffusion_t_values:
                                            for alpha in alpha_values:
                                                # Generate timestamp for this experiment
                                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                                effective_use_pe = bool(use_pe) and data_loading_uses_pe(data_loading_option)

                                                # Setup environment for this experiment
                                                exp_dir, experiment_name = setup_environment_for_experiment(
                                                    dataset_name, data_loading_option, model_type, beta, clients_num,
                                                    results_dir, timestamp, hop=hop, use_pe=effective_use_pe,
                                                    num_iterations=num_iterations, diffusion_t=diffusion_t, alpha=alpha
                                                )

                                                if resume_completed:
                                                    completed_path, completed_result = find_completed_result(
                                                        exp_dir, cfg.get("repetitions", 1)
                                                    )
                                                    if completed_result is not None:
                                                        print(f"\n{'='*80}")
                                                        print(f"Skipping completed experiment: {experiment_name}")
                                                        print(f"Loaded existing result: {completed_path}")
                                                        print(f"{'='*80}")
                                                        summary_rows.append(build_summary_row(
                                                            completed_result,
                                                            dataset_name,
                                                            data_loading_option,
                                                            model_type,
                                                            beta,
                                                            clients_num,
                                                            hop,
                                                            effective_use_pe,
                                                            num_iterations,
                                                            diffusion_t,
                                                            alpha,
                                                        ))
                                                        all_results.append(build_all_results_entry(
                                                            completed_result,
                                                            dataset_name,
                                                            data_loading_option,
                                                            model_type,
                                                            beta,
                                                            clients_num,
                                                            hop,
                                                            effective_use_pe,
                                                            use_pe,
                                                            num_iterations,
                                                            diffusion_t,
                                                            alpha,
                                                        ))
                                                        continue

                                                # Create a monkey patch for save_results_to_csv in run_utils
                                                from src.fedgnn.utils.run import save_results_to_csv as original_save_func

                                                def patched_save_func(results, filename=None):
                                                    # Use the original function but with our custom filename
                                                    csv_filename = os.path.join(exp_dir, f"training_{experiment_name}_{timestamp}.csv")
                                                    return original_save_func(results, csv_filename)

                                                # Apply the monkey patch to the imported function
                                                import src.fedgnn.utils.run as run_utils
                                                run_utils.save_results_to_csv = patched_save_func

                                                # Print experiment header
                                                print(f"\n{'='*80}")
                                                print(f"Running experiment: {experiment_name}")
                                                print(f"{'='*80}")

                                                # Set Ray port if specified
                                                if hasattr(args, 'ray_port') and args.ray_port:
                                                    # Ray will use the address directly, no need for env var
                                                    pass

                                                # Create a training configuration for the experiment
                                                training_cfg = cfg.copy()  # Pass all parameters through
                                                training_cfg["beta"] = beta
                                                training_cfg["hop"] = hop
                                                # also overwrite dataset_name, data_loading_option, model_type, clients_num
                                                training_cfg["dataset_name"] = dataset_name
                                                training_cfg["data_loading_option"] = data_loading_option
                                                training_cfg["model_type"] = model_type
                                                training_cfg["clients_num"] = clients_num
                                                training_cfg["use_pe"] = effective_use_pe
                                                training_cfg["requested_use_pe"] = use_pe
                                                training_cfg["repetitions"] = cfg.get("repetitions", 1)  # Default to 1 if not specified

                                                # Add current sweep parameters to training_cfg
                                                training_cfg["num_iterations"] = num_iterations
                                                training_cfg["diffusion_t"] = diffusion_t
                                                training_cfg["alpha"] = alpha

                                                # Start time measurement
                                                start_time = time.time()

                                                # Run the experiment
                                                result, output = main_experiment(
                                                    clients_num,
                                                    beta,
                                                    data_loading_option,
                                                    model_type,
                                                    training_cfg,
                                                    dataset_name=dataset_name,
                                                    hop=hop,
                                                    fulltraining_flag=fulltraining_flag,
                                                    manage_ray_lifecycle=False,
                                                )

                                                # Calculate experiment duration
                                                end_time = time.time()
                                                duration = end_time - start_time
                                                duration_formatted = format_time(duration)

                                                # Clean up driver-side references between conditions while
                                                # keeping the shared Ray runtime alive across the sweep.
                                                print("Cleaning up driver memory...")
                                                import gc
                                                gc.collect()

                                                device_cfg = str(training_cfg.get("device", cfg.get("device", "cuda"))).lower()
                                                cleanup_cuda = device_cfg != "cpu" and torch.cuda.is_available()
                                                if cleanup_cuda:
                                                    try:
                                                        torch.cuda.empty_cache()
                                                        torch.cuda.synchronize()
                                                        # Reset CUDA memory stats to clear any lingering state from torch_sparse
                                                        torch.cuda.reset_peak_memory_stats()
                                                        torch.cuda.reset_accumulated_memory_stats()
                                                    except RuntimeError as e:
                                                        print(f"[run_experiments] WARNING: cuda.synchronize failed ({e}), continuing")
                                                        torch.cuda.empty_cache()

                                                # Final GC round after CUDA cleanup
                                                gc.collect()

                                                # Add duration to the result
                                                result["duration"] = {
                                                    "seconds": duration,
                                                    "formatted": duration_formatted
                                                }

                                                # Add to summary table - now including sweep parameters
                                                summary_rows.append(build_summary_row(
                                                    result,
                                                    dataset_name,
                                                    data_loading_option,
                                                    model_type,
                                                    beta,
                                                    clients_num,
                                                    hop,
                                                    effective_use_pe,
                                                    num_iterations,
                                                    diffusion_t,
                                                    alpha,
                                                ))

                                                # Save results if requested
                                                if save_results:
                                                    # Save detailed results
                                                    filename = f"results_{experiment_name}_{timestamp}.json"
                                                    filepath = os.path.join(exp_dir, filename)

                                                    with open(filepath, 'w') as f:
                                                        json.dump(result, f, indent=2)

                                                    # Also save readable output with duration information
                                                    txt_filepath = os.path.join(exp_dir, f"results_{experiment_name}_{timestamp}.txt")
                                                    with open(txt_filepath, 'w') as f:
                                                        f.write(output)
                                                        f.write(f"\n\nExperiment Duration: {duration_formatted} (HH:MM:SS)\n")

                                                    print(f"Results saved to {filepath}")

                                                # Make sure the training CSV was copied to experiment directory
                                                copy_training_csv_to_experiment_dir(exp_dir, experiment_name, timestamp)

                                                # Store results for final summary
                                                all_results.append(build_all_results_entry(
                                                    result,
                                                    dataset_name,
                                                    data_loading_option,
                                                    model_type,
                                                    beta,
                                                    clients_num,
                                                    hop,
                                                    effective_use_pe,
                                                    use_pe,
                                                    num_iterations,
                                                    diffusion_t,
                                                    alpha,
                                                ))
    finally:
        if ray.is_initialized():
            ray.shutdown()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass

    # Save summary results to the summary directory
    save_summary_results(summary_rows, all_results, results_dir, summary_dir, cfg)
    
    return summary_rows, all_results

def print_summary(summary_rows):
    # Print summary table
    print("\n\n")
    print("=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    
    headers = [
        "Dataset",
        "Data Loading",
        "Model",
        "Beta",
        "Clients",
        "Avg Global Result",
        "Avg Client Result",
        "client_std_dev",
        "global_std_dev",
        "Duration",
        "Hop",
        "Use PE",
        "Iter",
        "t",
        "alpha",
    ]
    print(tabulate(summary_rows, headers=headers, tablefmt="grid"))
    
    print("\n")
    
    # Print the key results separately in a clear, easy-to-read format
    print("KEY RESULTS:")
    print("-" * 80)
    for row in summary_rows:
        print(f"{row[0]} with {row[1]} using {row[2]} (beta={row[3]}, clients={row[4]}, hop={row[10]}, use_pe={row[11]}, iter={row[12]}, t={row[13]}, alpha={row[14]}):")
        print(f"  - Average Global Result: {row[5]}")
        print(f"  - Average Client Result: {row[6]}")
        print(f"  - Client StdDev: {row[7]}")
        print(f"  - Global StdDev: {row[8]}")
        print(f"  - Experiment Duration: {row[9]}")
        print("-" * 80)

def create_example_config(output_path="experiment_config_example.yaml"):
    """Create an example YAML configuration file"""
    example_config = {
        "num_clients": [5, 10, 20],  # Multiple client counts for ablation studies
        "num_rounds": 10,
        "epochs": 3,
        "beta": [0.1, 0.5, 1.0, 5.0],  # Multiple beta values for ablation studies
        "lr": 0.5,
        "datasets": ["Cora", "Citeseer"],
        "data_loading": ["full", "adjacency", "zero_hop"],
        "models": ["GCN"],
        "results_dir": "runs/yaml_experiment_results",
        "save_results": True,
        "hop": 1,
        "fulltraining_flag": False
    }
    
    with open(output_path, 'w') as file:
        yaml.dump(example_config, file, default_flow_style=False)
    
    print(f"Example configuration file created at: {output_path}")

if __name__ == "__main__":
    args = parse_arguments()
    
    # Generate example config file if requested
    if args.config == "generate_example":
        create_example_config()
        exit(0)


    summary_rows, all_results = run_experiments(args)
    print_summary(summary_rows)
    
    # Only finish wandb if it was enabled and running
    if wandb.run is not None:
        wandb.finish()
