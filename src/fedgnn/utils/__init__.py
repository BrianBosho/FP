"""Shared utilities for `fedgnn`."""

from .config import load_config
from .memory import (
    clear_cuda_cache,
    clear_memory_basic,
    clear_memory_aggressive,
    clear_memory_with_model,
    clear_memory_for_diffusion,
    clear_memory_for_adjacency,
    log_memory_usage,
    clear_memory_between_batches,
    clear_memory_on_error,
    memory_guard,
    get_memory_info,
)
from .project_paths import (
    ResolvedResultsPaths,
    find_repo_root,
    resolve_results_and_summary_dirs,
)
from .run import (
    setup_logging,
    log_training_results,
    log_evaluation_results,
    save_results_to_csv,
    compare_model_parameters,
    prepare_results_data,
    compute_experiment_statistics,
    generate_experiment_output,
    verify_model_inference_mode,
    compare_model_predictions,
    verify_test_mask,
)
from .wandb import (
    initialize_wandb,
    log_client_training_metrics,
    log_client_validation_metrics,
    log_final_validation_metrics,
    log_test_metrics,
    to_cpu_scalar,
)

__all__ = [
    # config
    "load_config",
    # memory
    "clear_cuda_cache",
    "clear_memory_basic",
    "clear_memory_aggressive",
    "clear_memory_with_model",
    "clear_memory_for_diffusion",
    "clear_memory_for_adjacency",
    "log_memory_usage",
    "clear_memory_between_batches",
    "clear_memory_on_error",
    "memory_guard",
    "get_memory_info",
    # project_paths
    "ResolvedResultsPaths",
    "find_repo_root",
    "resolve_results_and_summary_dirs",
    # run
    "setup_logging",
    "log_training_results",
    "log_evaluation_results",
    "save_results_to_csv",
    "compare_model_parameters",
    "prepare_results_data",
    "compute_experiment_statistics",
    "generate_experiment_output",
    "verify_model_inference_mode",
    "compare_model_predictions",
    "verify_test_mask",
    # wandb
    "initialize_wandb",
    "log_client_training_metrics",
    "log_client_validation_metrics",
    "log_final_validation_metrics",
    "log_test_metrics",
    "to_cpu_scalar",
]
