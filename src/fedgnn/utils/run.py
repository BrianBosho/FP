"""
Runtime helpers (logging/results) (Phase C compatibility layer).
"""

from src.utils.run_utils import (  # noqa: F401
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

__all__ = [
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
]

