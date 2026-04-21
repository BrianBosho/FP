"""Legacy compatibility wrapper — real implementation is in `src.fedgnn.utils.wandb`."""

from src.fedgnn.utils.wandb import (  # noqa: F401
    initialize_wandb,
    log_client_training_metrics,
    log_client_validation_metrics,
    log_final_validation_metrics,
    log_test_metrics,
    to_cpu_scalar,
)

__all__ = [
    "initialize_wandb",
    "log_client_training_metrics",
    "log_client_validation_metrics",
    "log_final_validation_metrics",
    "log_test_metrics",
    "to_cpu_scalar",
]
