from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, DictConfig

from src.fedgnn.utils.project_paths import find_repo_root

# Keys present in base.yaml or used anywhere in the training path.
# Any key that appears in an experiment YAML but is *not* in this set
# is flagged as a potential misspelling (warn-only — never hard-errors).
_KNOWN_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    # Sweep dimensions
    "datasets", "dataset_name", "data_loading", "models", "model_type",
    "num_clients", "clients_num", "beta", "hop", "use_pe", "use_unified_model",
    # Training loop
    "num_rounds", "epochs", "lr", "optimizer", "decay", "repetitions",
    "early_stopping_patience", "log_per_round", "log_global_test_per_round",
    "fulltraining_flag", "experiment_seed", "partition_seed",
    # Aggregation / BN
    "aggregation", "bn_fl_strategy",
    # Data / features
    "num_iterations", "feature_prop_tolerance", "feature_prop_relative_tolerance",
    "log_feature_prop_energy", "feature_prop_device", "fp_max_concurrent",
    "prop_dtype", "feature_prop_init_strategy",
    "diffusion_t", "chebyshev_k", "chebyshev_t", "force_chebyshev_for_large_graphs",
    "alpha",
    # PE
    "pe_r", "pe_P", "normalize", "rfp_qr_max_nodes",
    # Device / GPU
    "device", "keep_data_on_gpu", "adaptive_device", "adaptive_time_threshold_sec",
    "use_amp", "grad_clip_norm", "use_minibatch", "auto_minibatch_if_large",
    "batch_size", "num_neighbors", "data_loading_device",
    # Ray
    "max_concurrent_clients", "client_num_gpus", "ray_num_gpus",
    "ray_object_store_memory_bytes", "ray_include_dashboard",
    "ray_object_spilling_threshold", "ray_max_io_workers",
    # Output / caching
    "results_dir", "save_results", "resume_completed", "use_shard_cache",
    "shard_cache_dir",
    # Evaluation
    "global_eval_uses_fp", "global_eval_pe_mode",
    # WandB
    "use_wandb", "wandb_project", "wandb_entity", "wandb_mode",
    # Misc
    "debug", "training", "model_architecture", "paths",
    # Scalar sweeps sometimes written directly
    "data_loading_option", "requested_use_pe",
})


def _normalize_training_config(cfg: DictConfig) -> DictConfig:
    """Expose nested training settings through legacy top-level keys.

    Experiment configs use a clearer nested block:

        training:
          lr: ...
          optimizer: ...
          weight_decay: ...
          epochs: ...
          patience: ...

    The training path still reads ``cfg["lr"]``, ``cfg["optimizer"]``,
    ``cfg["decay"]``, and ``cfg["epochs"]``.  Copy direct keys from the nested
    block into those legacy top-level fields so the locked experiment YAMLs are
    the values that actually run.
    """
    training = cfg.get("training")
    if training is None:
        return cfg

    # Base config has training.default; experiment configs have direct keys.
    if "lr" in training:
        cfg["lr"] = training["lr"]
    if "optimizer" in training:
        cfg["optimizer"] = training["optimizer"]
    if "weight_decay" in training:
        cfg["decay"] = training["weight_decay"]
    if "epochs" in training:
        cfg["epochs"] = training["epochs"]
    if "patience" in training:
        cfg["early_stopping_patience"] = training["patience"]

    return cfg


def validate_config_keys(cfg: DictConfig, source: str = "") -> list[str]:
    """Warn about top-level keys that look like misspellings or unknown options.

    Returns the list of unknown keys (never raises).  Prints a warning for each
    so the user can catch typos before a long run silently uses the wrong value.
    """
    unknown: list[str] = []
    for key in cfg:
        if key not in _KNOWN_TOP_LEVEL_KEYS:
            unknown.append(key)
    if unknown:
        loc = f" in {source}" if source else ""
        print(
            f"[config] WARNING: unknown top-level key(s){loc}: {unknown}. "
            "Check for misspellings — unknown keys are silently ignored."
        )
    return unknown


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration with base.yaml merging support.

    Args:
        config_path: Path to the specific config file

    Returns:
        Merged configuration (base.yaml + specific config)
    """
    config_path_abs = Path(config_path).resolve()
    repo_root = find_repo_root(config_path_abs)
    base_config_path = repo_root / "conf" / "base.yaml"

    # Load base configuration
    base_config = OmegaConf.load(str(base_config_path))

    # Load specific configuration
    specific_config = OmegaConf.load(str(config_path_abs))

    # Merge configurations (specific config overrides base)
    merged_config = OmegaConf.merge(base_config, specific_config)

    # Warn about unknown keys from the experiment-specific YAML
    validate_config_keys(specific_config, source=str(config_path_abs))

    return _normalize_training_config(merged_config)
