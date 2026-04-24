from pathlib import Path

from omegaconf import OmegaConf, DictConfig

from src.fedgnn.utils.project_paths import find_repo_root


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

    return _normalize_training_config(merged_config)
