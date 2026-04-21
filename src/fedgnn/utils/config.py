from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration with base.yaml merging support.

    Args:
        config_path: Path to the specific config file

    Returns:
        Merged configuration (base.yaml + specific config)
    """
    import os

    # Get the absolute path to the config file
    config_path_abs = os.path.abspath(config_path)

    # Find the conf directory by looking for it in the path
    # Assumes base.yaml is always in the top-level conf directory
    if 'conf' in config_path_abs:
        # Get everything up to and including 'conf'
        conf_index = config_path_abs.find('conf')
        conf_dir = config_path_abs[:conf_index + 4]  # +4 for 'conf'
    else:
        # Fallback to same directory as config file
        conf_dir = os.path.dirname(config_path_abs)

    base_config_path = os.path.join(conf_dir, "base.yaml")

    # Load base configuration
    base_config = OmegaConf.load(base_config_path)

    # Load specific configuration
    specific_config = OmegaConf.load(config_path)

    # Merge configurations (specific config overrides base)
    merged_config = OmegaConf.merge(base_config, specific_config)

    return merged_config
