import torch
from utils import load_config

def load_configuration(config_path="conf/base.yaml"):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (num_clients, beta, cfg) from the configuration
    """
    cfg = load_config(config_path)
    return cfg["num_clients"], cfg["beta"], cfg

def get_device():
    """
    Get the available computation device (CPU/GPU)
    
    Returns:
        torch.device: The device to use for computation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device