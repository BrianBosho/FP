import torch
from src.utils import load_config
from src.models import GCN, GAT, GCN_arxiv, GraphSAGEProducts
from pathlib import Path
from pathlib import Path
import yaml

def load_configuration(config_path=None):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file, 
                     if None, will look in the project root
        
    Returns:
        Tuple of (num_clients, beta, cfg) from the configuration
    """
    if config_path is None:
        # Get the project root (parent of src)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "conf" / "base.yaml"
    
    cfg = load_config(str(config_path))
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

def instantiate_model(model_type, num_features, num_classes, device, dataset_name="Cora"):
    """
    Instantiate a model based on the specified type and parameters.
    
    Args:
        model_type: The type of model to instantiate (GCN, GAT)
        num_features: Number of input features
        num_classes: Number of output classes
        device: Device to place model on (CPU/GPU)
        dataset_name: Name of the dataset (default: "Cora")
        
    Returns:
        The instantiated model placed on the specified device
    """
    if model_type == "GCN":
        if dataset_name == "ogbn-arxiv":
            model = GCN_arxiv(input_dim=num_features, hidden_dim=256, output_dim=40, dropout=0.5)
            print(f"Model is {model}")
            return model.to(device)
        elif dataset_name == "ogbn-products":
            model = GraphSAGEProducts(input_dim=num_features, hidden_dim=256, output_dim=47, dropout=0.5, num_layers=3)
            print(f"Model is {model}")
            return model.to(device)
        else:
            return GCN(num_features, 16, num_classes).to(device)
    elif model_type == "GAT":
        return GAT(num_features, 16, num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")