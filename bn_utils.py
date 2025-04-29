import torch
from torch_geometric.nn import BatchNorm

def extract_bn_stats(model):
    """Extract BatchNorm running statistics from a model in a Ray-friendly format."""
    layer_names = []
    running_means = []
    running_vars = []
    num_batches = []
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, BatchNorm)):
            layer_names.append(name)
            running_means.append(module.running_mean.detach().cpu().clone())
            running_vars.append(module.running_var.detach().cpu().clone())
            if hasattr(module, 'num_batches_tracked'):
                num_batches.append(module.num_batches_tracked.detach().cpu().clone())
            else:
                num_batches.append(None)
    
    return {
        "layer_names": layer_names,
        "running_means": running_means,
        "running_vars": running_vars, 
        "num_batches": num_batches
    }

def aggregate_bn_stats(bn_stats_list, sample_sizes=None):
    """
    Aggregate BatchNorm statistics from multiple clients - Ray-friendly version.
    
    Args:
        bn_stats_list: List of dictionaries in flattened format
        sample_sizes: List of client data sizes for weighted averaging
    
    Returns:
        Dictionary with aggregated BatchNorm statistics
    """
    if not bn_stats_list:
        return {}
    
    # Use equal weights if sample sizes not provided
    if sample_sizes is None:
        weights = [1.0 / len(bn_stats_list)] * len(bn_stats_list)
    else:
        total_samples = sum(sample_sizes)
        weights = [size / total_samples for size in sample_sizes]
    
    # Get the first client's data to initialize
    first_client = bn_stats_list[0]
    layer_names = first_client["layer_names"]
    
    # Initialize aggregated stats
    agg_running_means = {}
    agg_running_vars = {}
    agg_num_batches = {}
    
    # Process each layer
    for layer_idx, layer_name in enumerate(layer_names):
        # Initialize with zeros of the right shape
        mean_shape = first_client["running_means"][layer_idx].shape
        var_shape = first_client["running_vars"][layer_idx].shape
        
        agg_running_means[layer_name] = torch.zeros(mean_shape)
        agg_running_vars[layer_name] = torch.zeros(var_shape)
        agg_num_batches[layer_name] = 0
        
        # Accumulate weighted values from all clients
        for client_idx, client_stats in enumerate(bn_stats_list):
            if layer_idx >= len(client_stats["layer_names"]):
                continue  # Skip if this client doesn't have this layer
                
            weight = weights[client_idx]
            
            # Add weighted contributions
            agg_running_means[layer_name] += client_stats["running_means"][layer_idx] * weight
            agg_running_vars[layer_name] += client_stats["running_vars"][layer_idx] * weight
            
            # For num_batches, just sum them
            if client_stats["num_batches"][layer_idx] is not None:
                agg_num_batches[layer_name] += client_stats["num_batches"][layer_idx]
    
    # Package in a format for update_bn_stats
    return {
        "layer_names": layer_names,
        "agg_running_means": agg_running_means,
        "agg_running_vars": agg_running_vars,
        "agg_num_batches": agg_num_batches
    }

def update_bn_stats(model, aggregated_stats):
    """Update a model's BatchNorm running statistics with aggregated values - Ray-friendly version."""
    if not aggregated_stats or "layer_names" not in aggregated_stats:
        return model
        
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, BatchNorm)) and name in aggregated_stats["agg_running_means"]:
                # Update running mean
                module.running_mean.copy_(
                    aggregated_stats["agg_running_means"][name].to(module.running_mean.device)
                )
                
                # Update running variance
                module.running_var.copy_(
                    aggregated_stats["agg_running_vars"][name].to(module.running_var.device)
                )
                
                # Update num_batches_tracked if available
                if hasattr(module, 'num_batches_tracked') and name in aggregated_stats["agg_num_batches"]:
                    module.num_batches_tracked.copy_(
                        torch.tensor(aggregated_stats["agg_num_batches"][name]).to(module.num_batches_tracked.device)
                    )
    
    return model
