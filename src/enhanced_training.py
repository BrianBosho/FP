import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import logging
import torch_sparse
from torch_geometric.loader import NeighborSampler
import gc
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def to_cpu_scalar(tensor):
    """Convert a tensor to a CPU scalar value safely."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().item()
    return tensor

def accuracy(output, labels):
    """
    Calculate accuracy between predicted and ground truth labels
    
    Args:
        output: Model output/predictions
        labels: Ground truth labels
    
    Returns:
        Accuracy as a float
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def custom_forward_batch(model, x, adjs, device):
    """
    A custom implementation of forward_batch that can be used if the model doesn't have one.
    
    Args:
        model: The GNN model
        x: Input features
        adjs: Adjacency information from NeighborSampler
        device: Device to use
    
    Returns:
        Model output
    """
    # Special handling for PyG's EdgeIndex object
    if hasattr(adjs, 'edge_index') and hasattr(adjs, 'size'):
        edge_index = adjs.edge_index.to(device)
        
        # Simply forward the features and edge index through the standard model
        # This should work for most GNN implementations
        return model(x, edge_index)
    
    # If adjs is a list or tensor, just use the first item's edge_index
    if isinstance(adjs, list) and len(adjs) > 0:
        if hasattr(adjs[0], 'edge_index'):
            edge_index = adjs[0].edge_index.to(device)
            return model(x, edge_index)
        elif isinstance(adjs[0], tuple) and len(adjs[0]) >= 1:
            edge_index = adjs[0][0].to(device)
            return model(x, edge_index)
        elif isinstance(adjs[0], torch.Tensor):
            edge_index = adjs[0].to(device)
            return model(x, edge_index)
    
    # If adjs is a direct tensor
    if isinstance(adjs, torch.Tensor):
        edge_index = adjs.to(device)
        return model(x, edge_index)
    
    # If we can't determine what to do, log an error and try using the standard forward
    logging.error(f"Unable to determine how to process adjs of type {type(adjs)}. Attempting standard forward.")
    if hasattr(model, 'forward') and callable(getattr(model, 'forward')):
        # Default to standard forward if we can't determine what to do
        # This may fail depending on the model's expected input format
        return model(x, None)
    else:
        raise ValueError(f"Model has no callable forward method and adjs cannot be processed: {type(adjs)}")

def enhanced_train_minibatch(model, data, epochs, optimizer, criterion, writer, batch_size=512, num_neighbors=None, dataset_type="standard"):
    """
    Memory-efficient training for large graphs using mini-batch sampling.
    
    Args:
        model: The GNN model to train
        data: PyG data object containing the graph (or a list of data objects for OGBN-Products)
        epochs: Number of training epochs
        optimizer: PyTorch optimizer
        criterion: Loss function
        writer: TensorBoard writer for logging
        batch_size: Number of nodes per batch
        num_neighbors: Number of neighbors to sample for each layer
        dataset_type: Type of dataset - "standard", "arxiv", or "products"
    
    Returns:
        Tuple of (final_loss, final_accuracy, training_losses, training_accuracies)
    """
    # Reduce memory usage for OGBN-Products dataset
    if dataset_type == "products":
        batch_size = min(batch_size, 512)  # Smaller batch size for products
        epochs = min(epochs, 2)  # Fewer epochs to reduce memory pressure
        
    device = next(model.parameters()).device
    training_losses = []
    training_accuracies = []
    
    # Force garbage collection before starting
    gc.collect()
    torch.cuda.empty_cache()
    
    # Handle different data structures
    if isinstance(data, list):
        # For OGBN-Products dataset with modified data structure
        logging.info("Processing list data structure for OGBN-Products dataset in training")
        # The data is usually split as [data_obj, split_idx]
        if len(data) >= 2:
            data_obj = data[0]
            split_idx = data[1]
            
            # Get train indices from split_idx
            if isinstance(split_idx, dict) and 'train' in split_idx:
                train_idx = split_idx['train'].to(device)
                logging.info(f"Found training set with {len(train_idx)} nodes")
            elif hasattr(split_idx, 'train_mask'):
                # Handle case where split_idx is a Data object with train_mask
                train_idx = split_idx.train_mask.nonzero().squeeze().to(device)
                logging.info(f"Found training set with {len(train_idx)} nodes in Data object")
            else:
                # Fallback to a random subset of nodes
                logging.info(f"Using random sampling for training nodes, split_idx type: {type(split_idx)}")
                num_nodes = data_obj.x.size(0)
                train_idx = torch.randperm(num_nodes)[:10000].to(device)
                
            edge_index = data_obj.edge_index
            x = data_obj.x
            y = data_obj.y
        else:
            logging.error(f"Invalid data structure: list of length {len(data)}")
            return 0.0, 0.0, [], []
    else:
        # Standard PyG data object
        if not hasattr(data, 'train_mask'):
            logging.error("Data object does not have train_mask attribute")
            return 0.0, 0.0, [], []
            
        # Extract training node indices from the training mask
        train_idx = data.train_mask.nonzero().squeeze().to(device)
        edge_index = data.edge_index
        x = data.x
        y = data.y
    
    # Use a smaller subset if we have too many training nodes
    max_train_nodes = 5000 if dataset_type == "products" else 10000
    if len(train_idx) > max_train_nodes:
        logging.info(f"Using a subset of training nodes: {max_train_nodes} out of {len(train_idx)}")
        # Use a random subset to improve generalization
        perm = torch.randperm(len(train_idx))[:max_train_nodes]
        train_idx = train_idx[perm]
    
    logging.info(f"Training on {len(train_idx)} nodes with batch size {batch_size}")
    
    # Set default neighbors if not provided
    if num_neighbors is None:
        # Determine appropriate number of neighbors based on model layers
        num_layers = getattr(model, 'num_layers', 2)
        
        if num_layers == 2:
            # For a 2-layer model, we need only 1 neighbor sampling step
            # Use fewer neighbors for products dataset
            neighbors_size = 5 if dataset_type == "products" else 10
            num_neighbors = [neighbors_size]  # Just one sampling between input and output
            logging.info(f"Using 1 neighbor sampling step with {neighbors_size} neighbors for 2-layer model during training")
        else:
            # For models with more layers, use sampling for each layer transition
            neighbors_size = 5 if dataset_type == "products" else 10
            num_neighbors = [neighbors_size] * (num_layers - 1)
            logging.info(f"Using {num_layers-1} neighbor sampling steps with {neighbors_size} neighbors for {num_layers}-layer model during training")
            
    logging.info(f"Using neighbor sampling sizes: {num_neighbors}")
    
    # Create a neighbor sampler for efficient message passing
    try:
        train_loader = NeighborSampler(
            edge_index, 
            node_idx=train_idx,
            sizes=num_neighbors, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
    except Exception as e:
        logging.error(f"Error creating NeighborSampler: {e}")
        logging.error(f"Edge index shape: {edge_index.shape}, Train idx shape: {train_idx.shape}")
        return 0.0, 0.0, [], []
    
    # Check if the model has a forward_batch method
    has_forward_batch = hasattr(model, 'forward_batch') and callable(getattr(model, 'forward_batch'))
    if not has_forward_batch:
        logging.warning(f"Model does not have a forward_batch method. Using custom implementation.")
    
    # For tracking overall epoch stats
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_examples = 0
        
        # Process each batch
        for batch_idx, (batch_size, n_id, adjs) in enumerate(train_loader):
            # Debug first batch to understand structure
            if batch_idx == 0 and epoch == 0:
                logging.info(f"Batch {batch_idx}: adjs type={type(adjs)}")
                if hasattr(adjs, 'edge_index'):
                    logging.info(f"adjs.edge_index shape: {adjs.edge_index.shape}")
                if hasattr(adjs, 'size'):
                    logging.info(f"adjs.size: {adjs.size}")
            
            # Clear memory
            optimizer.zero_grad()
            
            # Get features
            batch_x = x[n_id].to(device)
            batch_y = y[n_id[:batch_size]].squeeze().to(device)
            
            # Forward pass - with neighbor sampling
            if has_forward_batch:
                # Use the model's forward_batch method
                out = model.forward_batch(batch_x, adjs)
            else:
                # Use our custom implementation
                out = custom_forward_batch(model, batch_x, adjs, device)
            
            # Compute loss
            loss = criterion(out, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Compute accuracy
            pred = out.argmax(dim=-1)
            correct = pred.eq(batch_y).sum().item()
            
            # Update statistics
            total_loss += float(loss) * batch_size
            total_correct += correct
            total_examples += batch_size
            
            # Clear memory
            del batch_x, batch_y, out, adjs, pred, loss
            
            # More aggressive garbage collection for products
            if dataset_type == "products" and batch_idx % 3 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                logging.info(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {total_loss/total_examples if total_examples > 0 else 0:.4f}, '
                           f'Acc: {total_correct/total_examples if total_examples > 0 else 0:.4f}')
        
        # Compute epoch statistics
        epoch_loss = total_loss / total_examples
        epoch_acc = total_correct / total_examples
        
        # Log epoch results
        logging.info(f'Epoch: {epoch}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        if writer:
            writer.add_scalar('Train/Loss', epoch_loss, epoch)
            writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
        
        # Save metrics
        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_acc)
        
        # Force garbage collection between epochs
        gc.collect()
        torch.cuda.empty_cache()
    
    # Clear memory
    del train_loader, train_idx, edge_index, x, y
    gc.collect()
    torch.cuda.empty_cache()
    
    # Return final metrics
    final_loss = training_losses[-1] if training_losses else 0.0
    final_acc = training_accuracies[-1] if training_accuracies else 0.0
    
    return final_loss, final_acc, training_losses, training_accuracies

def enhanced_test_minibatch(model, data, batch_size=512, num_neighbors=None, dataset_type="standard"):
    """
    Memory-efficient test for large graphs using mini-batch sampling.
    
    Args:
        model: The GNN model to test
        data: PyG data object containing the graph (or a list of data objects for OGBN-Products)
        batch_size: Number of nodes per batch
        num_neighbors: Number of neighbors to sample for each layer
        dataset_type: Type of dataset - "standard", "arxiv", or "products"
    
    Returns:
        Test accuracy
    """
    # Reduce memory usage for OGBN-Products dataset
    if dataset_type == "products":
        batch_size = min(batch_size, 512)  # Smaller batch size for products
        
    device = next(model.parameters()).device
    
    # Force garbage collection before starting
    gc.collect()
    torch.cuda.empty_cache()
    
    # Handle different data structures
    if isinstance(data, list):
        # For OGBN-Products dataset with modified data structure
        logging.info("Processing list data structure for OGBN-Products dataset")
        # The data is usually split as [data_obj, split_idx]
        if len(data) >= 2:
            data_obj = data[0]
            split_idx = data[1]
            
            # Get test indices from split_idx
            if isinstance(split_idx, dict) and 'test' in split_idx:
                test_idx = split_idx['test'].to(device)
                logging.info(f"Found test set with {len(test_idx)} nodes")
            elif hasattr(split_idx, 'test_mask'):
                # Handle case where split_idx is a Data object with test_mask
                test_idx = split_idx.test_mask.nonzero().squeeze().to(device)
                logging.info(f"Found test set with {len(test_idx)} nodes in Data object")
            else:
                # Fallback to a random subset of nodes
                logging.info(f"Using random sampling for test nodes, split_idx type: {type(split_idx)}")
                num_nodes = data_obj.x.size(0)
                test_idx = torch.randperm(num_nodes)[:10000].to(device)
                
            edge_index = data_obj.edge_index
            x = data_obj.x
            y = data_obj.y
        else:
            logging.error(f"Invalid data structure: list of length {len(data)}")
            return 0.0
    else:
        # Standard PyG data object
        if not hasattr(data, 'test_mask'):
            logging.error("Data object does not have test_mask attribute")
            return 0.0
            
        # Extract test node indices from the test mask
        test_idx = data.test_mask.nonzero().squeeze().to(device)
        edge_index = data.edge_index
        x = data.x
        y = data.y
    
    # Use a smaller subset if we have too many test nodes
    max_test_nodes = 5000 if dataset_type == "products" else 10000
    if len(test_idx) > max_test_nodes:
        logging.info(f"Using a subset of test nodes: {max_test_nodes} out of {len(test_idx)}")
        test_idx = test_idx[:max_test_nodes]
    
    # Set default neighbors if not provided
    if num_neighbors is None:
        # Determine appropriate number of neighbors based on model layers
        num_layers = getattr(model, 'num_layers', 2)
        
        if num_layers == 2:
            # For a 2-layer model, we need only 1 neighbor sampling step
            # Use fewer neighbors for products dataset
            neighbors_size = 5 if dataset_type == "products" else 10
            num_neighbors = [neighbors_size]  # Just one sampling between input and output
            logging.info(f"Using 1 neighbor sampling step with {neighbors_size} neighbors for 2-layer model during testing")
        else:
            # For models with more layers, use sampling for each layer transition
            neighbors_size = 5 if dataset_type == "products" else 10
            num_neighbors = [neighbors_size] * (num_layers - 1)
            logging.info(f"Using {num_layers-1} neighbor sampling steps with {neighbors_size} neighbors for {num_layers}-layer model during testing")
    
    logging.info(f"Testing with batch size: {batch_size}, Neighbor sampling: {num_neighbors}")
    
    # Create a neighbor sampler for efficient message passing
    try:
        test_loader = NeighborSampler(
            edge_index, 
            node_idx=test_idx,
            sizes=num_neighbors, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    except Exception as e:
        logging.error(f"Error creating NeighborSampler: {e}")
        logging.error(f"Edge index shape: {edge_index.shape}, Test idx shape: {test_idx.shape}")
        return 0.0
    
    # Check if the model has a forward_batch method
    has_forward_batch = hasattr(model, 'forward_batch') and callable(getattr(model, 'forward_batch'))
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (batch_size, n_id, adjs) in enumerate(test_loader):
            # Get features for nodes in this batch
            batch_x = x[n_id].to(device)
            
            # For test nodes, get their labels
            batch_y = y[n_id[:batch_size]].squeeze().to(device)
            
            # Use the same forward method as in training
            if has_forward_batch:
                out = model.forward_batch(batch_x, adjs)
            else:
                out = custom_forward_batch(model, batch_x, adjs, device)
                
            pred = out.argmax(dim=-1)
            
            correct += pred.eq(batch_y).sum().item()
            total += batch_size
            
            # Clear memory
            del batch_x, batch_y, out, adjs
            
            # Garbage collect every few batches
            if batch_idx % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
            # Print progress for large datasets
            if batch_idx % 10 == 0 and dataset_type == "products":
                logging.debug(f"Testing batch {batch_idx}/{len(test_loader)}, Current accuracy: {correct/total:.4f}")
    
    # Calculate accuracy
    test_acc = correct / total if total > 0 else 0
    
    # Clear memory
    del test_loader, test_idx, edge_index, x, y
    gc.collect()
    torch.cuda.empty_cache()
    
    return test_acc

def enhanced_train(model, data, epochs, optimizer, criterion, writer, dataset_type="standard"):
    """
    Enhanced training function incorporating FedGCN training logic with memory efficiency
    
    Args:
        model: The GNN model to train
        data: PyG data object containing the graph
        epochs: Number of training epochs
        optimizer: PyTorch optimizer
        criterion: Loss function
        writer: TensorBoard writer for logging
        dataset_type: Type of dataset - "standard", "arxiv", or "products"
    
    Returns:
        Tuple of (final_loss, final_accuracy, training_losses, training_accuracies)
    """
    # For large datasets like products, use mini-batch training
    if dataset_type == "products":
        logging.info("Using mini-batch training for OGBN-Products dataset")
        # Check if the model has a forward_batch method for mini-batch training
        if not hasattr(model, 'forward_batch'):
            logging.warning("Model does not have forward_batch method. Cannot use mini-batch training.")
            logging.warning("Please use a model that supports mini-batch training.")
        else:
            # Use smaller batch sizes and fewer neighbors for very large graphs
            batch_size = 1024
            
            # Determine appropriate number of neighbors based on model layers
            num_layers = getattr(model, 'num_layers', 2)
            
            # For a 2-layer model, we need sampling for both layers
            # If using a 3-layer model, we need sampling for all 3 layers, etc.
            # Create a list with the same length as the number of layers
            if num_layers == 2:
                # For a 2-layer model, we need only 1 neighbor sampling step
                num_neighbors = [10]  # Just one sampling between input and output
                logging.info(f"Using 1 neighbor sampling step for 2-layer model")
            else:
                # For models with more layers, use sampling for each layer transition
                num_neighbors = [10] * (num_layers - 1)
                logging.info(f"Using {num_layers-1} neighbor sampling steps for {num_layers}-layer model")
                
            logging.info(f"Batch size: {batch_size}, Neighbor sampling: {num_neighbors}")
            
            return enhanced_train_minibatch(
                model, data, epochs, optimizer, criterion, 
                writer, batch_size, num_neighbors, dataset_type
            )
    
    # For standard datasets, use the original training function
    training_losses = []
    training_accuracies = []
    torch.cuda.empty_cache()
    
    device = next(model.parameters()).device
    
    # For large datasets (arxiv and products), we'll avoid creating dense adjacency matrices
    is_large_dataset = dataset_type in ["arxiv", "products"]
    
    # Only create adjacency matrix for smaller datasets and models that need it
    adjacency = None
    if not is_large_dataset and hasattr(data, 'edge_index'):
        # Check if the model requires dense adjacency matrices
        from models import VanillaGNN
        if hasattr(model, 'model_type') and model.model_type == 'vanilla' or isinstance(model, VanillaGNN):
            # First check if creating the adjacency matrix is feasible
            num_nodes = data.x.size(0)
            memory_required = (num_nodes * num_nodes * 4) / (1024 ** 3)  # in GB
            
            if memory_required > 10:  # If more than 10GB would be needed
                logging.warning(f"Dense adjacency matrix would require {memory_required:.2f}GB. "
                               f"Using sparse operations for VanillaGNN.")
                # We'll handle this in the forward pass
            else:
                # Only create dense adjacency for smaller graphs
                adjacency = to_dense_adj(data.edge_index)[0]
                adjacency += torch.eye(len(adjacency), device=adjacency.device)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass based on model type and dataset
        if is_large_dataset:
            # For large datasets, always use edge_index (sparse) format
            output = model(data.x, data.edge_index)
        else:
            # For standard datasets, handle different model types
            if hasattr(model, 'model_type'):
                if model.model_type == 'vanilla' and adjacency is not None:
                    output = model(data.x, adjacency)
                else:
                    output = model(data.x, data.edge_index)
            else:
                # Handle different model types similar to FP's train.py
                from models import VanillaGNN, MLP
                from gnn_models import GCN, GAT
                
                if isinstance(model, VanillaGNN):
                    if adjacency is not None:
                        output = model(data.x, adjacency)
                    else:
                        # If adjacency was too large, use a sparse-friendly forward pass
                        # This assumes VanillaGNN can handle sparse inputs or has been modified to do so
                        output = model(data.x, data.edge_index)
                elif isinstance(model, (GCN, GAT)):
                    output = model(data.x, data.edge_index)
                elif isinstance(model, MLP):
                    output = model(data.x)
                else:
                    raise ValueError(f"Unknown model type: {type(model)}")
        
        # Calculate loss
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        
        # Apply gradient clipping (from FedGCN best practices)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate accuracy
        training_acc = accuracy(output[data.train_mask], data.y[data.train_mask])
        
        # Store metrics as CPU values
        training_losses.append(to_cpu_scalar(loss))
        training_accuracies.append(to_cpu_scalar(training_acc))
        
        if writer is not None:
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Accuracy/train', training_acc, epoch)
        
        if epoch % 2 == 0:
            logging.info(f'Epoch {epoch:>3}| Train Loss: {loss:.3f}| Train Accuracy: {training_acc:.3f}')
    
    # Evaluate after training
    val_loss, val_acc = enhanced_evaluate(model, data, criterion, dataset_type)
    logging.info(f'Final Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_acc:.3f}')
    
    # Clear any remaining references to large tensors
    if adjacency is not None:
        del adjacency
    torch.cuda.empty_cache()
    
    # Return CPU values
    return to_cpu_scalar(val_loss), to_cpu_scalar(val_acc), training_losses, training_accuracies

def enhanced_evaluate(model, data, criterion, dataset_type="standard"):
    """
    Enhanced evaluation function with memory efficiency
    
    Args:
        model: The GNN model to evaluate
        data: PyG data object containing the graph
        criterion: Loss function
        dataset_type: Type of dataset - "standard", "arxiv", or "products"
    
    Returns:
        Tuple of (validation_loss, validation_accuracy)
    """
    # For large datasets like products, use mini-batch evaluation
    if dataset_type == "products":
        logging.info("Using mini-batch evaluation for OGBN-Products dataset")
        # Check if the model has a forward_batch method for mini-batch training
        if hasattr(model, 'forward_batch'):
            # Evaluate with mini-batches
            test_acc = enhanced_test_minibatch(model, data, batch_size=1024, num_neighbors=None, dataset_type=dataset_type)
            # Return dummy loss value and actual accuracy
            return 0.0, test_acc
    
    model.eval()
    device = next(model.parameters()).device
    
    # For large datasets, we'll avoid creating dense adjacency matrices
    is_large_dataset = dataset_type in ["arxiv", "products"]
    
    # Only create adjacency matrix for smaller datasets and models that need it
    adjacency = None
    if not is_large_dataset and hasattr(data, 'edge_index'):
        # Check if the model requires dense adjacency matrices
        from models import VanillaGNN
        if hasattr(model, 'model_type') and model.model_type == 'vanilla' or isinstance(model, VanillaGNN):
            # First check if creating the adjacency matrix is feasible
            num_nodes = data.x.size(0)
            memory_required = (num_nodes * num_nodes * 4) / (1024 ** 3)  # in GB
            
            if memory_required <= 10:  # If less than 10GB would be needed
                # Only create dense adjacency for smaller graphs
                adjacency = to_dense_adj(data.edge_index)[0]
    
    with torch.no_grad():
        # Forward pass based on model type and dataset
        if is_large_dataset:
            # For large datasets, always use edge_index (sparse) format
            output = model(data.x, data.edge_index)
        else:
            # For standard datasets, handle different model types
            if hasattr(model, 'model_type'):
                if model.model_type == 'vanilla' and adjacency is not None:
                    output = model(data.x, adjacency)
                else:
                    output = model(data.x, data.edge_index)
            else:
                # Handle different model types
                from models import VanillaGNN, MLP
                from gnn_models import GCN, GAT
                
                if isinstance(model, VanillaGNN):
                    if adjacency is not None:
                        output = model(data.x, adjacency)
                    else:
                        # If adjacency was too large, use a sparse-friendly forward pass
                        output = model(data.x, data.edge_index)
                elif isinstance(model, (GCN, GAT)):
                    output = model(data.x, data.edge_index)
                elif isinstance(model, MLP):
                    output = model(data.x)
                else:
                    raise ValueError(f"Unknown model type: {type(model)}")
        
        # Calculate loss
        loss = criterion(output[data.val_mask], data.y[data.val_mask])
        
        # Calculate accuracy
        val_acc = accuracy(output[data.val_mask], data.y[data.val_mask])
    
    # Clear any remaining references to large tensors
    if adjacency is not None:
        del adjacency
    torch.cuda.empty_cache()
    
    # Return CPU values
    return to_cpu_scalar(loss), to_cpu_scalar(val_acc)

def enhanced_test(model, data, dataset_type="standard"):
    """
    Enhanced test function incorporating FedGCN test logic with memory efficiency
    
    Args:
        model: The GNN model to test
        data: PyG data object containing the graph (or a list of data objects for OGBN-Products)
        dataset_type: Type of dataset - "standard", "arxiv", or "products"
    
    Returns:
        Test accuracy
    """
    # For large datasets like products, use mini-batch testing
    if dataset_type == "products":
        logging.info("Using mini-batch testing for OGBN-Products dataset")
        # Check if the model has a forward_batch method for mini-batch training
        if hasattr(model, 'forward_batch'):
            return enhanced_test_minibatch(model, data, batch_size=1024, num_neighbors=None, dataset_type=dataset_type)
        else:
            logging.warning("Model does not have forward_batch method. Cannot use mini-batch testing.")
            logging.warning("Attempting standard testing, but may run out of memory.")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Handle different data structures
    if isinstance(data, list):
        # For OGBN-Products dataset with modified data structure
        logging.info("Processing list data structure for OGBN-Products dataset in testing")
        # The data is usually split as [data_obj, split_idx]
        if len(data) >= 2:
            data_obj = data[0]
            split_idx = data[1]
            
            # Get test indices from split_idx
            if isinstance(split_idx, dict) and 'test' in split_idx:
                test_mask = torch.zeros(data_obj.x.size(0), dtype=torch.bool, device=device)
                test_mask[split_idx['test']] = True
                
                is_large_dataset = True
                edge_index = data_obj.edge_index
                x = data_obj.x
                y = data_obj.y
                
                # Create a data object with the extracted attributes
                from torch_geometric.data import Data
                test_data = Data(x=x, edge_index=edge_index, y=y, test_mask=test_mask)
                
                # Use mini-batch testing for this data
                return enhanced_test_minibatch(model, test_data, batch_size=1024, num_neighbors=None, dataset_type=dataset_type)
            else:
                logging.error(f"Invalid split_idx structure: {type(split_idx)}")
                return 0.0
        else:
            logging.error(f"Invalid data structure: list of length {len(data)}")
            return 0.0
    
    # For large datasets, we'll avoid creating dense adjacency matrices
    is_large_dataset = dataset_type in ["arxiv", "products"]
    
    # Only create adjacency matrix for smaller datasets and models that need it
    adjacency = None
    if not is_large_dataset and hasattr(data, 'edge_index'):
        # Check if the model requires dense adjacency matrices
        from models import VanillaGNN
        if hasattr(model, 'model_type') and model.model_type == 'vanilla' or isinstance(model, VanillaGNN):
            # First check if creating the adjacency matrix is feasible
            num_nodes = data.x.size(0)
            memory_required = (num_nodes * num_nodes * 4) / (1024 ** 3)  # in GB
            
            if memory_required > 10:  # If more than 10GB would be needed
                logging.warning(f"Dense adjacency matrix would require {memory_required:.2f}GB. "
                               f"Using sparse operations for VanillaGNN.")
                # We'll handle this in the forward pass
            else:
                # Only create dense adjacency for smaller graphs
                adjacency = to_dense_adj(data.edge_index)[0]
                adjacency += torch.eye(len(adjacency), device=adjacency.device)
                
    # Prepare input data
    x = data.x.to(device)
    
    # Forward pass based on model type and dataset
    with torch.no_grad():
        if is_large_dataset:
            # For large datasets, always use edge_index (sparse) format
            output = model(x, data.edge_index)
        else:
            # For standard datasets, handle different model types
            if hasattr(model, 'model_type'):
                if model.model_type == 'vanilla' and adjacency is not None:
                    output = model(x, adjacency)
                else:
                    output = model(x, data.edge_index)
            else:
                # Handle different model types similar to FP's train.py
                from models import VanillaGNN, MLP
                from gnn_models import GCN, GAT
                
                if isinstance(model, VanillaGNN):
                    if adjacency is not None:
                        output = model(x, adjacency)
                    else:
                        # If adjacency was too large, use a sparse-friendly forward pass
                        output = model(x, data.edge_index)
                elif isinstance(model, (GCN, GAT)):
                    output = model(x, data.edge_index)
                elif isinstance(model, MLP):
                    output = model(x)
                else:
                    raise ValueError(f"Unknown model type: {type(model)}")
    
    # Calculate accuracy using test mask
    if hasattr(data, 'test_mask'):
        pred = output[data.test_mask].argmax(dim=1)
        correct = pred.eq(data.y[data.test_mask]).sum().item()
        total = int(data.test_mask.sum())
        test_acc = correct / total if total > 0 else 0
    else:
        logging.error("Data object does not have test_mask attribute")
        test_acc = 0.0
    
    # Clear memory
    del output
    torch.cuda.empty_cache()
    
    return test_acc 