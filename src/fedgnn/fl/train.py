import torch
import torch.nn.functional as F
from src.fedgnn.models import VanillaGNN, MLP, GCN, GAT, SparseVanillaGNN, GCN_arxiv, GraphSAGEProducts, PubmedGAT, GAT_Arxiv
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader, DataLoader
import random
import numpy as np

# loga data instead of printing it
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# import wandb

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _model_device(model):
    return next(model.parameters()).device


def _seed_node_count(batch):
    """NeighborLoader places target seed nodes first in each sampled batch."""
    return int(getattr(batch, "batch_size", batch.num_nodes))


def train(model, data, epochs, optimizer, criterion, writer, use_amp=False, seed=None, grad_clip_norm=1.0,
               struct_reg_lambda=0.0, struct_reg_warmup_rounds=0):
    """
    Train a GNN model with optional structural consistency regularization.
    
    Structural regularization: L_struct = lambda * ||h_first - x_fp||^2
    where h_first is the hidden representation after the first GNN layer,
    and x_fp are the propagated (imputed) node features.
    
    This encourages the first-layer hidden representation to be consistent
    with the propagated feature structure, preventing FP from overwhelming
    discriminative learning.
    """
    # C5: seed is opt-in.
    if seed is not None:
        set_seed(int(seed))

    if isinstance(model, VanillaGNN):
        adjacency = to_dense_adj(data.edge_index)[0]
        adjacency += torch.eye(len(adjacency), device=adjacency.device)
    else:
        adjacency = None

    training_losses = []
    training_accuracies = []
    torch.cuda.empty_cache()

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Structural regularization: register forward hook on first GNN layer ──────
    hook_handle = None
    first_layer_output = None
    if struct_reg_lambda > 0:
        # Find first message-passing layer by name heuristic (works for GCN, GAT, GraphSAGE)
        first_layer = None
        first_layer_name = None
        for name, module in model.named_modules():
            module_str = type(module).__name__
            # Skip output classification layer (goes to dim_out)
            if hasattr(model, 'dim_out') and hasattr(module, 'out_features') and module.out_features == model.dim_out:
                continue
            # Skip input embedding layer for GAT/GATConv
            if 'emb' in name.lower() or 'embedding' in name.lower():
                continue
            if any(x in module_str for x in ('Conv', 'GATConv', 'SAGEConv', 'SGConv', 'GCNConv')):
                first_layer = module
                first_layer_name = name
                break
        
        if first_layer is not None:
            def hook_fn(module, input, output):
                nonlocal first_layer_output
                first_layer_output = output
            hook_handle = first_layer.register_forward_hook(hook_fn)
            print(f"[A3 struct reg] Registered hook on first layer: {first_layer_name} ({type(first_layer).__name__})")
        else:
            print(f"[A3 struct reg] WARNING: Could not find first layer — struct reg disabled.")
            struct_reg_lambda = 0.0  # disable if no hook possible
    
    try:
        for epoch in range(epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()
            first_layer_output = None  # reset each epoch

            # Use autocast for mixed precision
            with torch.amp.autocast("cuda", enabled=use_amp):
                if isinstance(model, VanillaGNN):
                    output = model(data.x, adjacency)
                elif isinstance(model, (GCN, GAT, GCN_arxiv, GraphSAGEProducts, PubmedGAT, GAT_Arxiv, SparseVanillaGNN)):
                    output = model(data.x, data.edge_index)
                elif isinstance(model, MLP):
                    output = model(data.x)
                else:
                    raise ValueError("Unknown model")

                out = output
                task_loss = criterion(out[data.train_mask], data.y[data.train_mask])
                total_loss = task_loss

                # ── A3: Structural consistency regularization ────────────────────
                # L_struct = lambda * ||h_first - x_fp||^2
                if struct_reg_lambda > 0 and epoch >= struct_reg_warmup_rounds and first_layer_output is not None:
                    x_fp = data.x  # propagated features stored in data.x
                    h_first = first_layer_output
                    # Mean-pooled over feature dimension for per-node regularizer
                    struct_loss = torch.mean((h_first - x_fp) ** 2)
                    total_loss = task_loss + struct_reg_lambda * struct_loss
                elif struct_reg_lambda > 0 and first_layer_output is None and epoch >= struct_reg_warmup_rounds:
                    # Hook didn't fire — fallback silently
                    pass

            loss = total_loss  # for logging (includes struct reg contribution)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            train_total = data.train_mask.sum().item()
            training_acc = (torch.argmax(out[data.train_mask], dim=1) == data.y[data.train_mask]).sum().item() / train_total if train_total > 0 else float('nan')
            training_losses.append(loss.item())
            training_accuracies.append(training_acc)

            if writer is not None:
                writer.add_scalar('Loss/train', loss, epoch)
                writer.add_scalar('Accuracy/train', training_acc, epoch)

            if writer is not None:
                val_loss, val_acc = evaluate(model, data, criterion)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
                logging.info(f'Epoch {epoch:>3}| Train Loss: {loss:.3f}| Train Accuracy: {training_acc:.3f}| Val Loss: {val_loss:.3f}| Val Accuracy: {val_acc:.3f}')

        # Final validation
        final_val_loss, final_val_acc = evaluate(model, data, criterion)

        return final_val_loss, training_acc, training_losses, training_accuracies
    finally:
        # Always remove hook to prevent memory leaks
        if hook_handle is not None:
            hook_handle.remove()

def evaluate(model, data, criterion):
    model.eval()
    with torch.no_grad():
        if isinstance(model, VanillaGNN):
            output = model(data.x, to_dense_adj(data.edge_index)[0])
        elif isinstance(model, (GCN, GAT, GCN_arxiv, GAT_Arxiv, GraphSAGEProducts, PubmedGAT, SparseVanillaGNN)):
            output = model(data.x, data.edge_index)
        elif isinstance(model, MLP):
            output = model(data.x)
        else:
            raise ValueError("Unknown model")
        out = output
        loss = criterion(out[data.val_mask], data.y[data.val_mask])
        _, pred = torch.max(out[data.val_mask], dim=1)
        correct = (pred == data.y[data.val_mask]).sum()
        val_total = int(data.val_mask.sum())
        acc = int(correct) / val_total if val_total > 0 else float('nan')
    return loss, acc

def test(model, data):
    model.eval()
    with torch.no_grad():
        if isinstance(model, VanillaGNN):
            output = model(data.x, to_dense_adj(data.edge_index)[0])
        elif isinstance(model, (GCN, GAT, GCN_arxiv, GAT_Arxiv, GraphSAGEProducts, PubmedGAT, SparseVanillaGNN)):
            output = model(data.x, data.edge_index)
        elif isinstance(model, MLP):
            output = model(data.x)
        else:
            raise ValueError("Unknown model")
        out = output
        _, pred = torch.max(out[data.test_mask], dim=1)
        correct = (pred == data.y[data.test_mask]).sum()
        test_total = int(data.test_mask.sum())
        acc = int(correct) / test_total if test_total > 0 else float('nan')
    return acc

def train_with_minibatch(model, data, epochs, optimizer, criterion, writer, batch_size=1024, num_neighbors=[10, 10, 10], use_amp=False, seed=None, grad_clip_norm=1.0):
    """
    Train the model using mini-batches to reduce memory consumption

    Args:
        model: The neural network model
        data: PyG data object
        epochs: Number of training epochs
        optimizer: Optimizer for the model
        criterion: Loss function
        writer: SummaryWriter for logging
        batch_size: Size of mini-batches
        num_neighbors: Number of neighbors to sample at each layer
        use_amp: Whether to use automatic mixed precision

    Returns:
        loss: Final validation loss
        acc: Final training accuracy
        loss_list: List of training losses
        acc_list: List of training accuracies
    """
    # C5: seed is opt-in.  Default (None) preserves the historical behavior of
    # always seeding with 42 so neighbor sampling is deterministic.  When an
    # int is provided, that seed is used instead -- this lets the bench harness
    # vary stochasticity across runs while keeping each run reproducible.
    set_seed(int(seed) if seed is not None else 42)

    training_losses = []
    training_accuracies = []

    # Clear CUDA cache before training
    torch.cuda.empty_cache()

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Create data loader with neighborhood sampling
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]

    # Create a neighbor loader for node-level tasks
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=train_idx,
        shuffle=True
    )

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        total_train_nodes = 0

        # Process mini-batches
        for batch_idx, batch in enumerate(train_loader):
            # Clear memory every few batches to prevent accumulation
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

            optimizer.zero_grad()

            # Move only the sampled batch to the model device.
            batch = batch.to(_model_device(model))

            # Debug information - print batch dimensions
            logging.debug(f"Batch size: {batch.num_nodes}, Features dim: {batch.x.size()}, Edge dim: {batch.edge_index.size()}")

            # Use autocast for mixed precision
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Forward pass depending on model type
                if isinstance(model, VanillaGNN):
                    # For vanilla GNN, convert to dense adjacency for the batch
                    adj = to_dense_adj(batch.edge_index)[0]
                    adj = adj + torch.eye(len(adj), device=adj.device)
                    output = model(batch.x, adj)
                elif isinstance(model, (GCN, GAT, GCN_arxiv, GraphSAGEProducts, PubmedGAT, GAT_Arxiv, SparseVanillaGNN)):
                    output = model(batch.x, batch.edge_index)
                elif isinstance(model, MLP):
                    output = model(batch.x)
                else:
                    raise ValueError("Unknown model")

                # Compute loss only on target seed nodes, not sampled neighbors.
                seed_count = _seed_node_count(batch)
                if seed_count > 0:
                    try:
                        loss = criterion(output[:seed_count], batch.y[:seed_count])
                    except RuntimeError as e:
                        logging.error(f"Error in mini-batch training: {str(e)}")
                        logging.error(f"Batch info - nodes: {batch.num_nodes}, output shape: {output.shape}, seed_count: {seed_count}")
                        continue

            # Backward pass with gradient scaling (outside autocast)
            if seed_count > 0:
                try:
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        if grad_clip_norm is not None and grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if grad_clip_norm is not None and grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                        optimizer.step()

                    # Calculate training accuracy for this batch
                    train_acc = (torch.argmax(output[:seed_count], dim=1) == batch.y[:seed_count]).sum().item() / seed_count

                    # Weight loss and accuracy by number of nodes in batch
                    batch_node_count = seed_count
                    epoch_loss += loss.item() * batch_node_count
                    epoch_acc += train_acc * batch_node_count
                    num_batches += 1
                    total_train_nodes += batch_node_count

                    # Clear batch data to free memory
                    del batch
                except RuntimeError as e:
                    logging.error(f"Error in mini-batch training: {str(e)}")
                    logging.error(f"Batch info - nodes: {batch.num_nodes}, output shape: {output.shape}, seed_count: {seed_count}")
                    # Clear memory on error
                    torch.cuda.empty_cache()
                    continue

        # Calculate average loss and accuracy for the epoch (weighted by number of nodes)
        if total_train_nodes > 0:
            avg_loss = epoch_loss / total_train_nodes
            avg_acc = epoch_acc / total_train_nodes

            training_losses.append(avg_loss)
            training_accuracies.append(avg_acc)

            if writer is not None:
                writer.add_scalar('Loss/train', avg_loss, epoch)
                writer.add_scalar('Accuracy/train', avg_acc, epoch)

            logging.info(f'Epoch {epoch:>3}| Train Loss: {avg_loss:.3f}| Train Accuracy: {avg_acc:.3f}')

    # Evaluate the model
    loss, acc = evaluate_with_minibatch(model, data, criterion, batch_size=batch_size, num_neighbors=num_neighbors)
    logging.info(f'Epoch {epochs-1:>3}| Validation Loss: {loss:.3f}, Validation Accuracy: {acc:.3f}')

    final_train_acc = training_accuracies[-1] if training_accuracies else float('nan')
    return loss, final_train_acc, training_losses, training_accuracies

def evaluate_with_minibatch(model, data, criterion, batch_size=1024, num_neighbors=[10, 10, 10], seed=None):
    """Evaluate the model using mini-batches"""
    if seed is not None:
        set_seed(seed)

    model.eval()

    # Create validation loader
    val_idx = data.val_mask.nonzero(as_tuple=True)[0]
    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=val_idx,
        shuffle=False
    )

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(_model_device(model))

            # Forward pass
            if isinstance(model, VanillaGNN):
                adj = to_dense_adj(batch.edge_index)[0]
                adj = adj + torch.eye(len(adj), device=adj.device)
                output = model(batch.x, adj)
            elif isinstance(model, (GCN, GAT, GCN_arxiv, GAT_Arxiv, GraphSAGEProducts, PubmedGAT, SparseVanillaGNN)):
                output = model(batch.x, batch.edge_index)
            elif isinstance(model, MLP):
                output = model(batch.x)
            else:
                raise ValueError("Unknown model")

            seed_count = _seed_node_count(batch)
            if seed_count > 0:
                try:
                    # Compute loss
                    loss = criterion(output[:seed_count], batch.y[:seed_count])
                    total_loss += loss.item() * seed_count

                    # Compute accuracy
                    pred = torch.argmax(output[:seed_count], dim=1)
                    correct += (pred == batch.y[:seed_count]).sum().item()
                    total += seed_count
                except RuntimeError as e:
                    logging.error(f"Error in mini-batch evaluation: {str(e)}")
                    logging.error(f"Batch info - nodes: {batch.num_nodes}, output shape: {output.shape}, seed_count: {seed_count}")
                    continue

    # Calculate average loss and accuracy
    avg_loss = total_loss / total if total > 0 else float('nan')
    avg_acc = correct / total if total > 0 else float('nan')

    return avg_loss, avg_acc

def test_with_minibatch(model, data, batch_size=1024, num_neighbors=[10, 10, 10], seed=None):
    """Test the model using mini-batches"""
    if seed is not None:
        set_seed(seed)

    model.eval()

    # Create test loader
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]
    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=test_idx,
        shuffle=False
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(_model_device(model))

            # Forward pass
            if isinstance(model, VanillaGNN):
                adj = to_dense_adj(batch.edge_index)[0]
                adj = adj + torch.eye(len(adj), device=adj.device)
                output = model(batch.x, adj)
            elif isinstance(model, (GCN, GAT, GCN_arxiv, GAT_Arxiv, GraphSAGEProducts, PubmedGAT, SparseVanillaGNN)):
                output = model(batch.x, batch.edge_index)
            elif isinstance(model, MLP):
                output = model(batch.x)
            else:
                raise ValueError("Unknown model")

            seed_count = _seed_node_count(batch)
            if seed_count > 0:
                try:
                    # Compute accuracy
                    pred = torch.argmax(output[:seed_count], dim=1)
                    correct += (pred == batch.y[:seed_count]).sum().item()
                    total += seed_count
                except RuntimeError as e:
                    logging.error(f"Error in mini-batch testing: {str(e)}")
                    logging.error(f"Batch info - nodes: {batch.num_nodes}, output shape: {output.shape}, seed_count: {seed_count}")
                    continue

    # Calculate accuracy
    acc = correct / total if total > 0 else float('nan')

    return acc
