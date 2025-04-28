import torch
import torch.nn.functional as F
from src.models import VanillaGNN, MLP
from src.gnn_models import GCN, GAT, GCN_arxiv, GCN_products, SAGE_products
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader, DataLoader

# loga data instead of printing it
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def train(model, data, epochs, optimizer, criterion, writer):

    if isinstance(model, VanillaGNN):
        adjacency = to_dense_adj(data.edge_index)[0]
        adjacency += torch.eye(len(adjacency), device=adjacency.device)
    else:
        adjacency = None
    # adjacency += torch.eye(len(adjacency))
    


    # round_number = config["round_number"]-1
    training_losses = []
    training_accuracies = []
    torch.cuda.empty_cache()

    # Add this check before using the model
    if isinstance(model, GCN_arxiv) or isinstance(model, SAGE_products):
        # Ensure edge_index has correct format
        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            raise ValueError(f"Edge index has incorrect format: {data.edge_index.shape}")
        # Ensure x has correct format
        if data.x.dim() != 2:
            raise ValueError(f"Node features have incorrect format: {data.x.shape}")
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        if isinstance(model, VanillaGNN):
            output = model(data.x, adjacency)
        elif isinstance(model, GCN) or isinstance(model, GAT) or isinstance(model, GCN_arxiv) or isinstance(model, SAGE_products):
            output = model(data.x, data.edge_index)
        elif isinstance(model, MLP):
            output = model(data.x)
        else:
            raise ValueError("Unknown model")
        # out = model(data.x, data.edge_index)
        out = output
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
         # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


        optimizer.step()

        training_acc = (torch.argmax(out[data.train_mask], dim=1) == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
        # global_epoch = round_number * epochs + epoch
        # writer.add_scalar(f'Loss/train_round_{round_number}', loss, epoch) 
        training_losses.append(loss.item())
        training_accuracies.append(training_acc)
        if writer is not None:
            writer.add_scalar('Loss/train', loss, epoch)
            # write training accuracy
            writer.add_scalar('Accuracy/train', training_acc, epoch)
         
        # if epoch % 2 == 0:
        logging.info(f'Epoch {epoch:>3}| Train Loss: {loss:.3f}| Train Accuracy: {training_acc:.3f}')
    
    # evaluate the model by calling hr the evaluate function
    loss, acc = evaluate(model, data, criterion)
    logging.info(f'Epoch {epoch:>3}| Validation Loss: {loss:.3f}, Validation Accuracy: {acc:.3f}')


    return loss, training_acc, training_losses, training_accuracies

def evaluate(model, data, criterion):
    model.eval()
    with torch.no_grad():
        if isinstance(model, VanillaGNN):
            output = model(data.x, to_dense_adj(data.edge_index)[0])
        elif isinstance(model, GCN) or isinstance(model, GAT) or isinstance(model, GCN_arxiv) or isinstance(model, SAGE_products):
            output = model(data.x, data.edge_index)
        elif isinstance(model, MLP):
            output = model(data.x)
        else:
            raise ValueError("Unknown model in Evaluation")
        # out = model(data.x, data.edge_index)
        out = output
        # out = model(data.x, data.edge_index)
        loss = criterion(out[data.val_mask], data.y[data.val_mask])
        _, pred = torch.max(out[data.val_mask], dim=1)
        correct = (pred == data.y[data.val_mask]).sum()
        acc = int(correct) / int(data.val_mask.sum())
    return loss, acc

def test(model, data):
    model.eval()
    with torch.no_grad():
        if isinstance(model, VanillaGNN):
            output = model(data.x, to_dense_adj(data.edge_index)[0])
        elif isinstance(model, GCN) or isinstance(model, GAT) or isinstance(model, GCN_arxiv) or isinstance(model, SAGE_products):
            output = model(data.x, data.edge_index)
        elif isinstance(model, MLP):
            output = model(data.x)
        else:
            raise ValueError("Unknown model while testing")
        # out = model(data.x, data.edge_index)
        out = output
        _, pred = torch.max(out[data.test_mask], dim=1)
        correct = (pred == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
    return acc

def train_with_minibatch(model, data, epochs, optimizer, criterion, writer, batch_size=1024, num_neighbors=[10, 10, 10]):
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
    
    Returns:
        loss: Final validation loss
        acc: Final training accuracy
        loss_list: List of training losses
        acc_list: List of training accuracies
    """
    training_losses = []
    training_accuracies = []
    
    # Clear CUDA cache before training
    torch.cuda.empty_cache()
    
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
        
        # Process mini-batches
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Move batch to device
            batch = batch.to(data.x.device)
            
            # Forward pass depending on model type
            if isinstance(model, VanillaGNN):
                # For vanilla GNN, convert to dense adjacency for the batch
                adj = to_dense_adj(batch.edge_index)[0]
                adj = adj + torch.eye(len(adj), device=adj.device)
                output = model(batch.x, adj)
            elif isinstance(model, (GCN, GAT, GCN_arxiv, SAGE_products)):
                output = model(batch.x, batch.edge_index)
            elif isinstance(model, MLP):
                output = model(batch.x)
            else:
                raise ValueError("Unknown model")
            
            # Compute loss on the batch
            # Get the original node indices in the batch
            batch_train_mask = batch.train_mask
            
            # Ensure we're only computing loss on training nodes
            if batch_train_mask.sum() > 0:
                loss = criterion(output[batch_train_mask], batch.y[batch_train_mask])
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Calculate training accuracy for this batch
                train_acc = (torch.argmax(output[batch_train_mask], dim=1) == batch.y[batch_train_mask]).sum().item() / batch_train_mask.sum().item()
                
                epoch_loss += loss.item()
                epoch_acc += train_acc
                num_batches += 1
        
        # Calculate average loss and accuracy for the epoch
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            
            training_losses.append(avg_loss)
            training_accuracies.append(avg_acc)
            
            if writer is not None:
                writer.add_scalar('Loss/train', avg_loss, epoch)
                writer.add_scalar('Accuracy/train', avg_acc, epoch)
            
            logging.info(f'Epoch {epoch:>3}| Train Loss: {avg_loss:.3f}| Train Accuracy: {avg_acc:.3f}')
    
    # Evaluate the model
    loss, acc = evaluate_with_minibatch(model, data, criterion, batch_size=batch_size, num_neighbors=num_neighbors)
    logging.info(f'Epoch {epochs-1:>3}| Validation Loss: {loss:.3f}, Validation Accuracy: {acc:.3f}')
    
    return loss, training_accuracies[-1], training_losses, training_accuracies

def evaluate_with_minibatch(model, data, criterion, batch_size=1024, num_neighbors=[10, 10, 10]):
    """Evaluate the model using mini-batches"""
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
            batch = batch.to(data.x.device)
            
            # Forward pass
            if isinstance(model, VanillaGNN):
                adj = to_dense_adj(batch.edge_index)[0]
                adj = adj + torch.eye(len(adj), device=adj.device)
                output = model(batch.x, adj)
            elif isinstance(model, (GCN, GAT, GCN_arxiv, SAGE_products)):
                output = model(batch.x, batch.edge_index)
            elif isinstance(model, MLP):
                output = model(batch.x)
            else:
                raise ValueError("Unknown model")
            
            # Get validation mask for this batch
            batch_val_mask = batch.val_mask
            
            if batch_val_mask.sum() > 0:
                # Compute loss
                loss = criterion(output[batch_val_mask], batch.y[batch_val_mask])
                total_loss += loss.item() * batch_val_mask.sum().item()
                
                # Compute accuracy
                pred = torch.argmax(output[batch_val_mask], dim=1)
                correct += (pred == batch.y[batch_val_mask]).sum().item()
                total += batch_val_mask.sum().item()
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / total if total > 0 else 0
    avg_acc = correct / total if total > 0 else 0
    
    return avg_loss, avg_acc

def test_with_minibatch(model, data, batch_size=1024, num_neighbors=[10, 10, 10]):
    """Test the model using mini-batches"""
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
            batch = batch.to(data.x.device)
            
            # Forward pass
            if isinstance(model, VanillaGNN):
                adj = to_dense_adj(batch.edge_index)[0]
                adj = adj + torch.eye(len(adj), device=adj.device)
                output = model(batch.x, adj)
            elif isinstance(model, (GCN, GAT, GCN_arxiv, SAGE_products)):
                output = model(batch.x, batch.edge_index)
            elif isinstance(model, MLP):
                output = model(batch.x)
            else:
                raise ValueError("Unknown model")
            
            # Get test mask for this batch
            batch_test_mask = batch.test_mask
            
            if batch_test_mask.sum() > 0:
                # Compute accuracy
                pred = torch.argmax(output[batch_test_mask], dim=1)
                correct += (pred == batch.y[batch_test_mask]).sum().item()
                total += batch_test_mask.sum().item()
    
    # Calculate accuracy
    acc = correct / total if total > 0 else 0
    
    return acc
