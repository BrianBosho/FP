import torch
import torch.nn.functional as F
from models import VanillaGNN, MLP, GCN, GAT
from torch_geometric.utils import to_dense_adj

# loga data instead of printing it
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def train(model, data, epochs, optimizer, criterion, writer):
    adjacency = to_dense_adj(data.edge_index)[0]
    # adjacency += torch.eye(len(adjacency))
    adjacency += torch.eye(len(adjacency), device=adjacency.device)


    # round_number = config["round_number"]-1
    training_losses = []
    training_accuracies = []
    torch.cuda.empty_cache()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        if isinstance(model, VanillaGNN):
            output = model(data.x, adjacency)
        elif isinstance(model, GCN) or isinstance(model, GAT):
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
        elif isinstance(model, GCN) or isinstance(model, GAT):
            output = model(data.x, data.edge_index)
        elif isinstance(model, MLP):
            output = model(data.x)
        else:
            raise ValueError("Unknown model")
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
        elif isinstance(model, GCN) or isinstance(model, GAT):
            output = model(data.x, data.edge_index)
        elif isinstance(model, MLP):
            output = model(data.x)
        else:
            raise ValueError("Unknown model")
        # out = model(data.x, data.edge_index)
        out = output
        _, pred = torch.max(out[data.test_mask], dim=1)
        correct = (pred == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
    return acc
