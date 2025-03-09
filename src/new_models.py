import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second layer (logits output)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Example usage with a dataset (e.g., Cora)
# from torch_geometric.datasets import Planetoid
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# data = dataset[0]

# model = GCN(num_features=dataset.num_features, hidden_channels=16, num_classes=dataset.num_classes)
# print(model)


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset

class ThreeLayerGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super(ThreeLayerGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3: Output logits
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load Ogbn-Arxiv dataset
# dataset = PygNodePropPredDataset(name='ogbn-arxiv')
# data = dataset[0]

# model = ThreeLayerGCN(num_features=data.x.size(1), hidden_channels=64, num_classes=dataset.num_classes)
# print(model)
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First layer with activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second layer to produce logits
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load Ogbn-Products dataset
dataset = PygNodePropPredDataset(name='ogbn-products')
data = dataset[0]

model = GraphSAGE(num_features=data.x.size(1), hidden_channels=128, num_classes=dataset.num_classes)
print(model)
