# do all imports here
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear, Dropout
# from utils import accuracy

import torch
import torch.nn as nn

from torch_geometric.utils import to_dense_adj


class GCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training, p=0.5)
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training, p=0.5)
        h = self.gcn2(h, edge_index)
        return F.log_softmax(h, dim=1)

# lets do a GCn for ogb-arxiv
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GCN_arxiv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN_arxiv, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)



class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return F.log_softmax(h, dim=1)


class VanillaGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VanillaGNN, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = torch.relu(self.conv1(x))
        x = torch.matmul(adj, x)
        x = self.conv2(x)
        return torch.log_softmax(x, dim=1)
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# Add this new class after the existing VanillaGNN class
class SparseVanillaGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SparseVanillaGNN, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # Apply the first linear layer
        x = torch.relu(self.conv1(x))
        
        # Use sparse matrix multiplication via edge_index
        # This is equivalent to adj @ x but uses sparse operations
        row, col = edge_index
        x_j = x[col]  # Target node features
        out = torch.zeros_like(x)
        out.index_add_(0, row, x_j)  # Aggregate messages using sparse operations
        
        # Apply the second linear layer
        out = self.conv2(out)
        return torch.log_softmax(out, dim=1)
 