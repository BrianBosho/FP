# do all imports here
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, BatchNorm
from torch.nn import Linear, Dropout
import torch.nn as nn
from torch_geometric.utils import to_dense_adj


def get_model_config(cfg, model_type, dataset_name=None):
    """
    Extract model architecture hyperparameters from config.
    
    This function intelligently selects the right configuration based on:
    1. Global defaults
    2. Model-specific settings (e.g., GCN, GAT)
    3. Dataset-specific variants (e.g., GCN_arxiv for ogbn-arxiv, PubmedGAT for Pubmed)
    
    Args:
        cfg: Configuration dictionary
        model_type: Type of model (e.g., 'GCN', 'GAT')
        dataset_name: Optional dataset name for dataset-specific overrides
        
    Returns:
        Dictionary with model hyperparameters including:
        - hidden_dim: Hidden layer dimensions
        - num_layers: Number of GNN layers
        - dropout: Dropout rate
        - normalization: Type of normalization ('batch', 'layer', 'group', 'none')
        - num_heads: Number of attention heads (for GAT models)
    """
    # Default values (fallback if no config provided)
    default_config = {
        'hidden_dim': 16,
        'num_layers': 2,
        'dropout': 0.5,
        'normalization': 'none',
        'num_heads': 8,  # For GAT models
    }
    
    if cfg is None:
        return default_config
    
    # Get model architecture section from config
    model_arch = cfg.get('model_architecture', {})
    
    # Start with global defaults from config
    config = default_config.copy()
    if 'default' in model_arch:
        config.update(model_arch['default'])
    
    # Override with base model-specific settings
    if model_type in model_arch:
        config.update(model_arch[model_type])
    
    # Dataset-specific overrides (this allows using specialized configs for certain datasets)
    # For ogbn-arxiv: check for GCN_arxiv settings when using GCN
    if dataset_name == 'ogbn-arxiv' and model_type == 'GCN' and 'GCN_arxiv' in model_arch:
        config.update(model_arch['GCN_arxiv'])
    # For Pubmed: check for PubmedGAT settings when using GAT
    elif dataset_name == 'Pubmed' and model_type == 'GAT' and 'PubmedGAT' in model_arch:
        config.update(model_arch['PubmedGAT'])
    
    return config


class GCN(torch.nn.Module):
    """Graph Convolutional Network with configurable architecture"""
    def __init__(self, dim_in, dim_h, dim_out, num_layers=2, dropout=0.5, normalization='none'):
        super().__init__()
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.normalization = normalization
        
        # Build GCN layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(dim_in, dim_h))
        if normalization == 'batch':
            self.norms.append(torch.nn.BatchNorm1d(dim_h))
        elif normalization == 'layer':
            self.norms.append(torch.nn.LayerNorm(dim_h))
        elif normalization == 'group':
            self.norms.append(torch.nn.GroupNorm(8, dim_h))
        else:
            self.norms.append(torch.nn.Identity())
        
        # Hidden layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(dim_h, dim_h))
            if normalization == 'batch':
                self.norms.append(torch.nn.BatchNorm1d(dim_h))
            elif normalization == 'layer':
                self.norms.append(torch.nn.LayerNorm(dim_h))
            elif normalization == 'group':
                self.norms.append(torch.nn.GroupNorm(8, dim_h))
            else:
                self.norms.append(torch.nn.Identity())
        
        # Output layer
        self.convs.append(GCNConv(dim_h, dim_out))

    def forward(self, x, edge_index):
        # Input dropout
        x = F.dropout(x, training=self.training, p=0.0)
        
        # Hidden layers with normalization and dropout
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout_rate)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

# lets do a GCn for ogb-arxiv
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT_Arxiv(torch.nn.Module):
    """GAT for ogbn-arxiv with configurable architecture"""
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=40, 
                 dropout=0.5, num_layers=3, normalization='batch',
                 heads_hidden=4, heads_out=6):
        """
        Configurable GAT for ogbn-arxiv.

        Args:
            input_dim: Input feature dimension (default: 128)
            hidden_dim: Hidden layer dimension (default: 256)
            output_dim: Output dimension/number of classes (default: 40)
            dropout: Dropout rate (default: 0.5)
            num_layers: Number of layers (default: 3)
            normalization: Normalization type - 'batch', 'layer', 'group', or 'none' (default: 'batch')
            heads_hidden: Number of heads in hidden layers (default: 4)
            heads_out: Number of heads in output layer (default: 6)
        """
        nfeat = input_dim
        nhid = hidden_dim
        nclass = output_dim

        super(GAT_Arxiv, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # First layer
        self.convs.append(GATConv(nfeat, nhid, heads=heads_hidden, dropout=dropout))
        if normalization == 'batch':
            self.bns.append(nn.BatchNorm1d(nhid * heads_hidden))
        elif normalization == 'layer':
            self.bns.append(nn.LayerNorm(nhid * heads_hidden))
        elif normalization == 'group':
            self.bns.append(nn.GroupNorm(8, nhid * heads_hidden))
        else:
            self.bns.append(nn.Identity())

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(nhid * heads_hidden, nhid, heads=heads_hidden, dropout=dropout))
            if normalization == 'batch':
                self.bns.append(nn.BatchNorm1d(nhid * heads_hidden))
            elif normalization == 'layer':
                self.bns.append(nn.LayerNorm(nhid * heads_hidden))
            elif normalization == 'group':
                self.bns.append(nn.GroupNorm(8, nhid * heads_hidden))
            else:
                self.bns.append(nn.Identity())

        # Output layer (no concat)
        self.convs.append(GATConv(nhid * heads_hidden, nclass, heads=heads_out, concat=False, dropout=dropout))

        self.dropout = dropout
        self.num_layers = num_layers
        self.normalization = normalization
        self.dim_in = input_dim
        self.dim_h = hidden_dim
        self.dim_out = output_dim

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            if hasattr(bn, 'reset_parameters'):
                bn.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Apply all layers except last
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)


class GCN_arxiv(torch.nn.Module):
    """GCN for ogbn-arxiv with configurable architecture"""
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=40, dropout=0.5, num_layers=3, normalization='batch'):
        """
        Configurable GCN for ogbn-arxiv.
        
        Args:
            input_dim: Input feature dimension (default: 128)
            hidden_dim: Hidden layer dimension (default: 256)
            output_dim: Output dimension/number of classes (default: 40)
            dropout: Dropout rate (default: 0.5)
            num_layers: Number of layers (default: 3)
            normalization: Normalization type - 'batch', 'layer', 'group', or 'none' (default: 'batch')
        """
        nfeat = input_dim
        nhid = hidden_dim
        nclass = output_dim

        super(GCN_arxiv, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(nfeat, nhid))
        if normalization == 'batch':
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        elif normalization == 'layer':
            self.bns.append(torch.nn.LayerNorm(nhid))
        elif normalization == 'group':
            self.bns.append(torch.nn.GroupNorm(8, nhid))
        else:
            self.bns.append(torch.nn.Identity())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(nhid, nhid))
            if normalization == 'batch':
                self.bns.append(torch.nn.BatchNorm1d(nhid))
            elif normalization == 'layer':
                self.bns.append(torch.nn.LayerNorm(nhid))
            elif normalization == 'group':
                self.bns.append(torch.nn.GroupNorm(8, nhid))
            else:
                self.bns.append(torch.nn.Identity())
        
        # Output layer
        self.convs.append(GCNConv(nhid, nclass))

        self.dropout = dropout
        self.num_layers = num_layers
        self.normalization = normalization
        self.dim_in = input_dim
        self.dim_h = hidden_dim
        self.dim_out = output_dim

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            if hasattr(bn, 'reset_parameters'):
                bn.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        # Apply all layers except the last
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


# class GCN_arxiv(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
#         super(GCN_arxiv, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.gn1 = torch.nn.GroupNorm(8, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.gn2 = torch.nn.GroupNorm(8, hidden_dim)
#         self.conv3 = GCNConv(hidden_dim, output_dim)
#         self.dropout = dropout
#         self.dim_in = input_dim
#         self.dim_h = hidden_dim
#         self.dim_out = output_dim

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = self.gn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         x = self.conv2(x, edge_index)
#         x = self.gn2(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
        
#         x = self.conv3(x, edge_index)
#         return F.log_softmax(x, dim=1)

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv, BatchNorm

class GraphSAGEProducts(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, num_layers=3):
        """
        GraphSAGE model for OGBN-Products.

        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of hidden representations.
            output_dim (int): Number of classes (e.g., 47 for OGBN-Products).
            dropout (float): Dropout probability.
            num_layers (int): Number of layers (default is 3).
        """
        super(GraphSAGEProducts, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # First layer: input to hidden
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
        
        # Final layer: hidden to output
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Apply all layers except the last with activation, batch norm, and dropout.
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Final layer without activation/dropout
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    """Graph Attention Network with configurable architecture"""
    def __init__(self, dim_in, dim_h, dim_out, heads=8, dropout=0.6, num_layers=2, normalization='none'):
        super().__init__()
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.heads = heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.normalization = normalization
        
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATv2Conv(dim_in, dim_h, heads=heads))
        if normalization == 'batch':
            self.norms.append(torch.nn.BatchNorm1d(dim_h * heads))
        elif normalization == 'layer':
            self.norms.append(torch.nn.LayerNorm(dim_h * heads))
        elif normalization == 'group':
            self.norms.append(torch.nn.GroupNorm(8, dim_h * heads))
        else:
            self.norms.append(torch.nn.Identity())
        
        # Hidden layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(dim_h * heads, dim_h, heads=heads))
            if normalization == 'batch':
                self.norms.append(torch.nn.BatchNorm1d(dim_h * heads))
            elif normalization == 'layer':
                self.norms.append(torch.nn.LayerNorm(dim_h * heads))
            elif normalization == 'group':
                self.norms.append(torch.nn.GroupNorm(8, dim_h * heads))
            else:
                self.norms.append(torch.nn.Identity())
        
        # Output layer (single head for final prediction)
        self.convs.append(GATv2Conv(dim_h * heads, dim_out, heads=1))

    def forward(self, x, edge_index):
        # Apply all layers except the last
        for i in range(len(self.convs) - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
        
        # Output layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class PubmedGAT(torch.nn.Module):
    """GAT specifically tuned for Pubmed dataset with configurable architecture"""
    def __init__(self, dim_in, dim_h, dim_out, heads=8, dropout=0.6, num_layers=2, normalization='none'):
        super().__init__()
        self.dim_in = dim_in
        self.dim_h = dim_h if dim_h != 8 else 8  # Default to 8 for Pubmed
        self.dim_out = dim_out
        self.heads = heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.normalization = normalization
        
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATv2Conv(dim_in, self.dim_h, heads=heads))
        if normalization == 'batch':
            self.norms.append(torch.nn.BatchNorm1d(self.dim_h * heads))
        elif normalization == 'layer':
            self.norms.append(torch.nn.LayerNorm(self.dim_h * heads))
        elif normalization == 'group':
            self.norms.append(torch.nn.GroupNorm(8, self.dim_h * heads))
        else:
            self.norms.append(torch.nn.Identity())
        
        # Hidden layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(self.dim_h * heads, self.dim_h, heads=heads))
            if normalization == 'batch':
                self.norms.append(torch.nn.BatchNorm1d(self.dim_h * heads))
            elif normalization == 'layer':
                self.norms.append(torch.nn.LayerNorm(self.dim_h * heads))
            elif normalization == 'group':
                self.norms.append(torch.nn.GroupNorm(8, self.dim_h * heads))
            else:
                self.norms.append(torch.nn.Identity())
        
        # Output layer (8 heads for Pubmed, as in original)
        self.convs.append(GATv2Conv(self.dim_h * heads, dim_out, heads=8))

    def forward(self, x, edge_index):
        # Apply all layers except the last
        for i in range(len(self.convs) - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
        
        # Output layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


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
 