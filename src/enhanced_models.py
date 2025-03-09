import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm, GraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Union, Tuple, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedGCN(nn.Module):
    """
    Enhanced GCN model with configurable layers and dropout.
    Suitable for various graph datasets including OGBN-products and OGBN-arxiv.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, NumLayers=3):
        super(EnhancedGCN, self).__init__()
        self.model_type = 'gcn'
        self.dropout = dropout
        self.num_layers = NumLayers
        
        # Multiple layers for better feature extraction
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(nfeat, nhid))
        self.bns.append(BatchNorm(nhid))
        
        # Hidden layers
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid))
            self.bns.append(BatchNorm(nhid))
        
        # Output layer
        self.conv_out = GCNConv(nhid, nclass)
    
    def forward(self, x, edge_index):
        # Multiple GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv_out(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class EnhancedSAGE(nn.Module):
    """
    Enhanced GraphSAGE model optimized for large datasets like OGBN-products.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, NumLayers=2):
        super(EnhancedSAGE, self).__init__()
        self.model_type = 'sage'
        self.dropout = dropout
        self.num_layers = NumLayers
        
        # Multiple layers for better feature extraction
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(nfeat, nhid))
        self.bns.append(BatchNorm(nhid))
        
        # Hidden layers
        for _ in range(NumLayers - 2):
            self.convs.append(SAGEConv(nhid, nhid))
            self.bns.append(BatchNorm(nhid))
        
        # Output layer
        self.conv_out = SAGEConv(nhid, nclass)
    
    def forward(self, x, edge_index):
        # Multiple SAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv_out(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class EnhancedGAT(nn.Module):
    """
    Enhanced GAT model with multi-head attention.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, alpha=0.2, heads=4, NumLayers=3):
        super(EnhancedGAT, self).__init__()
        self.model_type = 'gat'
        self.dropout = dropout
        self.num_layers = NumLayers
        self.heads = heads
        
        # Multiple layers for better feature extraction
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(nfeat, nhid, heads=heads))
        self.bns.append(BatchNorm(nhid * heads))
        
        # Hidden layers
        for _ in range(NumLayers - 2):
            self.convs.append(GATConv(nhid * heads, nhid, heads=heads))
            self.bns.append(BatchNorm(nhid * heads))
        
        # Output layer
        self.conv_out = GATConv(nhid * heads, nclass, heads=1, concat=False)
    
    def forward(self, x, edge_index):
        # Multiple GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv_out(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class SAGE_Products(nn.Module):
    """
    GraphSAGE model specially optimized for OGBN-Products dataset.
    Based on the implementation from FedGCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, NumLayers=2):
        super(SAGE_Products, self).__init__()
        self.model_type = 'sage'
        self.dropout = dropout
        self.num_layers = NumLayers
        
        # Use nhid directly without multiplication to avoid dimension mismatches
        self.nhid = nhid
        
        # Multiple layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(nfeat, nhid))
        self.bns.append(BatchNorm(nhid))
        
        # Hidden layers
        for _ in range(NumLayers - 2):
            self.convs.append(SAGEConv(nhid, nhid))
            self.bns.append(BatchNorm(nhid))
        
        # Output layer
        self.conv_out = SAGEConv(nhid, nclass)
    
    def forward(self, x, edge_index):
        # Standard forward pass
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv_out(x, edge_index)
        
        return F.log_softmax(x, dim=1)

    def forward_batch(self, x, adjs):
        """
        Forward pass for mini-batch training using NeighborSampler.
        
        Args:
            x: Node features [N, in_channels]
            adjs: Adjacency information from NeighborSampler
        
        Returns:
            Output predictions
        """
        # Handle PyG's EdgeIndex object directly
        if hasattr(adjs, 'edge_index') and hasattr(adjs, 'size'):
            # This is a direct EdgeIndex object from newer PyG versions
            edge_index = adjs.edge_index
            target_nodes = x[:adjs.size[1]]  # Target nodes are first adjs.size[1] nodes
            
            # Apply the first layer (only layers_convs layer in a 2-layer model)
            x = self.convs[0](x, edge_index)
            x = self.bns[0](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Output layer
            x = self.conv_out(x, edge_index)
            
            # Return predictions only for target nodes
            return F.log_softmax(x[:adjs.size[1]], dim=1)
        
        # For list of tuples structure (older PyG versions)
        elif isinstance(adjs, list):
            for i, adj_tuple in enumerate(adjs):
                if i >= len(self.convs):
                    break
                    
                # Extract edge_index and size from the adj tuple
                # Handle different tuple formats
                if isinstance(adj_tuple, tuple):
                    if len(adj_tuple) == 3:
                        edge_index, _, size = adj_tuple
                    elif len(adj_tuple) == 2:
                        edge_index, size = adj_tuple
                    else:
                        raise ValueError(f"Unexpected adj_tuple length: {len(adj_tuple)}")
                elif hasattr(adj_tuple, 'edge_index') and hasattr(adj_tuple, 'size'):
                    edge_index = adj_tuple.edge_index
                    size = adj_tuple.size
                else:
                    raise ValueError(f"Unexpected adj_tuple type: {type(adj_tuple)}")
                    
                x_target = x[:size[1]]  # Target nodes are always placed first
                
                # Conv layer
                x = self.convs[i]((x, x_target), edge_index)
                
                # Normalization and activation
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Output layer - process the last layer differently
            if adjs:
                # Extract from the last adj
                if isinstance(adjs[-1], tuple):
                    if len(adjs[-1]) == 3:
                        edge_index, _, size = adjs[-1]
                    elif len(adjs[-1]) == 2:
                        edge_index, size = adjs[-1]
                    else:
                        raise ValueError(f"Unexpected adj_tuple length: {len(adjs[-1])}")
                elif hasattr(adjs[-1], 'edge_index') and hasattr(adjs[-1], 'size'):
                    edge_index = adjs[-1].edge_index
                    size = adjs[-1].size
                else:
                    raise ValueError(f"Unexpected adj_tuple type: {type(adjs[-1])}")
                    
                x_target = x[:size[1]]
                x = self.conv_out((x, x_target), edge_index)
            else:
                x = self.conv_out(x, None)
        
        # Handle direct tensor case
        elif isinstance(adjs, torch.Tensor):
            # Treat as direct edge_index tensor
            x = self.convs[0](x, adjs)
            x = self.bns[0](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Output layer
            x = self.conv_out(x, adjs)
        
        else:
            logging.error(f"Unexpected adjs type: {type(adjs)}")
            raise ValueError(f"Unexpected adjs type: {type(adjs)}")
        
        return F.log_softmax(x, dim=1)

class GCN_Arxiv(nn.Module):
    """Specialized GCN model for the OGBN-arxiv dataset."""
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, NumLayers=3):
        super(GCN_Arxiv, self).__init__()
        self.dropout = dropout
        self.model_type = 'gcn_arxiv'
        self.num_layers = NumLayers
        
        # Adjusted hidden dim for OGBN-Arxiv
        self.nhid = nhid
        
        # Multiple layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(nfeat, nhid))
        self.bns.append(BatchNorm(nhid))
        
        # Hidden layers
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid))
            self.bns.append(BatchNorm(nhid))
        
        # Output layer
        self.conv_out = GCNConv(nhid, nclass)
        
    def forward(self, x, edge_index):
        # Multiple GCN layers with residual connections
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv_out(x, edge_index)
        
        return F.log_softmax(x, dim=1)
        
    def forward_batch(self, x, adjs):
        """
        Forward pass for mini-batch training using NeighborSampler.
        
        Args:
            x: Node features [N, in_channels]
            adjs: Adjacency information from NeighborSampler
        
        Returns:
            Output predictions
        """
        # Handle PyG's EdgeIndex object directly
        if hasattr(adjs, 'edge_index') and hasattr(adjs, 'size'):
            # This is a direct EdgeIndex object from newer PyG versions
            edge_index = adjs.edge_index
            target_nodes = x[:adjs.size[1]]  # Target nodes are first adjs.size[1] nodes
            
            # Apply the first layer (only layers_convs layer in a 2-layer model)
            x = self.convs[0](x, edge_index)
            x = self.bns[0](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Output layer
            x = self.conv_out(x, edge_index)
            
            # Return predictions only for target nodes
            return F.log_softmax(x[:adjs.size[1]], dim=1)
        
        # For list of tuples structure (older PyG versions)
        elif isinstance(adjs, list):
            for i, adj_tuple in enumerate(adjs):
                if i >= len(self.convs):
                    break
                    
                # Extract edge_index and size from the adj tuple
                # Handle different tuple formats
                if isinstance(adj_tuple, tuple):
                    if len(adj_tuple) == 3:
                        edge_index, _, size = adj_tuple
                    elif len(adj_tuple) == 2:
                        edge_index, size = adj_tuple
                    else:
                        raise ValueError(f"Unexpected adj_tuple length: {len(adj_tuple)}")
                elif hasattr(adj_tuple, 'edge_index') and hasattr(adj_tuple, 'size'):
                    edge_index = adj_tuple.edge_index
                    size = adj_tuple.size
                else:
                    raise ValueError(f"Unexpected adj_tuple type: {type(adj_tuple)}")
                    
                x_target = x[:size[1]]  # Target nodes are always placed first
                
                # Conv layer
                x = self.convs[i]((x, x_target), edge_index)
                
                # Normalization and activation
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Output layer - process the last layer differently
            if adjs:
                # Extract from the last adj
                if isinstance(adjs[-1], tuple):
                    if len(adjs[-1]) == 3:
                        edge_index, _, size = adjs[-1]
                    elif len(adjs[-1]) == 2:
                        edge_index, size = adjs[-1]
                    else:
                        raise ValueError(f"Unexpected adj_tuple length: {len(adjs[-1])}")
                elif hasattr(adjs[-1], 'edge_index') and hasattr(adjs[-1], 'size'):
                    edge_index = adjs[-1].edge_index
                    size = adjs[-1].size
                else:
                    raise ValueError(f"Unexpected adj_tuple type: {type(adjs[-1])}")
                    
                x_target = x[:size[1]]
                x = self.conv_out((x, x_target), edge_index)
            else:
                x = self.conv_out(x, None)
        
        # Handle direct tensor case
        elif isinstance(adjs, torch.Tensor):
            # Treat as direct edge_index tensor
            x = self.convs[0](x, adjs)
            x = self.bns[0](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Output layer
            x = self.conv_out(x, adjs)
        
        else:
            logging.error(f"Unexpected adjs type: {type(adjs)}")
            raise ValueError(f"Unexpected adjs type: {type(adjs)}")
        
        return F.log_softmax(x, dim=1) 