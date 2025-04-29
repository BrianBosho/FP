import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, global_mean_pool, BatchNorm
from torch_geometric.utils import to_dense_adj


class GCN(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float = 0.5, NumLayers: int = 2
    ):
        """
        This constructor method initializes the GCN model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of GCN layers in the network.
        """
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, normalize=True, cached=True))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=True, cached=True))
        self.convs.append(GCNConv(nhid, nclass, normalize=True, cached=True))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        """
        This function is available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class SimpleGCN(torch.nn.Module):
    """Simple Graph Convolutional Network with two layers"""
    def __init__(self, dim_in, dim_h, dim_out):
        super(SimpleGCN, self).__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training, p=0.5)
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training, p=0.5)
        h = self.gcn2(h, edge_index)
        return F.log_softmax(h, dim=1)


class GCN_products(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the GCN_products model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of GCN layers in the network.
        """

        super(GCN_products, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, normalize=False))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, normalize=False))
        self.convs.append(GCNConv(nhid, nclass, normalize=False))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        """
        This function is available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class SAGE_products(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the Graph Sage model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of Graph Sage layers in the network
        """
        super(SAGE_products, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(nfeat, nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(SAGEConv(nhid, nhid))
        self.convs.append(SAGEConv(nhid, nclass))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        """
        This function is available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


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


class GCN_arxiv(torch.nn.Module):
    def __init__(
        self, nfeat: int, nhid: int, nclass: int, dropout: float, NumLayers: int
    ):
        """
        This constructor method initializes the Graph Sage model

        Arguments:
        nfeat: (int) - Number of input features
        nhid: (int) - Number of hidden features in the hidden layers of the network
        nclass: (int) - Number of output classes
        dropout: (float) - Dropout probability
        NumLayers: (int) - Number of Graph Sage layers in the network
        """
        super(GCN_arxiv, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(nhid))
        for _ in range(NumLayers - 2):
            self.convs.append(GCNConv(nhid, nhid, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.convs.append(GCNConv(nhid, nclass, cached=True))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        """
        This function is available to cater to weight initialization requirements as necessary.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        return None

    def forward(self, x: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        This function represents the forward pass computation of a GCN

        Arguments:
        x: (torch.Tensor) - Input feature tensor for the graph nodes
        adj_t: (SparseTensor) - Adjacency matrix of the graph

        Returns:
        The output of the forward pass, a PyTorch tensor

        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SimpleGCN_arxiv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(SimpleGCN_arxiv, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.gn1 = torch.nn.GroupNorm(8, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.gn2 = torch.nn.GroupNorm(8, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.gn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.gn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, NumLayers):
        super(GAT, self).__init__()
        # Note: This implementation needs to be updated as it's using GATConv which isn't imported
        self.dropout = dropout
        # Add proper implementation to be consistent with other models
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(nfeat, nhid, heads=8, dropout=dropout))
        for _ in range(NumLayers - 2):
            self.convs.append(GATv2Conv(nhid * 8, nhid, heads=8, dropout=dropout))
        self.convs.append(GATv2Conv(nhid * 8, nclass, heads=1, dropout=dropout))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)


class SimpleGAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super(SimpleGAT, self).__init__()
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
    