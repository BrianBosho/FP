import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse

class MemoryEfficientGNN(nn.Module):
    """
    Memory-efficient version of VanillaGNN that can handle both dense and sparse adjacency matrices.
    This is particularly useful for large graphs like OGBN-Products and OGBN-Arxiv where creating
    dense adjacency matrices would cause out-of-memory errors.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MemoryEfficientGNN, self).__init__()
        self.model_type = 'vanilla'  # For compatibility with existing code
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        
    def forward(self, x, adj_or_edge_index):
        """
        Forward pass that handles both dense adjacency matrices and sparse edge_index format.
        
        Args:
            x: Node features
            adj_or_edge_index: Either a dense adjacency matrix or a sparse edge_index
            
        Returns:
            Model output (log_softmax)
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x))
        
        # Check if we're given a dense adjacency matrix or sparse edge_index
        if isinstance(adj_or_edge_index, torch.Tensor) and adj_or_edge_index.dim() == 2 and adj_or_edge_index.shape[0] == adj_or_edge_index.shape[1]:
            # Dense adjacency matrix case
            x = torch.matmul(adj_or_edge_index, x)
        else:
            # Sparse edge_index case - perform sparse multiplication
            # This is equivalent to the dense matmul but memory-efficient
            edge_index = adj_or_edge_index
            row, col = edge_index
            
            # Create a sparse matrix and perform sparse matrix multiplication
            # We use the SparseTensor from torch_sparse for efficient operations
            adj_t = torch_sparse.SparseTensor(row=row, col=col, 
                                            value=torch.ones_like(row).float(),
                                            sparse_sizes=(x.size(0), x.size(0)))
            
            # Perform the sparse matrix multiplication
            x = torch_sparse.matmul(adj_t, x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)

class MemoryEfficientMLP(nn.Module):
    """
    Memory-efficient MLP for handling large datasets.
    Uses dropout and batch normalization for better training stability.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MemoryEfficientMLP, self).__init__()
        self.model_type = 'mlp'
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, edge_index=None):
        # edge_index parameter is ignored, included for API compatibility
        return F.log_softmax(self.layers(x), dim=1) 