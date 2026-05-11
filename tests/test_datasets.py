from src.dataprocessing.datasets import GraphDataset
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_loader = GraphDataset(device)

# Load Cora
print('Loading Cora...')
try:
    cora_data, cora_dataset = dataset_loader.load_dataset('Cora', device)
    print(f'Cora - Nodes: {cora_data.num_nodes}, Edges: {cora_data.edge_index.size(1)}')
    print(f'Cora - Features: {cora_data.x.shape}')
    print(f'Cora - Edge index shape: {cora_data.edge_index.shape}')
    print(f'Cora - Edge index type: {cora_data.edge_index.dtype}')
    print(f'Cora - Has isolated nodes: {(cora_data.edge_index[0].unique().size(0) < cora_data.num_nodes)}')
except Exception as e:
    print(f'Error loading Cora: {e}')
    import traceback
    traceback.print_exc()

# Load Computers
print('\nLoading Computers...')
try:
    comp_data, comp_dataset = dataset_loader.load_dataset('Computers', device)
    print(f'Computers - Nodes: {comp_data.num_nodes}, Edges: {comp_data.edge_index.size(1)}')
    print(f'Computers - Features: {comp_data.x.shape}')
    print(f'Computers - Edge index shape: {comp_data.edge_index.shape}')
    print(f'Computers - Edge index type: {comp_data.edge_index.dtype}')
    print(f'Computers - Has isolated nodes: {(comp_data.edge_index[0].unique().size(0) < comp_data.num_nodes)}')
except Exception as e:
    print(f'Error loading Computers: {e}')
    import traceback
    traceback.print_exc()
