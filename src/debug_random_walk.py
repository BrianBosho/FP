import torch
import traceback
from dataprocessing.propagation_functions import sparse_random_walk_with_restarts
from torch_sparse import SparseTensor

def debug_random_walk():
    print("Debugging sparse_random_walk_with_restarts function...")
    
    # Create a simple test graph
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9],
        [1, 0, 2, 1, 3, 2, 4, 3, 6, 5, 8, 7, 8]
    ]).long()
    
    num_nodes = 10
    device = 'cpu'
    
    try:
        print("Calling sparse_random_walk_with_restarts...")
        result = sparse_random_walk_with_restarts(edge_index, num_nodes, device)
        print(f"Success! Result type: {type(result)}")
        
        # Check if result is a SparseTensor
        if isinstance(result, SparseTensor):
            print(f"SparseTensor with shape: {result.sparse_sizes()}")
            print(f"Number of non-zero entries: {result.nnz()}")
            
            # Convert to dense for inspection
            dense_result = result.to_dense()
            print(f"Dense shape: {dense_result.shape}")
            print(f"First row: {dense_result[0]}")
            
            # Check if it's a valid transition matrix (rows sum to 1)
            row_sums = dense_result.sum(dim=1)
            print(f"Row sums (should be close to 1): {row_sums}")
            
            # Check if we can use it for matrix multiplication
            x = torch.randn(num_nodes, 5)
            try:
                from torch_sparse import matmul
                propagated = matmul(result, x)
                print(f"Matrix multiplication successful, result shape: {propagated.shape}")
            except Exception as e:
                print(f"Matrix multiplication failed: {str(e)}")
                traceback.print_exc()
        else:
            print(f"Result is not a SparseTensor but a {type(result)}")
    
    except Exception as e:
        print(f"Function call failed with error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_random_walk() 