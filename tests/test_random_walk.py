import torch
import traceback
import sys
from dataprocessing.data_utils import propagate_features

def test_random_walk():
    # Redirect output to a file
    with open('random_walk_test_output.txt', 'w') as f:
        try:
            # Create a simple test graph
            x = torch.randn(10, 5)
            f.write(f"Input tensor shape: {x.shape}\n")
            
            edge_index = torch.tensor([
                [0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9],
                [1, 0, 2, 1, 3, 2, 4, 3, 6, 5, 8, 7, 8]
            ]).long()
            f.write(f"Edge index shape: {edge_index.shape}\n")
            
            # Create a mask for the first 5 nodes
            mask = torch.zeros(10, dtype=torch.bool)
            mask[:5] = True
            f.write(f"Mask: {mask}\n")
            
            device = 'cpu'
            
            f.write("Testing random_walk mode...\n")
            result = propagate_features(x, edge_index, mask, device, mode='random_walk', alpha=0.2)
            
            f.write(f"Success! Output shape: {result.shape}\n")
            f.write(f"First node features: {result[0]}\n")
            
            # Check if features were propagated to masked nodes
            masked_features = result[mask]
            if torch.all(masked_features == 0):
                f.write("WARNING: No features propagated to masked nodes!\n")
            else:
                f.write("Features successfully propagated to masked nodes.\n")
                
        except Exception as e:
            f.write(f"Error: {str(e)}\n")
            traceback_str = traceback.format_exc()
            f.write(traceback_str)

if __name__ == "__main__":
    test_random_walk()
    print("Test completed. Check random_walk_test_output.txt for results.") 