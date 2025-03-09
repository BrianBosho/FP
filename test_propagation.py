import torch
from dataprocessing.data_utils import propagate_features
import traceback

def test_propagate_features():
    print("Testing propagate_features with random_walk mode...")
    
    # Create a simple test graph
    x = torch.randn(10, 5)  # 10 nodes, 5 features each
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9],
        [1, 0, 2, 1, 3, 2, 4, 3, 6, 5, 8, 7, 8]
    ]).long()
    
    # Create a mask where the first 5 nodes have known features
    mask = torch.zeros(10, dtype=torch.bool)
    mask[:5] = True
    
    device = 'cpu'
    
    # Test different propagation modes
    modes = ["random_walk", "page_rank", "diffusion", "adjacency", "propagation"]
    
    for mode in modes:
        print(f"\n=== Testing mode: {mode} ===")
        try:
            print(f"  Running propagate_features with mode={mode}...")
            result = propagate_features(x, edge_index, mask, device, mode=mode)
            print(f"  Success! Output shape: {result.shape}")
            print(f"  Output sample (first node, first 3 features): {result[0, :3]}")
            
            # Check if features were propagated to masked nodes
            if not mask.all():
                masked_nodes = (~mask).nonzero().squeeze()
                if len(masked_nodes) > 0:
                    first_masked = masked_nodes[0].item()
                    print(f"  First masked node ({first_masked}) features: {result[first_masked, :3]}")
                    
                    # Check if features were actually propagated (non-zero)
                    if torch.all(result[first_masked] == 0):
                        print("  WARNING: Features not propagated to masked nodes (all zeros)")
                    else:
                        print("  Features successfully propagated to masked nodes")
        except Exception as e:
            print(f"  Failed with error: {str(e)}")
            print("  Traceback:")
            traceback.print_exc()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    test_propagate_features() 