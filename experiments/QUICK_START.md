# Quick Start Guide for Enhanced Federated Learning

## Running a Single Experiment

```bash
# Basic experiment on Cora dataset with 3 clients
python FP/src/run_enhanced.py --dataset Cora --data_loading zero_hop --num_clients 3 --rounds 5

# Using k-hop neighbors with zero imputation
python FP/src/run_enhanced.py --dataset Cora --data_loading zero --hop 1 --num_clients 3 --rounds 5

# Using k-hop neighbors with page rank imputation
python FP/src/run_enhanced.py --dataset Cora --data_loading page_rank --hop 1 --num_clients 3 --rounds 5

# Run on Citeseer with memory optimizations
python FP/src/run_enhanced.py --dataset Citeseer --data_loading zero_hop --num_clients 3 --memory_efficient --rounds 10

# Run on OGBN-Arxiv with specialized settings
python FP/src/run_enhanced.py --dataset ogbn-arxiv --data_loading zero_hop --num_clients 3 --memory_efficient --rounds 10 --batch_size 1024
```

## Running Multiple Experiments

1. Edit the configuration file if needed:
   ```
   nano FP/experiments/experiment_config.yaml
   ```

2. Run the experiments:
   ```bash
   # Sequential mode
   python FP/src/run_experiments_enhanced.py --config FP/experiments/experiment_config.yaml
   
   # Parallel mode (faster)
   python FP/src/run_experiments_enhanced.py --config FP/experiments/experiment_config.yaml --parallel
   ```

3. For a quick test with minimal configuration:
   ```bash
   python FP/src/run_experiments_enhanced.py --config FP/experiments/quick_test_config.yaml
   ```

4. Find results in the `results` directory.

## Results Location and Format

All experiment results are saved in a timestamped directory under the `results` folder with a structure like:
```
results/results_20230615_120000/
├── Cora_zero_hop_GCN/
│   ├── results_Cora_zero_hop_GCN_20230615_120001.txt   # Text summary
│   ├── results_Cora_zero_hop_GCN_20230615_120001.json  # Full JSON data
│   ├── rounds_Cora_zero_hop_GCN_20230615_120001.csv    # Per-round metrics
│   ├── summary_Cora_zero_hop_GCN_20230615_120001.csv   # Overall summary
│   └── results_arrays_Cora_zero_hop_GCN_20230615_120001.csv  # Accuracy arrays
├── Citeseer_page_rank_GCN/
│   └── ... (similar files for this experiment)
└── all_experiments_summary.csv  # Combined results from all experiments
```

The `all_experiments_summary.csv` file in the root of the results directory combines data from all experiments for easy analysis. The file format is fully compatible with the original implementation for analysis scripts.

## Common Options

- `--dataset`: Dataset name (Cora, Citeseer, Pubmed, ogbn-arxiv, ogbn-products)
- `--data_loading`: Data loading method:
  - `zero_hop`: Basic partitioning without neighbors
  - `one_hop`, `two_hop`: Include k-hop neighbors with zero imputation
  - `zero`: K-hop with zero feature imputation
  - `page_rank`: K-hop with page rank imputation
  - `random_walk`: K-hop with random walk imputation
  - `diffusion`: K-hop with diffusion imputation
  - `propagation`: Propagation-based imputation
- `--hop`: Number of hops (used with k-hop methods)
- `--num_clients`: Number of federated clients
- `--beta`: Non-IID degree (0.1-1.0, lower is more non-IID)
- `--rounds`: Number of federated learning rounds
- `--memory_efficient`: Use memory optimizations (recommended for large datasets)
- `--batch_size`: Size of mini-batches (for large datasets)
- `--fulltraining`: Use full training mode

## Notes

- For OGBN-Products, use `--memory_efficient` and limit to 2 clients
- The models are automatically selected based on the dataset:
  - Cora/Citeseer/Pubmed: 2-layer GCN (hidden_dim=16)
  - OGBN-Arxiv: GCN_Arxiv model (hidden_dim=256)
  - OGBN-Products: SAGE_Products model (hidden_dim=256) 