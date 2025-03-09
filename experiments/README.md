# Enhanced Federated Learning Experiments

This directory contains configuration files and instructions for running federated learning experiments with the enhanced framework.

## Single Experiment

To run a single experiment:

```bash
cd /home/brian_bosho/FP
python FP/src/run_enhanced.py --dataset Cora --data_loading zero_hop --num_clients 3 --memory_efficient
```

## Multiple Experiments

To run multiple experiments based on a configuration file:

```bash
cd /home/brian_bosho/FP
python FP/src/run_experiments_enhanced.py --config FP/experiments/experiment_config.yaml
```

To run experiments in parallel:

```bash
python FP/src/run_experiments_enhanced.py --config FP/experiments/experiment_config.yaml --parallel
```

## Configuration Options

The enhanced framework automatically selects the appropriate model based on the dataset:

- **Small datasets** (Cora, Citeseer, Pubmed): 2-layer GCN with hidden dim=16
- **OGBN-Arxiv**: Specialized GCN_Arxiv model with hidden dim=256
- **OGBN-Products**: Specialized SAGE_Products model with hidden dim=256

### Common Parameters

- `--dataset`: Dataset name (case-insensitive)
- `--data_loading`: Data loading strategy (zero_hop, one_hop, two_hop, three_hop)
- `--num_clients`: Number of federated clients
- `--beta`: Parameter for Dirichlet distribution (controls non-IID degree)
- `--rounds`: Number of communication rounds
- `--batch_size`: Batch size for mini-batch training
- `--memory_efficient`: Use memory-efficient operations
- `--cuda`: Use GPU for training if available
- `--results_dir`: Directory to save results

## Example Configurations

### For Small Datasets (Cora, Citeseer, Pubmed)

```bash
python FP/src/run_enhanced.py --dataset Cora --data_loading zero_hop --num_clients 3 --rounds 20
```

### For Medium Datasets (OGBN-Arxiv)

```bash
python FP/src/run_enhanced.py --dataset ogbn-arxiv --data_loading zero_hop --num_clients 3 --rounds 20 --memory_efficient
```

### For Large Datasets (OGBN-Products)

```bash
python FP/src/run_enhanced.py --dataset ogbn-products --data_loading zero_hop --num_clients 2 --rounds 10 --batch_size 512 --memory_efficient
```

## Results

Results are saved in the `results` directory (or as specified by `--results_dir`) in JSON format. When running multiple experiments, a combined CSV and Excel file is also created.

Each result file contains:
- Configuration parameters
- Training and testing metrics for each round
- Final performance metrics

## Memory Management

For large datasets, the framework automatically:
- Limits the number of clients (max 2 for OGBN-Products)
- Reduces batch sizes
- Uses fewer epochs per round
- Applies aggressive garbage collection
- Uses mini-batch training with neighbor sampling 