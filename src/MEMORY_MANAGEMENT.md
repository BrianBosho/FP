# Memory Management for Large Datasets

This document provides guidance on how to handle memory issues when working with large graph datasets like OGBN-Products and OGBN-Arxiv in the enhanced federated learning framework.

## Memory Issues with Large Graphs

When working with large graph datasets, several memory-related issues can occur:

1. **Dense Adjacency Matrices**: Converting a sparse graph to a dense adjacency matrix can consume enormous amounts of memory. For example, OGBN-Products has over 2 million nodes, which would require terabytes of memory as a dense matrix.

2. **GPU Memory Limitations**: Even high-end GPUs have limited memory (typically 16-48GB), which is insufficient for large dense matrices.

3. **Parameter Synchronization Overhead**: In federated learning, transferring model parameters between clients and server can cause memory spikes.

## Memory-Efficient Solutions Implemented

We've implemented several strategies to address these memory issues:

1. **Specialized Models**: For large datasets, specialized models are automatically used:
   - OGBN-Arxiv always uses GCN_Arxiv
   - OGBN-Products always uses SAGE_Products

2. **Sparse Operations**: The framework avoids converting to dense matrices and uses sparse operations when possible.

3. **Memory Tracking**: Memory usage logging helps identify and fix memory bottlenecks.

4. **Automatic Memory Management**: The framework includes automatic checks for large datasets and adjusts the approach accordingly.

## Running With Large Datasets

To run federated learning on large datasets with memory efficiency, use the following command:

```bash
python src/run_enhanced.py \
    --dataset ogbn-products \
    --data_loading page_rank \
    --hop 1 \
    --num_clients 5 \
    --memory_efficient \
    --ray_memory 30 \
    --eval_every 2
```

Note: You don't need to specify a model type (`--model`) for OGBN-Arxiv or OGBN-Products as the appropriate models are automatically selected.

### Important Command-Line Arguments for Memory Management

- `--memory_efficient`: Enables memory-efficient PyTorch settings
- `--mixed_precision`: Uses mixed precision (FP16) training to reduce memory usage
- `--ray_memory`: Sets memory limit for Ray in GB
- `--redis_memory`: Sets memory limit for Redis in Ray in GB
- `--eval_every`: Evaluates only every N rounds to save memory
- `--cpu`: Forces CPU usage if GPU memory is insufficient

## Tips for Handling Memory Issues

1. **Reduce Client Count**: Use fewer clients (3-5) when working with large datasets.

2. **Reduce Hop Count**: Use smaller hop values (1-2) for k-hop neighborhood methods.

3. **Monitor Memory Usage**: The framework logs memory usage; watch for excessive memory consumption.

4. **Clear Cache Regularly**: The framework automatically calls `torch.cuda.empty_cache()` to free unused memory.

5. **Use CPU for Very Large Graphs**: If your GPU memory is insufficient, use the `--cpu` flag to run on CPU instead.

## Debugging Memory Issues

If you're still experiencing memory issues:

1. Enable detailed memory logging:
   ```python
   torch.cuda.memory_summary(device=None, abbreviated=False)
   ```

2. Reduce the size of your dataset for testing:
   ```bash
   python src/run_enhanced.py --dataset ogbn-products --num_clients 2 --memory_efficient
   ```

3. Check for memory leaks by observing if memory usage consistently increases across rounds.

4. For extreme cases, you may need to implement custom data sharding or mini-batch training for the very largest graphs.

## Dataset-Specific Recommendations

### OGBN-Products
- Automatically uses SAGE_Products model
- Maximum 5 clients recommended
- 1-hop neighborhood recommended
- Use `--memory_efficient` flag

### OGBN-Arxiv
- Automatically uses GCN_Arxiv model
- Maximum 10 clients recommended
- 1-2 hop neighborhood recommended
- Use `--memory_efficient` flag

## Implementation Details

The memory-efficient implementation includes:

1. **Automatic Model Selection**: For large datasets, specialized models are automatically used regardless of the model type specified.

2. **Sparse Operations**: The code avoids dense adjacency matrices for large graphs and uses sparse operations.

3. **Memory Management**: Automatic memory cleanup and efficient resource allocation are implemented throughout the codebase.

4. **Optimized Hyperparameters**: The framework automatically adjusts hyperparameters like number of epochs for large datasets. 