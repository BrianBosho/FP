# Propagation vs Training Time Analysis Framework

## Overview
This framework analyzes the relationship between:
1. **Propagation time** (one-time preprocessing during data partitioning)
2. **Training time** (federated learning across multiple rounds)

## Data Structure Understanding

### Directory Organization
```
results/experiment_group/
├── propagation_stats/                    # Preprocessing logs
│   ├── prop_exp_TIMESTAMP_mode_beta_hop.json
│   └── ...
├── Dataset_loading_Model_betaX_clientsY/ # Training results
│   ├── results_Dataset_loading_Model_betaX_clientsY_TIMESTAMP.json
│   └── ...
```

### File Contents

**Propagation Stats** (`prop_exp_*.json`):
- Contains runtime for each of 10 clients
- Convergence status and iterations per client
- Happens ONCE per experiment during data setup

**Training Results** (`results_*.json`):
- Total duration for all federated learning rounds
- Performance metrics per round
- May contain incomplete logs (check round count)

## Key Metrics Framework

### Propagation Metrics
- `total_propagation_time`: Sum of all client runtimes
- `avg_propagation_time`: Average client runtime
- `convergence_rate`: % of clients that converged
- `propagation_std`: Standard deviation across clients

### Training Metrics  
- `training_duration_seconds`: Total FL training time
- `training_rounds`: Number of completed rounds
- `avg_global_result`: Model performance
- `is_complete_log`: Whether all rounds are logged

### Efficiency Ratios
- `propagation_overhead_pct`: (Prop time / Train time) × 100
- `propagation_to_training_ratio`: Prop time / Train time
- `time_per_round`: Training time / Number of rounds

## Analysis Results Summary

### Example: Pubmed Beta=10 Experiment
```
Propagation Analysis:
├── Total time: 21.87 seconds
├── Average per client: 2.19 seconds  
├── Convergence rate: 0% (0/10 clients)
└── Mode: diffusion, Beta: 10, PE: False

Training Analysis:
├── Duration: 768.14 seconds (12:48)
├── Rounds logged: 2/10 (incomplete)
├── Performance: 0.773
└── Estimated full duration: ~3,840 seconds

Efficiency:
├── Propagation overhead: 2.85% (of logged time)
├── Estimated overhead: ~0.57% (of full training)
└── Propagation is negligible preprocessing cost
```

## Key Insights

1. **Propagation is preprocessing**: Happens once during data partitioning
2. **Training is the main cost**: Multiple federated learning rounds
3. **Low overhead**: Propagation typically <1-5% of training time
4. **Convergence varies**: Depends on beta, alpha, and dataset properties
5. **Log completeness**: Check round counts for accurate ratios

## Recommendations

### For Efficiency Analysis:
1. Match propagation stats with training results by experiment parameters
2. Account for incomplete training logs when calculating ratios
3. Compare overhead across different beta values and PE settings

### For Convergence Analysis:
1. Track convergence rate vs beta values
2. Analyze iteration counts vs alpha parameters  
3. Compare diffusion vs adjacency propagation modes

### For Configuration Optimization:
1. Higher beta values often reduce propagation time
2. PE (positional encoding) may affect convergence
3. Different data loading options have varying propagation costs

## Implementation

The analysis framework can:
- Parse all experiment directories automatically
- Match propagation stats with training results
- Generate comprehensive efficiency reports
- Export results for further analysis in CSV/Excel format
- Create visualizations of overhead trends

This enables systematic analysis of when propagation preprocessing provides good trade-offs vs computational cost.