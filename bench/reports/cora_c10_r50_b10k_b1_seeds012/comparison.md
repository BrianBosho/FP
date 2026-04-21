# Comparison: `fedavg_weighted` vs `baseline`

Rows: 2 experiment key(s) in common.

| Experiment key | Metric | Baseline (mean ± std, n) | Candidate (mean ± std, n) | Δ (cand − base) | Welch t | p-value | Cohen's d |
|---|---|---|---|---|---|---|---|
| dataset=Cora, data_loading_option=zero_hop, model_type=GCN, num_clients=10, beta=1, hop=1, use_pe=False, fulltraining_flag=False | `avg_global_acc` | 0.7770 ± 0.0156 (n=3) | 0.7750 ± 0.0192 (n=3) | -0.0020 | 0.140 | 0.896 | -0.114 |
| dataset=Cora, data_loading_option=zero_hop, model_type=GCN, num_clients=10, beta=1, hop=1, use_pe=False, fulltraining_flag=False | `avg_client_acc` | 0.6699 ± 0.0094 (n=3) | 0.6641 ± 0.0103 (n=3) | -0.0058 | 0.720 | 0.512 | -0.588 |
| dataset=Cora, data_loading_option=zero_hop, model_type=GCN, num_clients=10, beta=10000, hop=1, use_pe=False, fulltraining_flag=False | `avg_global_acc` | 0.7630 ± 0.0090 (n=3) | 0.7647 ± 0.0076 (n=3) | +0.0017 | -0.245 | 0.819 | 0.200 |
| dataset=Cora, data_loading_option=zero_hop, model_type=GCN, num_clients=10, beta=10000, hop=1, use_pe=False, fulltraining_flag=False | `avg_client_acc` | 0.6049 ± 0.0135 (n=3) | 0.6054 ± 0.0106 (n=3) | +0.0005 | -0.052 | 0.961 | 0.043 |

## Per-seed wall time (seconds)

| Experiment key | Baseline seeds (mean) | Candidate seeds (mean) |
|---|---|---|
| dataset=Cora, data_loading_option=zero_hop, model_type=GCN, num_clients=10, beta=1, hop=1, use_pe=False, fulltraining_flag=False | 51.94 (n=3) | 52.19 (n=3) |
| dataset=Cora, data_loading_option=zero_hop, model_type=GCN, num_clients=10, beta=10000, hop=1, use_pe=False, fulltraining_flag=False | 56.02 (n=3) | 56.21 (n=3) |
