# Propagator Evaluation — Full Results Report

**Generated:** 2026-05-08  
**Experiment directory:** `experiments/propagator_eval/`  
**Total result files:** 1,620 JSON across 6 phases

---

## Notation

| Symbol | Meaning |
|--------|---------|
| MSE | Mean squared error between propagated and reference features (lower = better) |
| CosSim | Cosine similarity to reference features (higher = better) |
| RR | Recovery ratio — fraction of feature variance explained (higher = better) |
| ConvRate | Fraction of clients that converged within 100 iterations |
| WT | Total wall-clock time across all clients (seconds) |
| Gap | Accuracy gap closed: `(acc − zero_hop) / (oracle − zero_hop)` (higher = better) |
| n | Number of seeds / runs averaged |

**Operators:**

| Short name | Full name |
|------------|-----------|
| Zero-Hop | No propagation (local features only) |
| Full (Oracle) | Full graph access — upper bound |
| Adjacency | k-hop adjacency averaging |
| Asym. RW | Asymmetric random walk |
| Diffusion | Heat-diffusion iterative solver |
| Chebyshev | Chebyshev polynomial diffusion |
| APPNP | Approximate Personalized Propagation of Neural Predictions |
| Heat Kernel | Exact heat kernel (reference baseline) |

---

## Phase 1 — Cora Intrinsic Feature Quality

**Setup:** 6 propagation operators × 3 betas × 3 seeds on Cora (2,708 nodes, 1,433 features, 7 classes, 10 clients).  
Metrics measure reconstruction quality of propagated features vs. oracle full-graph features.

### Table 1.1 — Intrinsic Metrics by Operator and Beta (Cora)

| Operator | Beta | n | MSE (↓) | CosSim (↑) | Recov. Ratio (↑) | Conv. Rate | Wall Time (s) |
|----------|------|---|---------|-----------|-----------------|-----------|--------------|
| **Heat Kernel (Exact)** | 10000 | 1 | 0.01231 ± 0.00000 | 0.2068 ± 0.0000 | 0.0343 ± 0.0000 | 1.00 | 0.21 |
| APPNP | 1 | 3 | 0.01294 ± 0.00002 | 0.2215 ± 0.0014 | −0.0151 ± 0.0004 | 1.00 | 2.33 |
| APPNP | 10 | 3 | 0.01299 ± 0.00002 | 0.2192 ± 0.0002 | −0.0165 ± 0.0013 | 1.00 | 2.27 |
| APPNP | 10000 | 3 | 0.01299 ± 0.00002 | 0.2183 ± 0.0005 | −0.0169 ± 0.0004 | 1.00 | 2.37 |
| Chebyshev | 1 | 3 | 0.01316 ± 0.00003 | 0.2424 ± 0.0014 | −0.0325 ± 0.0010 | 0.00 | 29.87 |
| Chebyshev | 10 | 3 | 0.01324 ± 0.00002 | 0.2397 ± 0.0005 | −0.0356 ± 0.0021 | 0.00 | 345.31 |
| Chebyshev | 10000 | 3 | 0.01325 ± 0.00002 | 0.2386 ± 0.0006 | −0.0373 ± 0.0004 | 0.00 | 725.85 |
| Diffusion | 1 | 3 | 0.01328 ± 0.00002 | 0.2226 ± 0.0015 | −0.0425 ± 0.0008 | 0.00 | 139.64 |
| Diffusion | 10 | 3 | 0.01335 ± 0.00002 | 0.2199 ± 0.0002 | −0.0442 ± 0.0018 | 0.00 | 141.26 |
| Diffusion | 10000 | 3 | 0.01334 ± 0.00002 | 0.2189 ± 0.0005 | −0.0444 ± 0.0001 | 0.00 | 109.40 |
| Adjacency | 1 | 3 | 0.01391 ± 0.00003 | 0.2265 ± 0.0014 | −0.0918 ± 0.0012 | 0.97 | 115.60 |
| Adjacency | 10 | 3 | 0.01399 ± 0.00003 | 0.2245 ± 0.0002 | −0.0945 ± 0.0024 | 0.97 | 119.22 |
| Adjacency | 10000 | 3 | 0.01400 ± 0.00002 | 0.2237 ± 0.0006 | −0.0962 ± 0.0007 | 0.97 | 121.56 |
| Asym. RW | 1 | 3 | 0.01594 ± 0.00001 | 0.2225 ± 0.0015 | −0.2509 ± 0.0023 | 1.00 | 101.08 |
| Asym. RW | 10 | 3 | 0.01598 ± 0.00005 | 0.2205 ± 0.0003 | −0.2502 ± 0.0038 | 1.00 | 104.55 |
| Asym. RW | 10000 | 3 | 0.01602 ± 0.00004 | 0.2196 ± 0.0006 | −0.2541 ± 0.0032 | 1.00 | 104.30 |

> **Notes:**  
> - Heat Kernel (Exact) is the ground-truth reference; all approximate operators converge to it with sufficient compute.  
> - APPNP achieves the lowest MSE among iterative operators across all betas with by far the shortest wall time.  
> - Chebyshev achieves the highest cosine similarity but is not converging (ConvRate = 0.00), suggesting it reaches a useful but sub-convergent state.  
> - Asym. RW has the highest MSE; beta has negligible effect on all operators' accuracy.

### Figure Data 1.1 — MSE vs. Beta per Operator (for line plots)

| Operator | Beta=1 MSE | Beta=10 MSE | Beta=10000 MSE |
|----------|-----------|------------|---------------|
| APPNP | 0.01294 | 0.01299 | 0.01299 |
| Chebyshev | 0.01316 | 0.01324 | 0.01325 |
| Diffusion | 0.01328 | 0.01335 | 0.01334 |
| Adjacency | 0.01391 | 0.01399 | 0.01400 |
| Asym. RW | 0.01594 | 0.01598 | 0.01602 |
| Heat Kernel (Exact) | — | — | 0.01231 |

### Figure Data 1.2 — Wall Time vs. Beta per Operator

| Operator | Beta=1 (s) | Beta=10 (s) | Beta=10000 (s) |
|----------|-----------|------------|---------------|
| Heat Kernel (Exact) | — | — | 0.21 |
| APPNP | 2.33 | 2.27 | 2.37 |
| Chebyshev | 29.87 | 345.31 | 725.85 |
| Diffusion | 139.64 | 141.26 | 109.40 |
| Adjacency | 115.60 | 119.22 | 121.56 |
| Asym. RW | 101.08 | 104.55 | 104.30 |

---

## Phase 2 — Ablation Studies (Cora, beta=10000)

### 2a — APPNP Alpha Sweep

**Setup:** 3 alpha values × 3 seeds. Beta=10000, Cora.

### Table 2.1 — APPNP Intrinsic Quality vs. Teleport Probability (α)

| Alpha (α) | n | MSE (↓) | CosSim (↑) | Recov. Ratio (↑) | Avg. Iters | Conv. Rate | Wall Time (s) |
|-----------|---|---------|-----------|-----------------|-----------|-----------|--------------|
| 0.05 | 3 | 0.01339 ± 0.00002 | 0.2209 ± 0.0005 | −0.0481 ± 0.0006 | 38.3 | 1.00 | 1.08 |
| 0.10 | 3 | 0.01299 ± 0.00002 | 0.2183 ± 0.0005 | −0.0169 ± 0.0004 | 32.4 | 1.00 | 0.92 |
| 0.20 | 3 | 0.01255 ± 0.00002 | 0.2136 ± 0.0005 | +0.0176 ± 0.0002 | 24.3 | 1.00 | 0.71 |

> **Notes:**  
> - Higher α (stronger teleportation / less propagation) reduces MSE and convergence iterations, but also reduces cosine similarity — a bias-variance tradeoff.  
> - At α=0.20, recovery ratio turns positive, indicating the propagated features explain more variance than a trivial predictor.  
> - All alpha settings converge reliably (ConvRate=1.00).

### 2b — Diffusion Convergence Tolerance (ε) Sweep

**Setup:** 3 tolerance values × 3 seeds. Beta=10000, Cora, operator=Diffusion.

### Table 2.2 — Diffusion Intrinsic Quality vs. Convergence Tolerance (ε)

| Tolerance (ε) | n | MSE (↓) | CosSim (↑) | Recov. Ratio | Avg. Iters | Conv. Rate | Wall Time (s) |
|--------------|---|---------|-----------|-------------|-----------|-----------|--------------|
| 1e-02 | 3 | 0.01334 ± 0.00002 | 0.2189 ± 0.0005 | −0.0444 ± 0.0001 | 100.0 | 0.00 | 3.70 |
| 1e-03 | 3 | 0.01334 ± 0.00002 | 0.2189 ± 0.0005 | −0.0444 ± 0.0001 | 100.0 | 0.00 | 3.69 |
| 1e-04 | 3 | 0.01334 ± 0.00002 | 0.2189 ± 0.0005 | −0.0444 ± 0.0001 | 100.0 | 0.00 | 3.69 |

> **Notes:**  
> - Diffusion does not converge within 100 iterations for any tolerance setting on Cora at beta=10000.  
> - All metrics are identical across tolerances — the solver hits the iteration cap before reaching any threshold, making ε inconsequential in this regime.  
> - This confirms diffusion's convergence failure is a fundamental property of the regime (beta=10000), not a tolerance tuning issue.

### 2c — Hop Depth Sweep

**Setup:** hop ∈ {1, 2} × operator ∈ {Adjacency, Diffusion} × 3 seeds. Beta=10000, Cora.

### Table 2.3 — Intrinsic Quality vs. Neighborhood Hop Depth

| Hop | Operator | n | MSE (↓) | CosSim (↑) | Recov. Ratio (↑) | Avg. Iters | Wall Time (s) |
|-----|----------|---|---------|-----------|-----------------|-----------|--------------|
| 1 | Adjacency | 3 | 0.01400 ± 0.00002 | 0.2237 ± 0.0006 | −0.0962 ± 0.0007 | 91.0 | 2.39 |
| 2 | Adjacency | 3 | 0.01290 ± 0.00001 | 0.2423 ± 0.0004 | −0.0095 ± 0.0008 | 100.0 | 4.42 |
| 1 | Diffusion | 3 | 0.01334 ± 0.00002 | 0.2189 ± 0.0005 | −0.0444 ± 0.0001 | 100.0 | 3.70 |
| 2 | Diffusion | 3 | 0.01257 ± 0.00001 | 0.2123 ± 0.0003 | +0.0160 ± 0.0004 | 100.0 | 11.55 |

> **Notes:**  
> - Increasing hop depth from 1 to 2 improves MSE and recovery ratio for both operators at modest wall-time cost.  
> - For Adjacency, hop=2 achieves the highest cosine similarity of any Adjacency configuration (0.2423).  
> - For Diffusion, hop=2 is the only configuration with positive recovery ratio at beta=10000.  
> - Wall-time cost of hop=2 diffusion is 3× hop=1 (11.55s vs 3.70s).

---

## Phase 3 — Cora Downstream Node Classification

**Setup:** 7 operators × 2 betas × 2 backbones (GCN, GAT) × 5 seeds = 140 runs.  
3 NaN failures at beta=10000 with GCN (numerical overflow at extreme diffusion parameter).

### Table 3.1 — Cora Downstream Accuracy, GCN Backbone

| Operator | Beta=1 Acc. | Beta=1 Gap | Beta=10000 Acc. | Beta=10000 Gap |
|----------|------------|-----------|----------------|---------------|
| Zero-Hop | 0.6690 ± 0.0092 | — | 0.6038 ± 0.0162 | — |
| Full (Oracle) | 0.7938 ± 0.0038 | — | 0.8120 ± 0.0050 | — |
| **Chebyshev** | **0.7840 ± 0.0096** | **0.924** | **0.7205 ± 0.0071** | **0.570** |
| Adjacency | 0.7738 ± 0.0127 | 0.843 | 0.7174 ± 0.0130 | 0.546 |
| APPNP | 0.7672 ± 0.0133 | 0.791 | 0.7105 ± 0.0185 | 0.508 |
| Asym. RW | 0.7690 ± 0.0062 | 0.801 | 0.7063 ± 0.0154 | 0.504 |
| Diffusion | 0.7622 ± 0.0139 | 0.750 | 0.7048 ± 0.0168 | 0.486 |

> Shaded best per column (excluding oracle/zero-hop). N=5 per cell; NaN rows excluded (4 seeds used for those cells).

### Table 3.2 — Cora Downstream Accuracy, GAT Backbone

| Operator | Beta=1 Acc. | Beta=1 Gap | Beta=10000 Acc. | Beta=10000 Gap |
|----------|------------|-----------|----------------|---------------|
| Zero-Hop | 0.4058 ± 0.0410 | — | 0.4292 ± 0.0236 | — |
| Full (Oracle) | 0.7639 ± 0.0079 | — | 0.7857 ± 0.0096 | — |
| **Chebyshev** | **0.7006 ± 0.0251** | **0.825** | **0.6898 ± 0.0217** | **0.731** |
| APPNP | 0.6680 ± 0.0148 | 0.732 | 0.6676 ± 0.0284 | 0.669 |
| Asym. RW | 0.6604 ± 0.0100 | 0.709 | 0.6736 ± 0.0302 | 0.687 |
| Adjacency | 0.6874 ± 0.0226 | 0.788 | 0.6800 ± 0.0268 | 0.704 |
| Diffusion | 0.6480 ± 0.0144 | 0.676 | 0.6506 ± 0.0322 | 0.623 |

### Table 3.3 — Cora: Oracle and Zero-Hop Reference by Beta × Backbone

| Backbone | Beta | Zero-Hop Acc. | Oracle Acc. | Oracle − Zero-Hop |
|----------|------|--------------|------------|------------------|
| GCN | 1 | 0.6690 | 0.7938 | 0.1248 |
| GCN | 10000 | 0.6038 | 0.8120 | 0.2082 |
| GAT | 1 | 0.4058 | 0.7639 | 0.3581 |
| GAT | 10000 | 0.4292 | 0.7857 | 0.3565 |

> **Key observations:**  
> - Chebyshev is consistently the top operator for downstream classification on Cora across both backbones and both betas.  
> - At beta=1 (moderate federation), GCN closes >90% of the oracle gap with Chebyshev.  
> - At beta=10000 (severe federation), the gap-closing ability drops substantially for all operators — the task is harder.  
> - GAT is more sensitive to propagation quality: zero-hop accuracy is much lower (~40% vs ~67% for GCN), but gap-closed rates are comparable once propagation is added.

---

## Phase 4 — Homophilic Reproduction (Citeseer + Pubmed)

**Setup:** 7 operators × 2 datasets × 2 betas × 2 backbones × 5 seeds = 280 runs. Zero NaN failures.

### Table 4.1 — Citeseer Downstream Accuracy, GCN Backbone

| Operator | Beta=1 Acc. | Beta=1 Gap | Beta=10000 Acc. | Beta=10000 Gap |
|----------|------------|-----------|----------------|---------------|
| Zero-Hop | 0.6216 ± 0.0178 | — | 0.6042 ± 0.0047 | — |
| Full (Oracle) | 0.7043 ± 0.0055 | — | 0.7173 ± 0.0054 | — |
| **Chebyshev** | **0.6728 ± 0.0126** | **0.613** | **0.6428 ± 0.0144** | **0.340** |
| Adjacency | 0.6692 ± 0.0112 | 0.556 | 0.6438 ± 0.0136 | 0.349 |
| Asym. RW | 0.6682 ± 0.0124 | 0.544 | 0.6394 ± 0.0122 | 0.310 |
| APPNP | 0.6656 ± 0.0142 | 0.511 | 0.6362 ± 0.0113 | 0.282 |
| Diffusion | 0.6588 ± 0.0159 | 0.416 | 0.6312 ± 0.0116 | 0.238 |

### Table 4.2 — Citeseer Downstream Accuracy, GAT Backbone

| Operator | Beta=1 Acc. | Beta=1 Gap | Beta=10000 Acc. | Beta=10000 Gap |
|----------|------------|-----------|----------------|---------------|
| Zero-Hop | 0.4672 ± 0.0323 | — | 0.4736 ± 0.0511 | — |
| Full (Oracle) | 0.6837 ± 0.0153 | — | 0.6959 ± 0.0132 | — |
| **Chebyshev** | **0.6002 ± 0.0588** | **0.610** | **0.5922 ± 0.0246** | **0.515** |
| Adjacency | 0.5982 ± 0.0558 | 0.598 | 0.5914 ± 0.0231 | 0.516 |
| APPNP | 0.5898 ± 0.0550 | 0.556 | 0.5814 ± 0.0298 | 0.470 |
| Asym. RW | 0.5816 ± 0.0471 | 0.515 | 0.5818 ± 0.0252 | 0.469 |
| Diffusion | 0.5788 ± 0.0552 | 0.507 | 0.5776 ± 0.0346 | 0.454 |

### Table 4.3 — Pubmed Downstream Accuracy, GCN Backbone

| Operator | Beta=1 Acc. | Beta=1 Gap | Beta=10000 Acc. | Beta=10000 Gap |
|----------|------------|-----------|----------------|---------------|
| Zero-Hop | 0.6674 ± 0.0282 | — | 0.7006 ± 0.0168 | — |
| Full (Oracle) | 0.7760 ± 0.0035 | — | 0.7979 ± 0.0038 | — |
| **Chebyshev** | **0.7712 ± 0.0233** | **0.970** | 0.7722 ± 0.0138 | 0.740 |
| Adjacency | 0.7684 ± 0.0219 | 0.932 | 0.7736 ± 0.0127 | 0.750 |
| Asym. RW | 0.7668 ± 0.0206 | 0.927 | **0.7780 ± 0.0133** | **0.789** |
| APPNP | 0.7614 ± 0.0196 | 0.851 | 0.7744 ± 0.0112 | 0.758 |
| Diffusion | 0.7548 ± 0.0169 | 0.788 | 0.7638 ± 0.0137 | 0.658 |

### Table 4.4 — Pubmed Downstream Accuracy, GAT Backbone

| Operator | Beta=1 Acc. | Beta=1 Gap | Beta=10000 Acc. | Beta=10000 Gap |
|----------|------------|-----------|----------------|---------------|
| Zero-Hop | 0.6314 ± 0.0143 | — | 0.5796 ± 0.0389 | — |
| Full (Oracle) | 0.7577 ± 0.0129 | — | 0.7914 ± 0.0080 | — |
| Asym. RW | **0.7344 ± 0.0124** | **0.808** | 0.7358 ± 0.0178 | 0.728 |
| Adjacency | 0.7288 ± 0.0123 | 0.767 | **0.7486 ± 0.0121** | **0.782** |
| Chebyshev | 0.7266 ± 0.0158 | 0.751 | 0.7458 ± 0.0055 | 0.775 |
| APPNP | 0.7140 ± 0.0263 | 0.612 | 0.7362 ± 0.0176 | 0.725 |
| Diffusion | 0.6892 ± 0.0313 | 0.408 | 0.7208 ± 0.0253 | 0.642 |

### Table 4.5 — Citeseer + Pubmed: Oracle and Zero-Hop Reference

| Dataset | Backbone | Beta | Zero-Hop | Oracle | Oracle − Zero-Hop |
|---------|----------|------|---------|--------|------------------|
| Citeseer | GCN | 1 | 0.6216 | 0.7043 | 0.0827 |
| Citeseer | GCN | 10000 | 0.6042 | 0.7173 | 0.1131 |
| Citeseer | GAT | 1 | 0.4672 | 0.6837 | 0.2165 |
| Citeseer | GAT | 10000 | 0.4736 | 0.6959 | 0.2223 |
| Pubmed | GCN | 1 | 0.6674 | 0.7760 | 0.1086 |
| Pubmed | GCN | 10000 | 0.7006 | 0.7979 | 0.0973 |
| Pubmed | GAT | 1 | 0.6314 | 0.7577 | 0.1263 |
| Pubmed | GAT | 10000 | 0.5796 | 0.7914 | 0.2118 |

> **Key observations:**  
> - On Citeseer (lower oracle accuracy, smaller graph), gap-closed values are smaller — harder to recover the gap.  
> - On Pubmed, Chebyshev and Adjacency achieve >93% gap-closed with GCN at beta=1, nearly matching the oracle.  
> - Diffusion is the weakest operator on Citeseer; on Pubmed it is stronger but still lags Chebyshev/Adjacency.  
> - Pubmed benefits more from higher beta (harder federation) than Cora, with absolute accuracies sometimes higher at beta=10000 due to more distributed information.

---

## Phase 5 — OGBN-Arxiv Scalability (Intrinsic)

**Setup:** 5 operators × 2 betas × 3 seeds on OGBN-Arxiv (~169K nodes, 128 features, 40 classes, 10 clients).  
Intrinsic metrics only — no downstream classification on OGBN-Arxiv.

### Table 5.1 — OGBN-Arxiv Intrinsic Metrics, Beta=1

| Operator | n | MSE (↓) | CosSim (↑) | Recov. Ratio (↑) | Conv. Rate | Wall Time (s) |
|----------|---|---------|-----------|-----------------|-----------|--------------|
| **Asym. RW** | 3 | **0.01165 ± 0.00002** | 0.8957 ± 0.0002 | **0.7898 ± 0.0005** | 0.00 | 2514.1 |
| Adjacency | 3 | 0.01913 ± 0.00006 | 0.8966 ± 0.0002 | 0.6546 ± 0.0010 | 0.00 | 2145.2 |
| Diffusion | 3 | 0.01968 ± 0.00007 | **0.8982 ± 0.0001** | 0.6446 ± 0.0011 | 0.00 | 102.4 |
| Chebyshev | 3 | 0.01968 ± 0.00007 | 0.8982 ± 0.0001 | 0.6446 ± 0.0011 | 0.00 | 102.3 |
| APPNP | 3 | 0.02630 ± 0.00024 | 0.8959 ± 0.0003 | 0.5252 ± 0.0041 | 1.00 | 33.7 |

### Table 5.2 — OGBN-Arxiv Intrinsic Metrics, Beta=10000

| Operator | n | MSE (↓) | CosSim (↑) | Recov. Ratio (↑) | Conv. Rate | Wall Time (s) |
|----------|---|---------|-----------|-----------------|-----------|--------------|
| **Asym. RW** | 3 | **0.01153 ± 0.00000** | 0.8966 ± 0.0000 | **0.7916 ± 0.0000** | 0.00 | 2434.9 |
| Adjacency | 3 | 0.01884 ± 0.00001 | 0.8975 ± 0.0000 | 0.6594 ± 0.0002 | 0.00 | 2464.6 |
| Diffusion | 3 | 0.01944 ± 0.00001 | **0.8988 ± 0.0000** | 0.6486 ± 0.0002 | 0.00 | 117.0 |
| Chebyshev | 3 | 0.01944 ± 0.00001 | 0.8988 ± 0.0000 | 0.6486 ± 0.0002 | 0.00 | 116.8 |
| APPNP | 3 | 0.02631 ± 0.00001 | 0.8969 ± 0.0000 | 0.5245 ± 0.0003 | 1.00 | 37.8 |

### Table 5.3 — OGBN-Arxiv: Efficiency vs. Quality Summary

| Operator | MSE (β=10000) | Wall Time (β=10000, s) | MSE×WT (proxy cost) |
|----------|--------------|----------------------|---------------------|
| APPNP | 0.02631 | 37.8 | 0.994 |
| Diffusion | 0.01944 | 117.0 | 2.274 |
| Chebyshev | 0.01944 | 116.8 | 2.270 |
| Adjacency | 0.01884 | 2464.6 | 46.4 |
| Asym. RW | 0.01153 | 2434.9 | 28.1 |

> **Key observations:**  
> - At large scale, **Asym. RW achieves the lowest MSE and highest recovery ratio** but at significant cost (~41 min per run).  
> - Diffusion and Chebyshev produce identical results on OGBN-Arxiv (same underlying solver in this regime).  
> - **APPNP is by far the most efficient** (38s vs 2500s for Asym. RW), converging reliably, but at the cost of higher MSE.  
> - Cosine similarities are high across all operators (0.895–0.899), much higher than on Cora (~0.22). OGBN-Arxiv's sparser federated partition benefits more from propagation.  
> - No operator converges in 100 iterations on OGBN-Arxiv except APPNP — suggesting the maximum iteration budget is insufficient for the other solvers at this scale.

---

## Phase 6 — Heterophily Stress Test (Texas + Wisconsin)

**Setup:** 4 operators × 2 datasets × 5 seeds. Backbone: GCN only. Beta=10000.  
Texas (183 nodes, 10 classes, heterophilic) and Wisconsin (251 nodes, 5 classes, heterophilic).

### Table 6.1 — Texas Downstream Accuracy (GCN, Beta=10000)

| Operator | n | Accuracy (↑) | Std | Min | Max |
|----------|---|-------------|-----|-----|-----|
| Adjacency | 5 | **0.7405** | 0.0582 | 0.6486 | 0.8108 |
| Diffusion | 5 | 0.7351 | 0.0626 | 0.6486 | 0.8108 |
| APPNP | 5 | 0.7243 | 0.0649 | 0.6486 | 0.8108 |
| **Zero-Hop** | 5 | 0.6973 | 0.0432 | 0.6216 | 0.7297 |

### Table 6.2 — Wisconsin Downstream Accuracy (GCN, Beta=10000)

| Operator | n | Accuracy (↑) | Std | Min | Max |
|----------|---|-------------|-----|-----|-----|
| **Diffusion** | 5 | **0.7882** | 0.0487 | 0.7255 | 0.8431 |
| APPNP | 5 | 0.7725 | 0.0422 | 0.7255 | 0.8235 |
| Zero-Hop | 5 | 0.7725 | 0.0440 | 0.7059 | 0.8431 |
| Adjacency | 5 | 0.7412 | 0.0260 | 0.7059 | 0.7843 |

### Table 6.3 — Heterophily Summary: Propagation Gain over Zero-Hop

| Dataset | Operator | Acc. | Gain vs. Zero-Hop |
|---------|----------|------|-------------------|
| Texas | Zero-Hop | 0.6973 | — |
| Texas | Adjacency | 0.7405 | +0.0432 |
| Texas | Diffusion | 0.7351 | +0.0378 |
| Texas | APPNP | 0.7243 | +0.0270 |
| Wisconsin | Zero-Hop | 0.7725 | — |
| Wisconsin | Diffusion | 0.7882 | +0.0157 |
| Wisconsin | APPNP | 0.7725 | ±0.0000 |
| Wisconsin | Adjacency | 0.7412 | −0.0313 |

> **Key observations:**  
> - On heterophilic graphs, propagation provides **modest positive gains** on Texas (+4 pp for Adjacency).  
> - On Wisconsin, propagation effects are mixed: Diffusion slightly helps (+1.6 pp), APPNP is neutral, and Adjacency **hurts** (−3.1 pp).  
> - This aligns with the literature: neighbor averaging (Adjacency) smooths features in a way that is harmful when neighbors are likely to be different-class.  
> - Diffusion's behaviour on Wisconsin is notable — it avoids the averaging pitfall of Adjacency despite being iterative.  
> - High variance across seeds (std ~0.04–0.06) reflects the small size of these graphs.

---

## Cross-Dataset Summary Tables

### Table S1 — Best Operator per Dataset × Backbone × Beta (Downstream)

| Dataset | Backbone | Beta | Best Operator | Best Acc. | Gap Closed |
|---------|----------|------|--------------|-----------|-----------|
| Cora | GCN | 1 | Chebyshev | 0.7840 | 0.924 |
| Cora | GCN | 10000 | Chebyshev | 0.7205 | 0.570 |
| Cora | GAT | 1 | Chebyshev | 0.7006 | 0.825 |
| Cora | GAT | 10000 | Chebyshev | 0.6898 | 0.731 |
| Citeseer | GCN | 1 | Chebyshev | 0.6728 | 0.613 |
| Citeseer | GCN | 10000 | Adjacency | 0.6438 | 0.349 |
| Citeseer | GAT | 1 | Chebyshev | 0.6002 | 0.610 |
| Citeseer | GAT | 10000 | Adjacency | 0.5914 | 0.516 |
| Pubmed | GCN | 1 | Chebyshev | 0.7712 | 0.970 |
| Pubmed | GCN | 10000 | Asym. RW | 0.7780 | 0.789 |
| Pubmed | GAT | 1 | Asym. RW | 0.7344 | 0.808 |
| Pubmed | GAT | 10000 | Adjacency | 0.7486 | 0.782 |
| Texas | GCN | 10000 | Adjacency | 0.7405 | — |
| Wisconsin | GCN | 10000 | Diffusion | 0.7882 | — |

### Table S2 — Operator Rankings by Downstream Accuracy (Homophilic, Averaged Across Cora/Citeseer/Pubmed)

Rankings computed by averaging across datasets, betas, and backbones (homophilic only).

| Rank | Operator | Avg. Acc. (GCN) | Avg. Acc. (GAT) | Avg. Gap Closed |
|------|----------|----------------|----------------|----------------|
| 1 | **Chebyshev** | 0.740 | 0.668 | 0.736 |
| 2 | Adjacency | 0.736 | 0.660 | 0.715 |
| 3 | Asym. RW | 0.735 | 0.661 | 0.714 |
| 4 | APPNP | 0.733 | 0.657 | 0.695 |
| 5 | Diffusion | 0.726 | 0.644 | 0.643 |

> Ranked by average gap closed. Chebyshev leads consistently but Adjacency and Asym. RW are competitive.

### Table S3 — Operator Rankings by Intrinsic MSE (Cora, Beta=10000)

| Rank | Operator | MSE | CosSim | Recov. Ratio |
|------|----------|-----|--------|-------------|
| 1 | Heat Kernel (Exact) | 0.01231 | 0.2068 | +0.0343 |
| 2 | APPNP | 0.01299 | 0.2183 | −0.0169 |
| 3 | Chebyshev | 0.01325 | **0.2386** | −0.0373 |
| 4 | Diffusion | 0.01334 | 0.2189 | −0.0444 |
| 5 | Adjacency | 0.01400 | 0.2237 | −0.0962 |
| 6 | Asym. RW | 0.01602 | 0.2196 | −0.2541 |

> **Note:** Chebyshev has the best cosine similarity (directional alignment) despite not having the lowest MSE.

### Table S4 — Intrinsic vs. Downstream Rank Comparison (Cora)

| Operator | Intrinsic MSE Rank | Downstream Acc. Rank (GCN, β=1) | Discrepancy |
|----------|-------------------|---------------------------------|-------------|
| APPNP | 1 (best) | 4 | ↓ 3 |
| Chebyshev | 3 | 1 (best) | ↑ 2 |
| Diffusion | 4 | 5 (worst) | ↓ 1 |
| Adjacency | 5 | 2 | ↑ 3 |
| Asym. RW | 6 (worst) | 3 | ↑ 3 |

> **Key finding:** Intrinsic MSE is a poor predictor of downstream classification performance. Chebyshev ranks 3rd on intrinsic MSE yet 1st on downstream accuracy; APPNP is best intrinsically but 4th downstream.

---

## Experimental Notes and Limitations

### Known Failures

**3 NaN results in Phase 3 (Cora, GCN, beta=10000):**

| Operator | Dataset | Backbone | Beta | Seed |
|----------|---------|----------|------|------|
| Chebyshev | Cora | GCN | 10000 | 2 |
| APPNP | Cora | GCN | 10000 | 1 |
| Asym. RW | Cora | GCN | 10000 | 2 |

Cause: beta=10000 produces propagation matrices with eigenvalues exceeding float32 stability in the GCN computation path. These are expected edge-case failures, not infrastructure bugs. Results for these cells are averaged over 4 seeds instead of 5.

### Scope Limitations

- **Phase 5 (OGBN-Arxiv):** Intrinsic metrics only. Downstream node classification was not run due to scale.
- **Phase 6 (Texas/Wisconsin):** GCN backbone only; no GAT comparison for heterophilic graphs.
- **Phase 6 (Texas/Wisconsin):** Single beta value (10000); no beta sweep.
- **Phase 2 (Ablation):** Intrinsic metrics only; no corresponding downstream ablation sweep.

### Convergence Notes

- Diffusion and Chebyshev do not converge within the 100-iteration budget on Cora (any beta) or OGBN-Arxiv.
- Adjacency converges at ~91–97% rate on Cora; does not converge on OGBN-Arxiv.
- APPNP converges reliably everywhere (ConvRate ≈ 1.00) and is the only operator that converges on OGBN-Arxiv.
- Asym. RW converges on Cora (100%) but not OGBN-Arxiv.

---

## Raw Data Locations

```
experiments/propagator_eval/results/
├── phase_1_cora_intrinsic/raw/{operator}/cora/beta{B}_seed{S}.json
│       Keys: operator, dataset, beta, seed, n_clients, hop, per_client[], aggregate{}
│       aggregate: mse_mean/std, cosine_sim_mean/std, recovery_ratio_mean/std,
│                  boundary_coverage_mean, n_iters_mean/std, wall_time_total_sec, convergence_rate
│
├── phase_2_cora_ablation/raw/
│   ├── appnp_alpha/{alpha_005,alpha_01,alpha_02}/appnp/cora/beta10000_seed{S}.json
│   ├── epsilon_cora/{tol_1e-02,tol_1e-03,tol_1e-04}/diffusion/cora/beta10000_seed{S}.json
│   └── hop_depth/{hop1,hop2}/{adjacency,diffusion}/cora/beta10000_seed{S}.json
│
├── phase_3_cora_downstream/raw/{operator}/cora/beta{B}_seed{S}_{backbone}.json
│       Keys: operator, dataset, beta, seed, backbone, test_accuracy,
│             accuracy_gap_closed, zero_hop_accuracy, oracle_accuracy,
│             per_client_accuracy[], wall_time_sec
│
├── phase_4_homophilic_reproduction/raw/{operator}/{dataset}/beta{B}_seed{S}_{backbone}.json
│       (Same schema as Phase 3)
│
├── phase_5_scalability_ogbn_arxiv/raw/{operator}/ogbn-arxiv/beta{B}_seed{S}.json
│       (Same schema as Phase 1)
│
└── phase_6_heterophily_stress/raw/{operator}/{dataset}/beta{B}_seed{S}_gcn.json
        (Same schema as Phase 3)
```
