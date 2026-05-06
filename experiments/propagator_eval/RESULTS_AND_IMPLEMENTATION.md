# Results Capture & Implementation Details

This document specifies every result we capture, how it's computed, and where the code changes go.

---

## Phase-Based Execution Checklist

We now use a **phase-based** execution order rather than starting from the full cross-dataset matrix.

Execution order:
1. `phase_1_cora_intrinsic`
2. `phase_2_cora_ablation`
3. `phase_3_cora_downstream`
4. `phase_4_homophilic_reproduction`
5. `phase_5_scalability_ogbn_arxiv`
6. `phase_6_heterophily_stress`

### Phase outputs we expect

- **Phase 1:** Cora intrinsic raw JSON, processed summaries, plots, `phase_1_findings.md`
- **Phase 2:** Cora ablation summaries and frozen defaults
- **Phase 3:** Cora downstream summaries plus intrinsic/downstream joined tables
- **Phase 4:** cross-dataset reproduction summaries for Citeseer and Pubmed
- **Phase 5:** OGBN-Arxiv scalability summaries, including explicit failed-run accounting when needed
- **Phase 6:** homophily-vs-heterophily comparison summaries

### Metric freeze for the first pass

Required intrinsic metrics:
- `mse`
- `cosine_sim`
- `recovery_ratio`
- `boundary_coverage`
- `n_iters` / iteration count
- `residuals`
- `converged`
- `wall_time_sec`

Deferred unless later analysis needs it:
- `spectral_fidelity`

### Layout note

The new configs write into phase-based result roots such as:

```text
results/phase_1_cora_intrinsic/raw/
results/phase_3_cora_downstream/raw/
```

The current runners still emit legacy operator/dataset subdirectories *inside* each phase's `raw/` directory. That is acceptable for now; phase-level post-processing should normalize this downstream.

---

## 1. Result Artifacts

Every run produces a single JSON file.

### File naming convention

```
results/{layer}/{operator}/{dataset}/beta{beta}_seed{seed}.json
```

Examples:
```
results/intrinsic/chebyshev_diffusion/cora/beta10_seed0.json
results/downstream/appnp/ogbn-arxiv/beta1_seed2.json
results/downstream/zero_hop/texas/beta10000_seed0.json
```

### 1.1 Intrinsic result file schema

```json
{
  "operator": "chebyshev_diffusion",
  "dataset": "cora",
  "beta": 10,
  "seed": 0,
  "n_clients": 10,
  "hop": 1,
  "per_client": [
    {
      "client_id": 0,
      "n_known": 142,
      "n_unknown": 38,
      "missing_neighbor_frac": 0.21,
      "boundary_coverage": 0.74,
      "mse": 0.034,
      "cosine_sim": 0.87,
      "recovery_ratio": 0.61,
      "n_iters": 34,
      "converged": true,
      "wall_time_sec": 0.12,
      "residuals": [6.2, 4.1, 2.8, 1.9, 1.2]
    }
  ],
  "aggregate": {
    "mse_mean": 0.036,
    "mse_std": 0.008,
    "cosine_sim_mean": 0.85,
    "cosine_sim_std": 0.03,
    "recovery_ratio_mean": 0.59,
    "recovery_ratio_std": 0.05,
    "boundary_coverage_mean": 0.71,
    "n_iters_mean": 36.2,
    "n_iters_std": 4.1,
    "wall_time_total_sec": 1.4,
    "convergence_rate": 1.0
  }
}
```

**`residuals`** — list of per-iteration Dirichlet residual values (one float per iteration). Flat list, not a list of dicts. Length = `n_iters`.

**`boundary_coverage`** — fraction of unknown nodes that have at least one known boundary neighbor. Low boundary coverage (sparse boundary) predicts poor reconstruction per Calder et al. on harmonic extension degeneracy. Log as a covariate for per-client analysis.

### 1.2 Downstream result file schema

```json
{
  "operator": "appnp",
  "dataset": "cora",
  "beta": 1,
  "seed": 2,
  "backbone": "gcn",
  "test_accuracy": 0.812,
  "accuracy_gap_closed": 0.74,
  "zero_hop_accuracy": 0.687,
  "oracle_accuracy": 0.810,
  "per_client_accuracy": [0.81, 0.79, 0.82, 0.80, 0.83, 0.78, 0.84, 0.79, 0.81, 0.80]
}
```

**`accuracy_gap_closed`** = `(acc_operator − acc_zero_hop) / (acc_oracle − acc_zero_hop)`. Computed post-hoc once zero-hop and oracle baselines are available. Can exceed 1.0 if operator outperforms oracle (unlikely but possible on small graphs).

---

## 2. Metrics — Computation Details

### 2.1 MSE (Feature Reconstruction)

```python
unknown = ~mask  # mask=True means known features
mse = F.mse_loss(out[unknown], X_true[unknown]).item()
```

Units: per-element squared error (lower is better). For Cora (1433 features) expect 0.01–0.5.

### 2.2 Cosine Similarity

```python
cos_sims = F.cosine_similarity(out[unknown], X_true[unknown], dim=1)
cosine_sim = cos_sims.mean().item()
```

Units: [-1, 1], higher is better.

### 2.3 Recovery Ratio

```python
zero_hop = torch.zeros_like(X_true)
zero_hop[mask] = x[mask]
mse_zero_hop = F.mse_loss(zero_hop[unknown], X_true[unknown]).item()
recovery_ratio = (mse_zero_hop - mse) / (mse_zero_hop + 1e-12)
```

Units: [0, 1] typically. 0 = no improvement over zero-hop. 1 = perfect recovery. Negative = operator made things worse.

### 2.4 Boundary Coverage

```python
# For each unknown node, check if it has at least one known neighbor
has_known_neighbor = scatter(mask[col].float(), row, reduce='max')[unknown]
boundary_coverage = has_known_neighbor.mean().item()
```

Units: [0, 1]. Low values flag clients where reconstruction is theoretically hard.

### 2.5 Dirichlet Residual (per iteration)

Recorded as the `residuals` list. At iteration t:

```python
row, col = edge_index
diffs = X_t[row] - X_t[col]
residual = (diffs ** 2).sum().item()  # Frob norm of L @ X_t, squared
```

For Dirichlet-minimizer operators (O1, O3) this should decrease monotonically. For APPNP/RW it may not — that is expected and correct.

### 2.6 Feature Change Norm (per iteration)

```python
feature_change_norm = torch.norm(X_t - X_prev).item()
```

Used for convergence check (`< eps`). Not stored separately — convergence is determined by this, residuals list stores the Dirichlet residual.

### 2.7 Wall-Clock Time

```python
import time
t_start = time.perf_counter()
# ... propagation loop only ...
wall_time = time.perf_counter() - t_start
```

Excludes matrix construction and dataset loading.

### 2.8 Test Accuracy and Gap Closed (downstream only)

Test accuracy from existing `main_experiment()` return value. Gap closed computed post-hoc from baseline files:

```python
gap_closed = (acc_op - acc_zero) / (acc_oracle - acc_zero + 1e-12)
```

---

## 3. Implementation Details by File

### 3.1 `src/fedgnn/data/propagation.py` — Propagation Core

**Modified function: `propagate_features()`**

New parameters:
```python
intrinsic_eval: bool = False,
X_true: Tensor | None = None,
```

New propagation modes added to the mode-dispatch:
- `"appnp"` — `X^(t+1) = (1-α)·Â·X^(t) + α·X^(0)` with boundary reset; α from config
- `"asymmetric_random_walk"` — `P = D^{-1}A`; standard iterative loop with boundary reset
- `"heat_kernel_exact"` — single-shot `H = V·diag(exp(-t·λ))·V^T` via eigendecomp; no loop

Per-iteration instrumentation (always active, not just intrinsic mode):
```python
residuals = []
# inside loop at each iter t:
diffs = X_t[row] - X_t[col]
residuals.append((diffs ** 2).sum().item())
```

Post-loop intrinsic metrics (only when `intrinsic_eval=True` and `X_true` provided):
```python
intrinsic_metrics = {
    "mse": ...,
    "cosine_sim": ...,
    "recovery_ratio": ...,
    "boundary_coverage": ...,
}
```

Return value:
- `intrinsic_eval=False` → `Tensor` (existing behavior unchanged)
- `intrinsic_eval=True` → `dict` with keys: `X_imputed`, `n_iters`, `converged`, `wall_time`, `residuals`, `intrinsic_metrics`

New helper: `_frob_dirichlet_residual(X, edge_index)` — computes `sum((X[row]-X[col])^2)`.

### 3.2 `src/fedgnn/data/data_utils.py` — New Operator Kernels

**`get_row_normalized_adjacency(edge_index, num_nodes)`**
- Computes `P = D^{-1}A`; adds self-loops for isolated nodes
- Returns `(edge_index_with_loops, edge_weight)` matching the symmetric adj format

**`heat_kernel_exact(edge_index, num_nodes, device, t=1.0)`**
- Builds dense normalized Laplacian; eigendecomp via `torch.linalg.eigh`
- Returns dense `[n_nodes, n_nodes]` matrix H
- Raises `ValueError` if `num_nodes > 10000` (safety guard)

### 3.3 `src/fedgnn/data/loaders.py` — Mode Dispatch

Add three new `elif` branches to the imputation dispatch (~line 105):
- `"appnp"` → `propagation_mode="appnp"`
- `"asymmetric_random_walk"` → `propagation_mode="asymmetric_random_walk"`
- `"heat_kernel_exact"` → `propagation_mode="heat_kernel_exact"`

### 3.4 `src/fedgnn/data/partitioning.py` — Mask Access

Add `return_masks: bool = False` to `partition_data()`.

When `True`: skip FP/PE, return `(clients_data, masks_list, split_data_indexes)` where `masks_list[i]` is a boolean tensor (True = original node, False = k-hop neighbor). Used by the intrinsic runner to hold out features before running propagation.

### 3.5 `src/fedgnn/experiments/run_intrinsic_eval.py` — New Script

**Purpose:** Run Layer 1-3 experiments (propagation only, no GNN training).

**Flow per (operator, dataset, beta, seed):**
1. Load dataset
2. Partition via `partition_data(return_masks=True)`
3. Per client:
   - `X_true = subgraph.x.clone()`
   - `subgraph.x[~mask] = 0`
   - Call `propagate_features(..., mode=operator, intrinsic_eval=True, X_true=X_true)`
4. Aggregate across clients
5. Save JSON to `results/intrinsic/{operator}/{dataset}/beta{beta}_seed{seed}.json`

**CLI:**
```bash
python -m src.fedgnn.experiments.run_intrinsic_eval \
    --config experiments/propagator_eval/configs/L1_L3_primary.yaml \
    --operator adjacency --dataset Cora --beta 10000 --seed 0
```

### 3.6 `src/fedgnn/experiments/run_downstream_eval.py` — New Script

**Purpose:** Run Layer 4 experiments (full FL training). Reuses `main_experiment()`.

**Flow per (operator, dataset, beta, backbone, seed):**
1. Build config from YAML with per-dataset optimizer settings
2. Call `main_experiment()` with `data_loading=operator`
3. Save JSON to `results/downstream/{operator}/{dataset}/beta{beta}_seed{seed}.json`
4. Gap-closed computed post-hoc once zero-hop and oracle baseline files exist

**CLI:**
```bash
python -m src.fedgnn.experiments.run_downstream_eval \
    --config experiments/propagator_eval/configs/L4_downstream_operators.yaml \
    --operator diffusion --dataset Cora --beta 1 --backbone GCN --seed 0
```

### 3.7 `src/fedgnn/experiments/run_experiments.py` — Minor Update

Add to `FEATURE_PROP_DATA_LOADING_OPTIONS`:
```python
"appnp", "asymmetric_random_walk", "heat_kernel_exact"
```

### 3.8 `conf/intrinsic_eval.yaml` — Base Config

Shared defaults for both eval runners. Per-run configs in `experiments/propagator_eval/configs/` override these.

---

## 4. Alpha Semantics Per Operator

`alpha` means different things per mode. Document clearly in code and paper.

| Mode | `alpha` meaning | Update rule | Default |
|------|-----------------|-------------|---------|
| `adjacency` | Blend weight | `α·(Â@X) + (1-α)·X` | 0.5 |
| `diffusion` | Blend weight | `α·(H@X) + (1-α)·X` | 0.5 |
| `chebyshev_diffusion` | Blend weight | `α·(cheb(X)) + (1-α)·X` | 0.5 |
| `asymmetric_random_walk` | Blend weight | `α·(P@X) + (1-α)·X` | 0.5 |
| `appnp` | Teleport probability | `(1-α)·(Â@X) + α·X^(0)` | 0.1 |
| `heat_kernel_exact` | N/A (single-shot) | `H @ X` | N/A |

For APPNP, `alpha` is the weight on the initial features X^(0). All other iterative modes use it as a blending weight between the diffused and previous iterate.

---

## 5. Files Summary

| File | Action | What Changes |
|------|--------|-------------|
| `src/fedgnn/data/propagation.py` | MODIFY | Add 3 new modes; add `intrinsic_eval`/`X_true` params; add per-iter `residuals` list; add `boundary_coverage`; add timing |
| `src/fedgnn/data/data_utils.py` | MODIFY | Add `get_row_normalized_adjacency()`, `heat_kernel_exact()` |
| `src/fedgnn/data/loaders.py` | MODIFY | Add 3 new dispatch branches |
| `src/fedgnn/data/partitioning.py` | MODIFY | Add `return_masks=True` option |
| `src/fedgnn/experiments/run_experiments.py` | MODIFY | Add 3 new modes to valid options set |
| `src/fedgnn/experiments/run_intrinsic_eval.py` | CREATE | Layer 1-3 runner |
| `src/fedgnn/experiments/run_downstream_eval.py` | CREATE | Layer 4 runner |
| `conf/intrinsic_eval.yaml` | CREATE | Base config for eval runners |
