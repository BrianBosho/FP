# Propagator Analysis — Experimental Protocol
# For use with IDE agent / codebase implementation

---

## 0. Context

This protocol generates the experiments for the paper:
**"Intrinsic Evaluation of Propagation Operators for
Communication-Free Subgraph Federated Learning"**

The codebase already implements FedProp with two operators
(Adj, Diffusion). This protocol extends it with three new
operators, adds intrinsic evaluation instrumentation, and
defines the exact run matrix.

---

## 1. What Already Exists — Do Not Reimplement

| Component | Status | Notes |
|---|---|---|
| Normalized Adjacency propagator | DONE | FedProp-Adj |
| Diffusion / Taylor propagator | DONE | FedProp-Diff |
| IID/non-IID partitioning (β) | DONE | β = 10000, 10, 1 |
| GCN backbone | DONE | 2-layer, 16 hidden |
| GAT backbone | DONE | 2-layer, 8×8 heads |
| Cora, Citeseer, Pubmed datasets | DONE | |
| OGBN-Arxiv dataset | DONE | |
| Downstream accuracy logging | DONE | per-client + aggregate |
| FedAvg aggregation | DONE | |

---

## 2. New Operators to Implement

Implement each as a standalone propagation function with the
same interface as the existing Adj/Diffusion operators.

### 2.1 Interface Contract

Every propagator must accept:
- `A` : adjacency matrix (sparse, local client subgraph)
- `X` : feature matrix [n_nodes × n_features]
- `boundary_mask` : boolean mask, True = known boundary node
- `X_boundary` : known feature values for boundary nodes
- `T_max` : maximum iterations (default: 100)
- `eps` : convergence tolerance on feature change norm (default: 1e-4)

Every propagator must return:
- `X_imputed` : completed feature matrix [n_nodes × n_features]
- `n_iters` : number of iterations run
- `residuals` : list of per-iteration residual values (see Section 3)
- `wall_time` : total propagation time in seconds

### 2.2 Operators to Add

**Chebyshev (O4)**
- Approximate the heat kernel using Chebyshev polynomials of order K
- Parameter: K ∈ {3, 5, 10} — default K=5
- Apply boundary reset at each iteration as with existing operators
- Expected behaviour: faster convergence than Taylor on denser graphs

**APPNP / PPR (O6)**
- Update rule: X^(t+1) = (1−α) · Â · X^(t) + α · X^(0)
- Where Â is the normalized adjacency, X^(0) is the initialization
- Apply boundary reset after each update
- Parameter: α ∈ {0.05, 0.1, 0.2} — default α=0.1
- Note: fixed point is NOT the Dirichlet minimizer — it is a
  personalized PageRank solution. This is expected and correct.

**Random Walk (O2)**
- Propagation matrix: P = D^{-1} A  (asymmetric)
- Apply boundary reset at each iteration
- Note: asymmetric operator; fixed point differs from Adj.
  Include as comparison baseline, not as a Dirichlet solver.

**Heat Kernel Exact (O5) — Reference Only**
- Compute exp(−t·L) via eigendecomposition
- Run ONLY on Cora and Citeseer (small graphs — O(n^3) cost)
- Single-shot (no iteration); t = 1.0 default
- Use as a theoretical reference point, not a practical operator

---

## 3. Instrumentation to Add

These changes go into the existing propagation loop.
Do not restructure the loop — add logging around it.

### 3.1 Per-Iteration Logging (inside loop)

At each iteration t, record:

```
log[t] = {
    "iter": t,
    "dirichlet_residual": frob_norm(L @ X_t) ** 2,
    "feature_change_norm": frob_norm(X_t1 - X_t),
    "reconstruction_mse": mse(X_t[unknown_mask], X_true[unknown_mask])
        # only if X_true is available (intrinsic eval mode)
}
```

Store `log` as a list and return it as `residuals`.

### 3.2 Feature Reconstruction Evaluation (intrinsic mode)

Add an `intrinsic_eval` flag to the propagation call.

When `intrinsic_eval=True`:
- Accept `X_true` as an additional argument (ground truth features)
- At convergence, compute and return:

```
intrinsic_metrics = {
    "mse":              mean over unknown nodes of ||x*_i - x_true_i||^2,
    "cosine_sim":       mean cosine similarity over unknown nodes,
    "recovery_ratio":   (mse_zero_hop - mse_operator) / mse_zero_hop,
    "spectral_fidelity": frob_norm(cov(X_imputed) - cov(X_true))
}
```

Where `mse_zero_hop` is the MSE when unknown features are
left as zeros (the baseline). Compute this once per client
and pass it in, or recompute inside the function.

### 3.3 Wall-Clock Timing

Wrap the propagation loop only (exclude graph construction
and matrix precomputation):

```python
import time
t_start = time.perf_counter()
# ... propagation loop ...
wall_time = time.perf_counter() - t_start
```

Return `wall_time` in the output dict.

---

## 4. Datasets

| Dataset | Scale | Homophily | Role | New? |
|---|---|---|---|---|
| Cora | Small | 0.81 | Primary | No |
| Citeseer | Small | 0.74 | Primary | No |
| Pubmed | Medium | 0.80 | Primary | No |
| OGBN-Arxiv | Large | 0.66 | Scaling | No |
| Texas | Small | 0.11 | Heterophilic test | Yes — add |
| Wisconsin | Small | 0.20 | Heterophilic test | Yes — add |

**Dropped from earlier plan:** Amazon Computers, Amazon Photos.
These do not add coverage that the six above don't already provide.

Texas and Wisconsin are available in PyG as:
`torch_geometric.datasets.WebKB(root, name="Texas")`
`torch_geometric.datasets.WebKB(root, name="Wisconsin")`

---

## 5. Run Matrix

### Layer 1 + 2 + 3 — Intrinsic / Process / Efficiency
(No GNN training. Pure propagation runs. Fast.)

Run for every combination of:
- **Operators:** Adj, Diffusion, Chebyshev(K=5), APPNP(α=0.1), RW
- **Datasets:** Cora, Citeseer, Pubmed, OGBN-Arxiv, Texas, Wisconsin
- **Partitions:** β = 10000, β = 10, β = 1
- **Seeds:** 3 seeds (sufficient for propagation variance)
- **Hop depth:** L=1 (default); L=2 as ablation on Cora only

Heat kernel exact: Cora + Citeseer only, β = 10000 only, 1 seed.

**Total Layer 1–3 runs:**
5 operators × 6 datasets × 3 β × 3 seeds = 270 runs
+ Heat kernel: 2 datasets × 1 seed = 2 runs
+ L=2 ablation on Cora: 5 operators × 3 β × 3 seeds = 45 runs

Each run is a propagation loop only — on Cora this is seconds;
on OGBN-Arxiv expect ~1–5 minutes per run depending on operator.

### Layer 4 — Downstream Accuracy
(GNN training. Run only on homophilic datasets.)

Run for every combination of:
- **Operators:** Adj, Diffusion, Chebyshev(K=5), APPNP(α=0.1), RW
- **Datasets:** Cora, Citeseer, Pubmed, OGBN-Arxiv
- **Partitions:** β = 10000, β = 1
  (Skip β=10 for downstream to reduce runs — β=1 is the hard case)
- **Backbones:** GCN, GAT
- **Seeds:** 5 seeds

Also run:
- Zero-hop baseline (no propagation) for all above
- Oracle baseline (true remote features) for Cora/Citeseer/Pubmed

**Total Layer 4 runs:**
5 operators × 4 datasets × 2 β × 2 backbones × 5 seeds = 400 runs
+ Zero-hop: 4 datasets × 2 β × 2 backbones × 5 seeds = 80 runs
+ Oracle: 3 datasets × 2 β × 2 backbones × 5 seeds = 60 runs

---

## 6. Ablations (Run After Main Matrix)

Only run these if time permits before the submission deadline.

| Ablation | Factor | Levels | Datasets | Seeds |
|---|---|---|---|---|
| K sensitivity | Chebyshev order | K = 3, 5, 10 | Pubmed | 3 |
| α sensitivity | APPNP α | 0.05, 0.1, 0.2 | Cora | 3 |
| ε sensitivity | Convergence tol. | 1e-2, 1e-3, 1e-4 | Cora, OGBN | 3 |
| L=2 topology | Hop depth | L=1, L=2 | Cora | 3 |

---

## 7. Output Format

All runs save a single JSON result file per run.

### File naming convention:
```
results/{layer}/{operator}/{dataset}/beta{beta}_seed{seed}.json
```

Example:
```
results/intrinsic/chebyshev/cora/beta10_seed0.json
results/downstream/appnp/ogbn-arxiv/beta1_seed2.json
```

### Intrinsic result file schema:
```json
{
  "operator": "chebyshev",
  "dataset": "cora",
  "beta": 10,
  "seed": 0,
  "n_clients": 10,
  "per_client": [
    {
      "client_id": 0,
      "n_known": 142,
      "n_unknown": 38,
      "missing_neighbor_frac": 0.21,
      "mse": 0.034,
      "cosine_sim": 0.87,
      "recovery_ratio": 0.61,
      "n_iters": 34,
      "wall_time_sec": 0.12,
      "residuals": [6.2, 4.1, 2.8, ...]
    }
  ],
  "aggregate": {
    "mse_mean": 0.036,
    "mse_std": 0.008,
    "cosine_sim_mean": 0.85,
    "recovery_ratio_mean": 0.59,
    "n_iters_mean": 36.2,
    "wall_time_total_sec": 1.4
  }
}
```

### Downstream result file schema:
```json
{
  "operator": "chebyshev",
  "dataset": "cora",
  "beta": 1,
  "seed": 2,
  "backbone": "gcn",
  "test_accuracy": 0.812,
  "accuracy_gap_closed": 0.74,
  "zero_hop_accuracy": 0.687,
  "oracle_accuracy": 0.810,
  "per_client_accuracy": [0.81, 0.79, ...]
}
```

---

## 8. Priority Order

Run in this order. Stop and check results after each phase
before proceeding to the next.

**Phase 1 — Implement and validate new operators (2–3 days)**
1. Implement Chebyshev propagator
2. Implement APPNP propagator
3. Implement Random Walk propagator
4. Validate each on Cora: check that the loop converges,
   residuals decay, and output shapes are correct
5. Implement Heat Kernel exact on Cora only as a sanity check

**Phase 2 — Add instrumentation (1 day)**
1. Add per-iteration logging to all operators
2. Add intrinsic evaluation mode (X_true holdout + metrics)
3. Add wall-clock timing
4. Validate on Cora: run intrinsic eval and confirm
   recovery_ratio is between 0 and 1

**Phase 3 — Run Layer 1–3 (propagation only) (2–3 days)**
1. Run all operators on Cora first — verify JSON outputs
2. Run full matrix: all operators × all datasets × all β
3. Texas and Wisconsin last (need dataset loading)

**Phase 4 — Run Layer 4 (downstream) (3–5 days)**
1. Run zero-hop and oracle baselines first
2. Run Adj and Diffusion (already implemented, validation)
3. Run Chebyshev, APPNP, RW downstream
4. GCN before GAT (faster)

**Phase 5 — Ablations (if time)**
Run only if Phase 3 and Phase 4 are complete before June 1.

---

## 9. Fixed-Point Notes for Implementation

These notes are important for correctly implementing and
interpreting each operator. The IDE agent should be aware
that operators have different fixed points.

| Operator | Fixed Point | Equals Dirichlet minimizer? |
|---|---|---|
| Normalized Adj (O1) | Harmonic extension of X_V | YES |
| Random Walk (O2) | Asymmetric smoothing | NO |
| Diffusion/Taylor (O3) | Harmonic extension (same as O1) | YES |
| Chebyshev (O4) | Heat kernel steady state | APPROXIMATELY |
| Heat Kernel exact (O5) | Exact heat kernel solution | APPROXIMATELY |
| APPNP/PPR (O6) | Personalized PageRank solution | NO |

The fixed point for all boundary-reset operators is:
X*_U = (I − P_UU)^{-1} P_UV X_V

This equals the Dirichlet minimizer only when P is derived
from the Laplacian (Adj, Diffusion). For APPNP and RW,
the fixed point is a different well-defined solution —
not wrong, but not the harmonic extension.

This distinction should be preserved in the paper's theory
section. The intrinsic MSE metrics will reveal empirically
whether this matters in practice.

---

## 10. Scope Freeze

The experiment matrix above is final. Do not add:
- Additional operators beyond the five primary + heat kernel
- Additional datasets beyond the six listed
- Additional backbone architectures
- Link prediction or graph classification tasks

Emergency rule: if Phase 3 is complete and Phase 4 is
running but not finished by June 5, submit with Phase 3
results only and note downstream results as preliminary
or in-progress in the limitations section.