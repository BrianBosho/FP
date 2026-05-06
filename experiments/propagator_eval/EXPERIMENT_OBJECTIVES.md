# Experiment Objectives & Run Matrix

## Paper

**"Intrinsic Evaluation of Propagation Operators for Communication-Free Subgraph Federated Learning"**

**Positioning:** The paper contributes systematic intrinsic evaluation of the communication-free imputation family (FedProp line), not a new accuracy claim. It does not need to beat generator-based (FedSage+/FedDEP) or pre-communication (FedGCN/FedGAT) methods.

---

## Staged Evaluation Strategy

We now execute the project in **anchor-dataset-first** order rather than launching the full matrix up front.

Execution order:
1. **Phase 1 — Cora intrinsic core**
2. **Phase 2 — Cora ablations**
3. **Phase 3 — Cora downstream evaluation**
4. **Phase 4 — Citeseer/Pubmed homophilic reproduction**
5. **Phase 5 — OGBN-Arxiv scalability**
6. **Phase 6 — Texas/Wisconsin heterophily stress test**

The first objective is to deeply characterize propagator behavior on **Cora**.
Only after the Cora analysis pipeline is complete do we reproduce the frozen protocol on additional datasets.

For the first-pass intrinsic suite:
- keep `boundary_coverage` as required
- defer `spectral_fidelity` unless later analysis explicitly needs it

---

## 1. Research Questions

**RQ1 — Operator quality (intrinsic):** How well do different propagation operators recover missing cross-client features, independent of any downstream task?

**RQ2 — Convergence behavior:** How many iterations does each operator need? How does the residual decay? Do all operators converge?

**RQ3 — Downstream impact:** How does intrinsic propagation quality translate to federated GNN test accuracy? Is better feature recovery always better accuracy?

**RQ4 — Scaling:** How do operators behave as graph size increases (Cora → OGBN-Arxiv) and as data heterogeneity increases (β=10000 → β=1)?

**RQ5 — Heterophily:** Do operators that are not Dirichlet minimizers (APPNP, RW) behave differently on heterophilic graphs (Texas, Wisconsin)? Does low intrinsic recovery ratio predict low downstream accuracy gain?

**RQ6 — Hop-depth × heterogeneity interaction:** When client-owned nodes are IID-balanced (β=10000), does increasing subgraph depth from L=1 to L=2 hurt communication-free propagation because each client receives a much larger remote/imputed neighborhood with only a small owned training boundary? Conversely, can non-IID partitions (β=1) sometimes look better because their expanded neighborhoods are smaller or more class-coherent, even though label balance is worse?

This RQ is motivated by the Cora observation that β=10000 with 10 clients and L=2 gives nearly equal owned partitions, but each client expands from roughly 270 owned nodes to about 2,000 visible nodes. The propagator must then infer features for many remote nodes from only ~10-16 local training nodes per client. The experiment should distinguish a true IID/non-IID learning effect from a topology/imputation effect caused by the L-hop expansion.

---

## 2. Operators Under Study

| ID | Operator | Code name | Fixed Point | Dirichlet minimizer? | New? |
|----|----------|-----------|-------------|----------------------|------|
| O1 | Normalized Adjacency (FedProp-Adj) | `adjacency` | Harmonic extension | YES | No — exists |
| O2 | Random Walk (D^{-1}A) | `asymmetric_random_walk` | Asymmetric smoothing | NO | **Yes** |
| O3 | Diffusion / Taylor (FedProp-Diff) | `diffusion` | Harmonic extension | YES | No — exists |
| O4 | Chebyshev (K=5) | `chebyshev_diffusion` | Heat kernel steady state | APPROXIMATELY | No — exists |
| O5 | Heat Kernel Exact | `heat_kernel_exact` | Exact heat kernel | APPROXIMATELY | **Yes** — reference only |
| O6 | APPNP / PPR (α=0.1) | `appnp` | Personalized PageRank | NO | **Yes** |

**Fixed-point note:** The general boundary-reset fixed point is `X*_U = (I − P_UU)^{-1} P_UV X_V`. This equals the Dirichlet minimizer only for O1/O3. O2/O6 converge to different well-defined solutions — not wrong, but not the harmonic extension. This distinction is preserved in the theory section.

### Propagator Interface Contract

Every propagator accepts:
- `A` — adjacency matrix (sparse, local client subgraph)
- `X` — feature matrix [n_nodes × n_features]
- `boundary_mask` — boolean, True = known boundary node
- `X_boundary` — known feature values for boundary nodes
- `T_max` — maximum iterations (default: 100)
- `eps` — convergence tolerance on feature change norm (default: 1e-4)

Every propagator returns:
- `X_imputed` — completed feature matrix [n_nodes × n_features]
- `n_iters` — iterations run
- `residuals` — list of per-iteration residual values
- `wall_time` — propagation time in seconds

---

## 3. Datasets

| Dataset | Nodes | Edges | Features | Classes | Homophily | Role | New? |
|---------|-------|-------|----------|---------|-----------|------|------|
| Cora | 2,708 | 10,556 | 1,433 | 7 | 0.81 | Primary small | No |
| Citeseer | 3,327 | 9,104 | 3,703 | 6 | 0.74 | Primary small | No |
| Pubmed | 19,717 | 88,648 | 500 | 3 | 0.80 | Primary medium | No |
| OGBN-Arxiv | 169,343 | 1,166,243 | 128 | 40 | 0.66 | Scaling large | No |
| Texas | 183 | 325 | 1,703 | 5 | 0.11 | Heterophilic | **Yes** |
| Wisconsin | 251 | 515 | 1,703 | 5 | 0.20 | Heterophilic | **Yes** |

Texas and Wisconsin loaded via `torch_geometric.datasets.WebKB(root, name=...)`.

**Not included:** Amazon Computers, Amazon Photos — coverage is redundant given the six datasets above.

---

## 4. Evaluation Layers

Layer 1-3 require no GNN training (fast). Layer 4 requires full FL training (slow).

### Layer 1 — Feature Reconstruction (intrinsic quality)

For each client subgraph, hold out true features of k-hop neighbors, run propagation, compare imputed vs. true.

**Metrics per client:** MSE (unknown nodes only), cosine similarity, recovery ratio `(mse_zero_hop − mse_op) / mse_zero_hop`, boundary coverage (fraction of unknown nodes with ≥1 known neighbor).

### Layer 2 — Convergence Dynamics

**Metrics per client:** Per-iteration Dirichlet residual `||L @ X_t||_F^2`, per-iteration feature change norm `||X_{t+1} − X_t||_F`, per-iteration reconstruction MSE (when X_true available), total iterations, converged flag.

### Layer 3 — Computational Efficiency

**Metrics:** Wall-clock time (propagation loop only, excludes matrix construction), total time per dataset.

### Layer 4 — Downstream Accuracy

**Metrics:** Test accuracy (per-client + aggregate), accuracy gap closed `(acc_op − acc_zero) / (acc_oracle − acc_zero)`, per-client accuracy distribution.

---

## 5. Run Matrix

### 5.1 Layer 1-3: Intrinsic / Process / Efficiency

| Factor | Values | Count |
|--------|--------|-------|
| Operators | adjacency, asymmetric_random_walk, diffusion, chebyshev_diffusion, appnp | 5 |
| Datasets | Cora, Citeseer, Pubmed, OGBN-Arxiv, Texas, Wisconsin | 6 |
| Partitions (β) | 10000, 10, 1 | 3 |
| Seeds | 0, 1, 2 | 3 |
| Hop depth | L=1 default; L=2 Cora ablation | 1 + ablation |

**Base runs:** 5 × 6 × 3 × 3 = **270 runs**

**Heat kernel exact (O5):** Cora + Citeseer only, β=10000, 1 seed = **2 runs**

**L=2 ablation:** Cora only, 5 operators × 3 β × 3 seeds = **45 runs**. Primary contrast: compare L=1 vs L=2 under β=10000 and β=1 to test RQ6.

**Total Layer 1-3:** **317 runs** (~4 hours estimated)

### 5.2 Layer 4: Downstream Accuracy

**Homophilic — full matrix:**

| Factor | Values | Count |
|--------|--------|-------|
| Operators | adjacency, asymmetric_random_walk, diffusion, chebyshev_diffusion, appnp | 5 |
| Datasets | Cora, Citeseer, Pubmed, OGBN-Arxiv | 4 |
| Partitions (β) | 10000, 1 (skip β=10 to reduce runs) | 2 |
| Backbones | GCN, GAT | 2 |
| Seeds | 0, 1, 2, 3, 4 | 5 |

Operator runs: 5 × 4 × 2 × 2 × 5 = **400 runs**
Zero-hop baseline: 4 × 2 × 2 × 5 = **80 runs**
Oracle baseline (Cora/Citeseer/Pubmed only): 3 × 2 × 2 × 5 = **60 runs**

**Heterophilic — theory-validation run:**

Three representative operators (one Dirichlet: Adj; one Dirichlet: Diff; one non-Dirichlet: APPNP) + zero-hop baseline, IID only, GCN only. Purpose: validate that low Layer 1 recovery ratio predicts low Layer 4 accuracy gain.

| Factor | Values | Count |
|--------|--------|-------|
| Operators | adjacency, diffusion, appnp + zero_hop | 4 |
| Datasets | Texas, Wisconsin | 2 |
| Partitions (β) | 10000 only | 1 |
| Backbones | GCN | 1 |
| Seeds | 0, 1, 2, 3, 4 | 5 |

Heterophilic runs: 4 × 2 × 1 × 1 × 5 = **40 runs**

**Total Layer 4:** **580 runs** (~100 hours estimated)

### 5.3 Ablations (if time permits before June 1)

| Ablation | Factor | Levels | Datasets | Seeds | Runs |
|----------|--------|--------|----------|-------|------|
| K sensitivity | Chebyshev order | K = 3, 5, 10 | Pubmed | 3 | 9 |
| α sensitivity | APPNP α | 0.05, 0.1, 0.2 | Cora | 3 | 9 |
| ε sensitivity | Convergence tol | 1e-2, 1e-3, 1e-4 | Cora, OGBN | 3 | 18 |
| Hop-depth × β topology | Hop depth × partition heterogeneity | L=1, L=2 × β=10000, β=1 | Cora | 3 | 12 |

**Total ablations:** **48 runs** (intrinsic only, no training)

---

## 6. Total Run Count

| Category | Runs | Est. Time/Run | Total Est. |
|----------|------|---------------|------------|
| Layer 1-3 intrinsic | 317 | 1s – 5min | ~4 hours |
| Layer 4 downstream | 580 | 2min – 30min | ~110 hours |
| Ablations | 48 | 1s – 5min | ~30 min |
| **Grand total** | **945** | | **~115 hours** |

Layer 4 dominates. Parallelize across GPUs. Layer 1-3 is fast enough to run sequentially.

---

## 7. Priority Order

1. **Phase 0 — Protocol freeze and repo cleanup**
   - adopt the phase-based configs
   - treat local execution docs as authoritative
   - keep old layer-based configs only for backward compatibility
2. **Phase 1 — Cora intrinsic core**
   - run the five primary operators on Cora
   - run the Cora heat-kernel reference companion
   - generate the first operator comparison tables and findings note
3. **Phase 2 — Cora ablations**
   - use Cora-only ablations to freeze default settings before scaling
4. **Phase 3 — Cora downstream evaluation**
   - connect intrinsic quality to downstream GCN/GAT performance on the anchor dataset
5. **Phase 4 — Citeseer/Pubmed homophilic reproduction**
   - reuse the frozen Cora protocol without introducing new exploratory branches
6. **Phase 5 — OGBN-Arxiv scalability**
   - document runtime, convergence, and any resource-limit failures
7. **Phase 6 — Texas/Wisconsin heterophily stress test**
   - preserve the heterophilic extension as the final stress-test phase

**Emergency rule:** If time becomes tight, prioritize a complete Cora intrinsic + Cora downstream story before scaling to the full dataset set.

---

## 8. Scope Freeze

Do not add:
- Additional operators beyond the five primary + heat kernel reference
- Additional datasets beyond the six listed
- Additional backbone architectures
- Link prediction or graph classification tasks
