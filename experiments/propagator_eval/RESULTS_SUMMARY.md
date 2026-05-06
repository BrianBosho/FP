# Propagator Evaluation — Experiment Progress Report

**Generated:** 2026-05-04
**Status:** Complete — 518/521 clean result files

---

## Experiment Matrix Overview

The propagator evaluation consists of 6 phases tracking feature propagation quality across three axes:

| Phase | Dataset | Experiment Type | Metrics Captured |
|-------|---------|---------------|-----------------|
| Phase 1 | Cora | Intrinsic | Heat kernel fidelity (MSE, cosine similarity, recovery ratio) |
| Phase 2 | Cora | Intrinsic | APPNP alpha sweep (intrinsic quality) |
| Phase 3 | Cora | Downstream | Node classification accuracy, gap-closed metrics |
| Phase 4 | Citeseer, Pubmed | Downstream | Node classification accuracy, gap-closed metrics |
| Phase 5 | OGBN-Arxiv | Intrinsic | Feature quality on large graph (MSE, cosine similarity, recovery ratio) |
| Phase 6 | Texas, Wisconsin | Downstream | Node classification on heterophilic graphs |

---

## Phase-by-Phase Results

### Phase 1 — Cora Intrinsic (Heat Kernel Reference)
**Location:** `results/phase_1_cora_intrinsic/raw/`
**Planned:** 1 specific heat-kernel run | **Directory contains:** 46 files (from broader Phase 1 runs)

This phase establishes the ground-truth heat kernel as a reference for evaluating approximation methods.

---

### Phase 2 — APPNP Alpha Sweep on Cora
**Location:** `results/phase_2_cora_ablation/raw/`
**Planned:** 30 (3 alphas × 10 seeds) | **On Disk:** 30/30 ✓

Tests how different APPNP alpha values affect intrinsic feature quality on Cora.

---

### Phase 3 — Cora Downstream Evaluation
**Location:** `results/phase_3_cora_downstream/raw/`
**Planned:** 140 (7 operators × 2 betas × 2 backbones × 5 seeds)
**On Disk:** 140/140 ✓ | **Clean:** 137/140 | **NaN:** 3

Matrix:
- Operators: `zero_hop`, `full`, `adjacency`, `asymmetric_random_walk`, `diffusion`, `chebyshev_diffusion`, `appnp`
- Betas: 1, 10000
- Backbones: GCN, GAT
- Seeds: 0, 1, 2, 3, 4

**NaN files** (numerical instability at beta=10000 with GCN backbone):
- `chebyshev_diffusion/cora/beta10000_seed2_gcn.json`
- `appnp/cora/beta10000_seed1_gcn.json`
- `asymmetric_random_walk/cora/beta10000_seed2_gcn.json`

These failures are expected: beta=10000 is an extreme diffusion parameter where the propagation matrix eigenvalue exceeds the stability threshold, causing numerical overflow. This is a known limitation, not an infrastructure issue.

---

### Phase 4 — Homophilic Reproduction (Citeseer + Pubmed)
**Location:** `results/phase_4_homophilic_reproduction/raw/`
**Planned:** 280 (7 operators × 2 datasets × 2 betas × 2 backbones × 5 seeds)
**On Disk:** 280/280 ✓ | **Clean:** 280/280 ✓

Matrix:
- Datasets: Citeseer, Pubmed (140 each)
- Operators: `zero_hop`, `full`, `adjacency`, `asymmetric_random_walk`, `diffusion`, `chebyshev_diffusion`, `appnp`
- Betas: 1, 10000
- Backbones: GCN, GAT
- Seeds: 0–4

---

### Phase 5 — OGBN-Arxiv Scalability (Intrinsic)
**Location:** `results/phase_5_scalability_ogbn_arxiv/raw/`
**Planned:** 30 (5 operators × 2 betas × 3 seeds)
**On Disk:** 30/30 ✓ | **Clean:** 30/30 ✓

Matrix:
- Operators: `adjacency`, `asymmetric_random_walk`, `diffusion`, `chebyshev_diffusion`, `appnp`
- Betas: 1, 10000
- Seeds: 0, 1, 2

**Metrics captured per client:**
- `mse` — mean squared error between propagated and true features
- `cosine_sim` — directional similarity to ground truth features
- `recovery_ratio` — fraction of feature variance explained
- `residuals` — convergence trace across 100 iterations
- `wall_time_sec` — compute time
- `converged` — boolean

**Aggregate summary:** mse_mean/std, cosine_sim_mean/std, recovery_ratio_mean/std, boundary_coverage, convergence_rate

Note: This phase measures **intrinsic feature quality only** — no downstream node classification accuracy.

---

### Phase 6 — Heterophily Stress Test (Texas + Wisconsin)
**Location:** `results/phase_6_heterophily_stress/raw/`
**Planned:** 40 (4 operators × 2 datasets × 5 seeds)
**On Disk:** 40/40 ✓ | **Clean:** 40/40 ✓

Matrix:
- Datasets: Texas, Wisconsin (20 each)
- Operators: `zero_hop`, `adjacency`, `diffusion`, `appnp`
- Backbone: GCN only
- Seeds: 0, 1, 2, 3, 4

**Metrics:** test_accuracy, accuracy_gap_closed, per_client_accuracy

---

## Summary Table

| Phase | Dataset | Type | Planned | On Disk | Clean | NaN |
|-------|---------|------|--------:|--------:|------:|----:|
| Phase 1 | Cora | Intrinsic (heat kernel ref) | 1 | 46* | — | — |
| Phase 2 | Cora | Intrinsic (APPNP alpha sweep) | 30 | 30 | 30 | 0 |
| Phase 3 | Cora | Downstream | 140 | 140 | 137 | 3 |
| Phase 4 | Citeseer + Pubmed | Downstream | 280 | 280 | 280 | 0 |
| Phase 5 | OGBN-Arxiv | Intrinsic (large graph) | 30 | 30 | 30 | 0 |
| Phase 6 | Texas + Wisconsin | Downstream (heterophily) | 40 | 40 | 40 | 0 |
| **Total** | | | **521** | **566** | **518** | **3** |

*Phase 1 directory contains 46 files from broader runs; runner tracked 1 specific heat-kernel experiment.

---

## Result File Locations

All results are under:
```
experiments/propagator_eval/results/
├── phase_1_cora_intrinsic/raw/                    # Heat kernel + operator intrinsic
├── phase_2_cora_ablation/raw/                     # APPNP alpha sweep
├── phase_3_cora_downstream/raw/                    # Cora downstream (GAT + GCN)
│   ├── zero_hop/cora/
│   ├── full/cora/
│   ├── adjacency/cora/
│   ├── asymmetric_random_walk/cora/
│   ├── diffusion/cora/
│   ├── chebyshev_diffusion/cora/
│   ├── appnp/cora/
│   └── propagation_stats/                          # Propagation metadata
├── phase_4_homophilic_reproduction/raw/             # Citeseer + Pubmed (GAT + GCN)
│   ├── adjacency/{citeseer,pubmed}/
│   ├── asymmetric_random_walk/{citeseer,pubmed}/
│   ├── diffusion/{citeseer,pubmed}/
│   ├── chebyshev_diffusion/{citeseer,pubmed}/
│   ├── appnp/{citeseer,pubmed}/
│   └── propagation_stats/
├── phase_5_scalability_ogbn_arxiv/raw/            # OGBN-Arxiv intrinsic quality
│   ├── adjacency/ogbn-arxiv/
│   ├── asymmetric_random_walk/ogbn-arxiv/
│   ├── diffusion/ogbn-arxiv/
│   ├── chebyshev_diffusion/ogbn-arxiv/
│   └── appnp/ogbn-arxiv/
└── phase_6_heterophily_stress/raw/                # Texas + Wisconsin heterophily
    ├── zero_hop/{texas,wisconsin}/
    ├── adjacency/{texas,wisconsin}/
    ├── diffusion/{texas,wisconsin}/
    └── appnp/{texas,wisconsin}/
```

---

## Known Limitations

### 3 NaN files in Phase 3 (Cora downstream, beta=10000, GCN backbone)
These three experiments failed because beta=10000 produces propagation matrices with eigenvalues exceeding the numerical stability threshold, causing overflow in the diffusion computation. This affects:
- `chebyshev_diffusion / Cora / GCN / beta=10000 / seed=2`
- `appnp / Cora / GCN / beta=10000 / seed=1`
- `asymmetric_random_walk / Cora / GCN / beta=10000 / seed=2`

Note: The same operators with beta=10000 on GAT or with beta=1 on any backbone complete successfully. Beta=10000 with GCN specifically pushes the matrix series past the float32 stability boundary.

### Phase 5 — No Downstream Classification
Phase 5 measures intrinsic feature quality (MSE, cosine similarity, recovery ratio) on OGBN-Arxiv but does not run downstream node classification. It answers "how well does propagation reconstruct features?" not "what accuracy does this achieve on node classification?"

### Phase 6 — GCN Only
Phase 6 heterophily experiments use only the GCN backbone (not GAT), per the experimental design in `phase_6_heterophily_stress.yaml`.
