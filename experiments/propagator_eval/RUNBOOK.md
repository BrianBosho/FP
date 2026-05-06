# Propagator Evaluation — Operations Runbook

*Complete guide for an agent to pick up and run all propagator-eval experiments.*

---

## 0. What This Is

This is the experimental framework for the paper:

**"Intrinsic Evaluation of Propagation Operators for Communication-Free Subgraph Federated Learning"**

It evaluates 6 propagation operators (O1–O6) across 6 datasets in two tracks:
- **Layer 1–3** (intrinsic): propagation only, no GNN training. Fast (~4 hours total).
- **Layer 4** (downstream): full FL training. Slow (~110 hours total).

The authoritative design is `V2.md`. This runbook is the executable companion.

## Execution Order (authoritative)

We now use an **anchor-dataset-first** execution strategy.

Run in this order:

1. **Phase 1: Cora intrinsic core**
2. **Phase 2: Cora ablations**
3. **Phase 3: Cora downstream**
4. **Phase 4: Citeseer/Pubmed homophilic reproduction**
5. **Phase 5: OGBN-Arxiv scalability**
6. **Phase 6: Texas/Wisconsin heterophily stress test**

Do **not** launch the old full six-dataset matrix before Phases 1–3 are complete.

The new phase-based strategy doc lives at:

- `docs/EXPERIMENT_STRATEGY_ANCHOR_FIRST.md`

The new phase-based configs live alongside the legacy configs in `configs/`.

---

## 1. File Map

### 1.1 Files in this directory

```
experiments/propagator_eval/
├── V2.md                              # Locked experimental protocol (source of truth)
├── EXPERIMENT_OBJECTIVES.md           # Research questions, run matrix, scope
├── RESULTS_AND_IMPLEMENTATION.md      # JSON schemas, metric definitions, file-by-file code plan
├── RUNBOOK.md                         # THIS FILE — how to run everything
├── docs/
│   └── EXPERIMENT_STRATEGY_ANCHOR_FIRST.md  # Authoritative execution order
│
└── configs/
    ├── README.md                      # Phase-based config map
    ├── phase_1_cora_intrinsic.yaml
    ├── phase_1_cora_intrinsic_heat_kernel.yaml
    ├── phase_2_cora_ablation.yaml
    ├── phase_2_cora_ablation_appnp_alpha.yaml
    ├── phase_2_cora_ablation_epsilon_cora.yaml
    ├── phase_2_cora_ablation_hop_depth.yaml
    ├── phase_3_cora_downstream.yaml
    ├── phase_4_homophilic_reproduction.yaml
    ├── phase_5_scalability_ogbn_arxiv.yaml
    ├── phase_6_heterophily_stress.yaml
    ├── L1_L3_primary.yaml             # 270 runs: 5 ops × 6 datasets × 3 β × 3 seeds
    ├── L1_L3_heat_kernel_ref.yaml     # 2 runs: O5 on Cora + Citeseer only
    ├── L1_L3_hop2_ablation.yaml       # 45 runs: L=2 on Cora only
    ├── L4_downstream_operators.yaml   # 400 runs: 5 ops × 4 homophilic datasets × 2 β × 2 bb × 5 seeds
    ├── L4_downstream_heterophilic.yaml # 40 runs: 3 ops + zero_hop × Texas/Wisconsin × GCN × β=10000
    ├── L4_baselines.yaml              # 140 runs: zero_hop + oracle baselines
    └── ablations_intrinsic.yaml       # 42 runs: K, α, ε, hop-depth sensitivity sweeps
```

### 1.2 Modified source files (in repo root)

| File | Lines changed | What was added |
|---|---|---|
| `src/fedgnn/data/data_utils.py` | `get_row_normalized_adjacency()` at line 465, `heat_kernel_exact()` at line 493 | O2 and O5 operator kernels |
| `src/fedgnn/data/propagation.py` | Helper functions at lines 81–122, new modes at lines 363–392, intrinsic eval at lines 399–498 | `appnp`, `asymmetric_random_walk`, `heat_kernel_exact` modes; `intrinsic_eval`/`X_true` params; per-iter residuals; intrinsic metrics; wall-clock timing |
| `src/fedgnn/data/loaders.py` | 3 new `elif` branches at lines 113–125 | Dispatch for `appnp`, `asymmetric_random_walk`, `heat_kernel_exact` |
| `src/fedgnn/data/partitioning.py` | `return_masks` param at line 276, early return at line 485 | Returns boolean masks per client when `return_masks=True` |
| `src/fedgnn/experiments/run_experiments.py` | 3 new entries in `FEATURE_PROP_DATA_LOADING_OPTIONS` at lines 58–61 | Validates new modes in the existing experiment runner |
| `conf/intrinsic_eval.yaml` | New file | Base defaults shared by both eval runners |

### 1.3 New runner scripts (in repo root)

| File | Purpose | CLI |
|---|---|---|
| `src/fedgnn/experiments/run_intrinsic_eval.py` | Layer 1–3 (propagation only) | `python -m src.fedgnn.experiments.run_intrinsic_eval --config <yaml> [filters]` |
| `src/fedgnn/experiments/run_downstream_eval.py` | Layer 4 (full FL training) | `python -m src.fedgnn.experiments.run_downstream_eval --config <yaml> [filters]` |

---

## 2. Operators

| ID | Code name | Fixed point | Dirichlet? | Where implemented |
|---|---|---|---|---|
| O1 | `adjacency` | Harmonic extension | YES | `propagation.py` mode dispatch, existing |
| O2 | `asymmetric_random_walk` | Asymmetric smoothing | NO | `propagation.py` line 369 + `data_utils.py:get_row_normalized_adjacency()` |
| O3 | `diffusion` | Harmonic extension | YES | `propagation.py`, existing |
| O4 | `chebyshev_diffusion` | Heat kernel steady state | APPROX | `propagation.py`, existing |
| O5 | `heat_kernel_exact` | Exact heat kernel | APPROX | `propagation.py` line 374 + `data_utils.py:heat_kernel_exact()` |
| O6 | `appnp` | Personalized PageRank | NO | `propagation.py` line 363 |

### Operator update rules (inside iteration loop)

- **O1, O3, O4, O2**: `out = α·(P@out) + (1-α)·out`, then `out[mask] = x[mask]`
- **O6 (APPNP)**: `out = (1-α_p)·(Â@out) + α_p·X₀`, then `out[mask] = x[mask]`
  - `α_p` = `appnp_alpha` from config (default 0.1). NOT the same as `alpha`.
  - `X₀` = initial features saved before the loop.
- **O5 (heat kernel)**: Single-shot `out = H @ out`, then `out[mask] = x[mask]`. No iteration loop.

### Key parameters

| Param | Default | Controls | Used by |
|---|---|---|---|
| `alpha` | 0.5 | Blend weight (diffused vs previous) | O1, O2, O3, O4 |
| `appnp_alpha` | 0.1 | Teleport probability to X₀ | O6 |
| `diffusion_t` | 0.1 | Diffusion time | O3, O5 |
| `chebyshev_k` | 5 | Chebyshev polynomial order | O4 |
| `chebyshev_t` | 0.9 | Chebyshev diffusion time | O4 |
| `num_iterations` | 100 (L1-3), 50 (L4) | Max iterations | All iterative ops |
| `feature_prop_tolerance` | 1e-4 | Convergence tolerance | All iterative ops |

---

## 3. Datasets

| Name | Code name | Nodes | Homophily | Loader |
|---|---|---|---|---|
| Cora | `Cora` | 2,708 | 0.81 | `Planetoid` via `datasets.py` |
| Citeseer | `Citeseer` | 3,327 | 0.74 | `Planetoid` via `datasets.py` |
| Pubmed | `Pubmed` | 19,717 | 0.80 | `Planetoid` via `datasets.py` |
| OGBN-Arxiv | `OGBN-Arxiv` | 169,343 | 0.66 | OGB via `datasets.py` |
| Texas | `Texas` | 183 | 0.11 | `WebKB` via `datasets.py` |
| Wisconsin | `Wisconsin` | 251 | 0.20 | `WebKB` via `datasets.py` |

All already supported. `datasets.py` line 57 maps `"webkb": ["Texas", "Wisconsin"]`.

---

## 4. Run Matrix — Exact Counts

### 4.1 Layer 1–3 (intrinsic, no training)

| Config file | Runs | What |
|---|---|---|
| `L1_L3_primary.yaml` | 270 | 5 ops × 6 ds × 3 β × 3 seeds |
| `L1_L3_heat_kernel_ref.yaml` | 2 | O5 × Cora+Citeseer × β=10000 × seed 0 |
| `L1_L3_hop2_ablation.yaml` | 45 | 5 ops × Cora × 3 β × 3 seeds, hop=2 |
| `ablations_intrinsic.yaml` | 42 | K/α/ε/hop sensitivity sweeps |
| **Total L1–3** | **359** | ~4 hours |

### 4.2 Layer 4 (downstream, FL training)

| Config file | Runs | What |
|---|---|---|
| `L4_downstream_operators.yaml` | 400 | 5 ops × 4 ds × 2 β × 2 bb × 5 seeds |
| `L4_baselines.yaml` | 140 | zero_hop (80) + oracle (60, no OGBN) |
| `L4_downstream_heterophilic.yaml` | 40 | 3 ops + zero_hop × Texas/Wisc × GCN × β=10000 × 5 seeds |
| **Total L4** | **580** | ~110 hours |

### 4.3 Grand total: 939 runs, ~115 hours

---

## 5. How to Run — Step by Step

### Prerequisites

- Python 3.10+ with PyTorch, PyG, torch_scatter, torch_sparse
- CUDA GPU (>= 8 GB VRAM for OGBN-Arxiv)
- Working directory: `/home/bosho/FP`
- `PYTHONPATH` includes repo root (or run from repo root)

### Phase 1 quick start (current preferred entry point)

```bash
cd /home/bosho/FP

# Cora intrinsic core
python -m src.fedgnn.experiments.run_intrinsic_eval \
    --config experiments/propagator_eval/configs/phase_1_cora_intrinsic.yaml

# Cora heat-kernel reference companion
python -m src.fedgnn.experiments.run_intrinsic_eval \
    --config experiments/propagator_eval/configs/phase_1_cora_intrinsic_heat_kernel.yaml
```

Write findings into:

```text
experiments/propagator_eval/results/phase_1_cora_intrinsic/notes/phase_1_findings.md
```

### Phase 1 — Smoke test (5 minutes)

Validate one run of each new operator before launching the full grid.

```bash
cd /home/bosho/FP

# Test each new operator on Cora, β=10000, seed=0
python -m src.fedgnn.experiments.run_intrinsic_eval \
    --config experiments/propagator_eval/configs/L1_L3_primary.yaml \
    --operator appnp --dataset Cora --beta 10000 --seed 0

python -m src.fedgnn.experiments.run_intrinsic_eval \
    --config experiments/propagator_eval/configs/L1_L3_primary.yaml \
    --operator asymmetric_random_walk --dataset Cora --beta 10000 --seed 0

python -m src.fedgnn.experiments.run_intrinsic_eval \
    --config experiments/propagator_eval/configs/L1_L3_heat_kernel_ref.yaml \
    --operator heat_kernel_exact --dataset Cora --beta 10000 --seed 0
```

**Expected output per run:**
- Printed: `mse=... recovery=...`
- JSON file at `experiments/propagator_eval/results/intrinsic/{operator}/cora/beta10000_seed0.json`

**If it fails:** Check the error. Common issues:
- Missing `appnp_alpha` in config → add to the YAML or pass via `--config`
- `heat_kernel_exact` OOM on non-small graph → only run on Cora/Citeseer
- `torch_sparse` import error → install PyG dependencies

### Phase 2 — Run full Layer 1–3 (~4 hours)

Run in this order:

```bash
# 1. Primary grid (270 runs) — all operators on all datasets
python -m src.fedgnn.experiments.run_intrinsic_eval \
    --config experiments/propagator_eval/configs/L1_L3_primary.yaml

# 2. Heat kernel reference (2 runs) — small graphs only
python -m src.fedgnn.experiments.run_intrinsic_eval \
    --config experiments/propagator_eval/configs/L1_L3_heat_kernel_ref.yaml

# 3. Hop-2 ablation (45 runs) — Cora only
python -m src.fedgnn.experiments.run_intrinsic_eval \
    --config experiments/propagator_eval/configs/L1_L3_hop2_ablation.yaml

# 4. Ablations (42 runs) — K, α, ε, hop sensitivity
python -m src.fedgnn.experiments.run_intrinsic_eval \
    --config experiments/propagator_eval/configs/ablations_intrinsic.yaml
```

**You can also filter to a single operator/dataset for debugging:**

```bash
python -m src.fedgnn.experiments.run_intrinsic_eval \
    --config experiments/propagator_eval/configs/L1_L3_primary.yaml \
    --operator diffusion --dataset OGBN-Arxiv
```

### Phase 3 — Run Layer 4 baselines first (~20 hours)

Zero-hop and oracle baselines are needed before gap-closed can be computed.

```bash
python -m src.fedgnn.experiments.run_downstream_eval \
    --config experiments/propagator_eval/configs/L4_baselines.yaml
```

### Phase 4 — Run Layer 4 operator sweeps (~90 hours)

```bash
# Homophilic operators (400 runs)
python -m src.fedgnn.experiments.run_downstream_eval \
    --config experiments/propagator_eval/configs/L4_downstream_operators.yaml

# Heterophilic validation (40 runs)
python -m src.fedgnn.experiments.run_downstream_eval \
    --config experiments/propagator_eval/configs/L4_downstream_heterophilic.yaml
```

### Phase 5 — Compute gap-closed (post-hoc)

After ALL Layer 4 runs complete:

```bash
python -m src.fedgnn.experiments.run_downstream_eval \
    --config experiments/propagator_eval/configs/L4_downstream_operators.yaml \
    --compute-gaps
```

This reads the zero_hop and oracle baseline JSON files and fills `accuracy_gap_closed`, `zero_hop_accuracy`, `oracle_accuracy` in every operator result file.

---

## 6. Output File Layout

All results go under `experiments/propagator_eval/results/`.

```
results/
├── intrinsic/                                    # Layer 1-3
│   ├── adjacency/
│   │   ├── cora/beta10000_seed0.json
│   │   ├── cora/beta10_seed0.json
│   │   ├── citeseer/beta10000_seed0.json
│   │   └── ...
│   ├── asymmetric_random_walk/
│   ├── diffusion/
│   ├── chebyshev_diffusion/
│   ├── appnp/
│   └── heat_kernel_exact/
│       ├── cora/beta10000_seed0.json
│       └── citeseer/beta10000_seed0.json
│
├── intrinsic_hop2/                               # L=2 ablation
│   └── {operator}/cora/beta{beta}_seed{seed}.json
│
├── ablations/                                    # Sensitivity sweeps
│   └── {operator}/{dataset}/beta{beta}_seed{seed}.json
│
└── downstream/                                   # Layer 4
    ├── adjacency/cora/beta10000_seed0_gcn.json
    ├── zero_hop/cora/beta1_seed2_gat.json
    ├── full/cora/beta10000_seed0_gcn.json        # Oracle
    ├── appnp/texas/beta10000_seed0_gcn.json      # Heterophilic
    └── ...
```

### Intrinsic JSON schema

```json
{
  "operator": "diffusion",
  "dataset": "cora",
  "beta": 10000,
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
      "residuals": [6.2, 4.1, 2.8, 1.9]
    }
  ],
  "aggregate": {
    "mse_mean": 0.036,
    "mse_std": 0.008,
    "cosine_sim_mean": 0.85,
    "recovery_ratio_mean": 0.59,
    "boundary_coverage_mean": 0.71,
    "n_iters_mean": 36.2,
    "wall_time_total_sec": 1.4,
    "convergence_rate": 1.0
  }
}
```

### Downstream JSON schema

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
  "per_client_accuracy": [0.81, 0.79, ...],
  "wall_time_sec": 42.3
}
```

`accuracy_gap_closed` = `(acc_op - acc_zero) / (acc_oracle - acc_zero)`. Filled post-hoc by `--compute-gaps`.

---

## 7. Per-Dataset Hyperparameters (Layer 4)

The downstream runner reads these from the `training_per_dataset` block in each YAML config.

| Dataset | Backbone | Optimizer | LR | Weight Decay |
|---|---|---|---|---|
| Cora | GCN | SGD | 0.5 | 5e-4 |
| Cora | GAT | Adam | 0.005 | 5e-4 |
| Citeseer | GCN | SGD | 0.5 | 5e-4 |
| Citeseer | GAT | Adam | 0.005 | 5e-4 |
| Pubmed | GCN | Adam | 0.01 | 5e-4 |
| Pubmed | GAT | Adam | 0.01 | 5e-4 |
| OGBN-Arxiv | GCN | Adam | 0.01 | 0.0 |
| OGBN-Arxiv | GAT | Adam | 0.01 | 0.0 |
| Texas | GCN | Adam | 0.01 | 5e-4 |
| Wisconsin | GCN | Adam | 0.01 | 5e-4 |

Model architecture per backbone:
- **GCN**: hidden=16, layers=2, dropout=0.5 (32 hidden for Texas/Wisconsin)
- **GAT**: hidden=8, layers=2, heads=8, dropout=0.6

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Unknown propagation mode: appnp` | Stale `__pycache__` | `find src -name __pycache__ -exec rm -rf {} +` |
| `heat_kernel_exact` OOM | Graph too large | Only use on Cora (2708) and Citeseer (3327). Already enforced by 10k safety limit in the function. |
| `main_experiment()` config shape error | Missing top-level key in run_cfg | Check `_build_run_config()` in `run_downstream_eval.py` adds all required keys |
| JSON result file not created | Output dir permissions | `mkdir -p experiments/propagator_eval/results` |
| `torch_sparse` import error | PyG not installed | `pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{version}.html` |
| `return_masks` gives wrong mask | Client has no `mapping` attribute | Fallback uses `(x == 0).all(dim=1)` — only correct if features were zeroed before calling. For intrinsic eval this is correct since `x_init[~mask] = 0`. |

---

## 9. Validation Checklist

Before trusting results, verify:

- [ ] **Smoke test passes**: Each of the 3 new operators produces a JSON with `mse > 0` and `recovery_ratio` between 0 and 1 on Cora.
- [ ] **Residuals decay**: For O1 (adjacency) and O3 (diffusion), the `residuals` list in the JSON should decrease monotonically.
- [ ] **APPNP converges**: O6 should converge but residuals may NOT decrease monotonically (expected — PPR fixed point is not the Dirichlet minimizer).
- [ ] **Heat kernel is best intrinsic**: O5 should have the lowest MSE on Cora/Citeseer (it's the exact solution).
- [ ] **Downstream accuracy**: O3 (diffusion) and O1 (adjacency) should match or exceed results from the existing `run_experiments.py` on the same config — sanity check that the new runner doesn't break training.
- [ ] **Gap-closed in [0, 1]**: After `--compute-gaps`, all operator files should have `accuracy_gap_closed` between 0 and 1. Values > 1 mean the operator beat oracle (possible on small graphs, investigate).

---

## 10. Execution Priority

| Priority | Phase | Time | Command |
|---|---|---|---|
| 1 | Smoke test (3 runs) | 5 min | Phase 1 above |
| 2 | L1-3 primary (270 runs) | ~3 hr | `--config L1_L3_primary.yaml` |
| 3 | L1-3 heat kernel (2 runs) | 1 min | `--config L1_L3_heat_kernel_ref.yaml` |
| 4 | L1-3 hop-2 (45 runs) | 30 min | `--config L1_L3_hop2_ablation.yaml` |
| 5 | L4 baselines (140 runs) | ~20 hr | `--config L4_baselines.yaml` |
| 6 | L4 operators (400 runs) | ~80 hr | `--config L4_downstream_operators.yaml` |
| 7 | L4 heterophilic (40 runs) | ~1 hr | `--config L4_downstream_heterophilic.yaml` |
| 8 | Gap-closed computation | 1 min | `--compute-gaps` |
| 9 | Ablations (42 runs) | 30 min | `--config ablations_intrinsic.yaml` |

**Emergency rule:** If Phase 4 (L4) is not finished by June 5, submit with Phase 2 (L1-3) results only and note downstream as preliminary.
