# Propagator Eval — Anchor-Dataset-First Strategy

## Core Decision

We are no longer treating the full six-dataset matrix as the first execution step.

The execution order is now:

1. **Phase 1 — Cora intrinsic core**
2. **Phase 2 — Cora ablations**
3. **Phase 3 — Cora downstream evaluation**
4. **Phase 4 — Citeseer/Pubmed homophilic reproduction**
5. **Phase 5 — OGBN-Arxiv scalability**
6. **Phase 6 — Texas/Wisconsin heterophily stress test**

This preserves the full paper scope while making the project easier to debug, analyze, and write.

## Why Cora First?

Cora is the anchor dataset because it is:

- small enough for fast iteration,
- standard in propagation and GNN work,
- homophilic enough for operator differences to be interpretable,
- easy to visualize and debug,
- suitable for locking the metric suite and analysis template.

## What Freezes Early

Before we scale to more datasets, the following must be fixed on Cora:

- intrinsic metrics: `mse`, `cosine_sim`, `recovery_ratio`, `boundary_coverage`, `iteration_count`, `residuals`, `convergence_flag`, `wall_clock_time`
- default operator hyperparameters
- default hop/depth settings
- default partition settings
- the post-processing table/plot template

`boundary_coverage` is required.

`spectral_fidelity` is deferred unless a later analysis explicitly needs it.

Active plan note: use only **β = 10000** and **β = 1**. Do not schedule new runs at **β = 10** unless we deliberately revise the protocol.

## Phase Map

### Phase 1 — Cora intrinsic core

Goal: compare all propagators on Cora only and generate the first paper-quality intrinsic analysis.

Primary config:
- `configs/phase_1_cora_intrinsic.yaml`

Companion reference config:
- `configs/phase_1_cora_intrinsic_heat_kernel.yaml`

Expected outcome:
- stable JSON outputs,
- first comparison tables,
- first operator plots,
- `results/phase_1_cora_intrinsic/notes/phase_1_findings.md`

### Phase 2 — Cora ablations

Goal: determine which knobs materially affect conclusions before freezing the protocol.

Manifest:
- `configs/phase_2_cora_ablation.yaml`

Runnable companion configs:
- `configs/phase_2_cora_ablation_appnp_alpha.yaml`
- `configs/phase_2_cora_ablation_epsilon_cora.yaml`
- `configs/phase_2_cora_ablation_hop_depth.yaml`

Note: the legacy `ablations_intrinsic.yaml` is still kept for reference, but the new phase files are the authoritative execution order.

### Phase 3 — Cora downstream

Goal: connect intrinsic quality to downstream FL performance on the same anchor dataset.

Primary config:
- `configs/phase_3_cora_downstream.yaml`

### Phase 4 — Homophilic reproduction

Goal: reproduce the frozen Cora protocol on Citeseer and Pubmed.

Primary config:
- `configs/phase_4_homophilic_reproduction.yaml`

### Phase 5 — OGBN-Arxiv scalability

Goal: measure runtime/convergence practicality at larger scale.

Primary config:
- `configs/phase_5_scalability_ogbn_arxiv.yaml`

### Phase 6 — Heterophily stress test

Goal: evaluate whether propagation helps, hurts, or destabilizes under weak homophily.

Primary config:
- `configs/phase_6_heterophily_stress.yaml`

## Results Layout

The new configs write into phase-based result roots:

```text
results/
  phase_1_cora_intrinsic/
    raw/
    processed/
    plots/
    logs/
    notes/
  phase_2_cora_ablation/
  phase_3_cora_downstream/
  phase_4_homophilic_reproduction/
  phase_5_scalability_ogbn_arxiv/
  phase_6_heterophily_stress/
```

The current runners still use the legacy per-operator/per-dataset JSON layout *inside* each phase's `raw/` directory. That is acceptable for now; phase-level post-processing can normalize this later.

## Stop/Go Gates

- **After Phase 1:** do not proceed unless the Cora intrinsic outputs, tables, and plots are complete.
- **After Phase 2:** do not proceed unless default settings are frozen.
- **After Phase 3:** do not proceed unless intrinsic and downstream results can be joined into one Cora narrative.
- **After Phase 4:** do not proceed unless reproduction on Citeseer/Pubmed is interpretable.

## Immediate Next Step

Run Phase 1 on Cora first.

Do **not** launch the old full-matrix configs until Phases 1–3 are complete.
