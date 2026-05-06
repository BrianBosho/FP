# Propagator Eval — Execution Checklist

_Last updated: 2026-05-01 18:26 CAT_

## Current resource snapshot

- GPU: NVIDIA L40-48Q
  - total: 49,152 MiB
  - used: 10,627 MiB
  - free: 33,105 MiB
  - utilization: 6%
- RAM:
  - total: 62 GiB
  - available: 53 GiB
- CPU:
  - load average: 4.22 / 5.76 / 5.89
- Other active jobs detected:
  - `python scripts/run_robustness_multiseed.py --llm-model qwen2.5-7b-instruct`
  - `/home/bosho/.conda/envs/fedgnn/bin/python -m src.experiments.run_experiments --config experiments/configs/rerun/R1_cora_missing.yaml`

## Monitoring table

| Item | Status | Done | Remaining | Notes |
|---|---|---:|---:|---|
| Phase 1 Cora smoke run | completed | 1 | 0 | diffusion / Cora / beta10000 / seed0 succeeded |
| Phase 1 Cora intrinsic core | completed | 45 | 0 | full 5 operators × 3 betas × 3 seeds finished |
| Phase 1 Cora heat-kernel reference | pending | 0 | 1 | separate companion run on CPU |
| Phase 2 Cora ablations | pending | 0 | 30 runs | appnp alpha, epsilon, hop-depth |
| Phase 3 Cora downstream | pending | 0 | 140 runs | zero-hop + full + 5 operators |
| Phase 4 homophilic reproduction | pending | 0 | 280 runs | Citeseer + Pubmed |
| Phase 5 scalability | pending | 0 | 30 runs | OGBN-Arxiv intrinsic block |
| Phase 6 heterophily stress | pending | 0 | 40 runs | Texas + Wisconsin downstream |

## Safe-start decision

- [x] Resources are sufficient for a **small, low-impact Cora intrinsic run**.
- [x] Resources are **not** reserved for a full downstream / Ray-heavy launch right now.
- [x] Start with Cora only.
- [x] Use low thread counts and low process priority.

## What has been done

- [x] Reorganized the plan around an anchor-dataset-first strategy.
- [x] Added phase-based configs.
- [x] Added strategy and runbook updates.
- [x] Created a Cora findings template.
- [x] Created shareable PDFs for the execution plan and experiment explainer.
- [x] Verified the current box has enough headroom for a small start.
- [x] Fixed intrinsic runner import path (`datasets` → `loaders`).
- [x] Fixed numeric tolerance parsing in phase configs (`1e-4` → `1.0e-4`).

## What is pending

### Phase 1 — Cora intrinsic core
- [x] Run first smoke test on Cora
- [x] Identify and fix intrinsic runner import bug
- [x] Identify and fix numeric tolerance parsing bug
- [x] Verify JSON output lands in the phase-1 results tree
- [x] Run additional Cora operators after smoke test passes
- [ ] Summarize first findings in `results/phase_1_cora_intrinsic/notes/phase_1_findings.md`

### Phase 2 — Cora ablations
- [ ] APPNP alpha sweep
- [ ] epsilon sweep on Cora
- [ ] hop-depth comparison
- [ ] freeze defaults

### Phase 3 — Cora downstream
- [ ] zero-hop baseline
- [ ] oracle/full baseline
- [ ] operator runs
- [ ] intrinsic/downstream join

### Later phases
- [ ] Citeseer/Pubmed reproduction
- [ ] OGBN-Arxiv scalability
- [ ] Texas/Wisconsin heterophily stress test

## First run target

Current first run:

- phase: `phase_1_cora_intrinsic`
- operator: `diffusion`
- dataset: `Cora`
- beta: `10000`
- seed: `0`

Expected output path:

- `experiments/propagator_eval/results/phase_1_cora_intrinsic/raw/diffusion/Cora/beta10000_seed0.json`

## Notes

- Live monitoring file: `experiments/propagator_eval/MONITORING.md`
- Sequential runner status file: `experiments/propagator_eval/SEQUENTIAL_RUNNER_STATUS.md`
- Sequential runner script: `experiments/propagator_eval/run_remaining_sequential.py`
- Active beta policy for future runs: **β = 10000** and **β = 1** only

Phase 1 Cora intrinsic is complete. The next pending item is the Cora heat-kernel reference, followed by Phase 2 ablations.
