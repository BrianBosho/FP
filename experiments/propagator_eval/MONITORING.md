# Propagator Eval — Monitoring Log

This file is updated by lightweight monitoring snapshots.

## Live overview

| Field | Value |
|---|---|
| Current phase focus | Phase 1 — Cora intrinsic core |
| Current run | Phase 1 Cora sequential queue |
| Experiment log | `experiments/propagator_eval/results/phase_1_cora_intrinsic/logs/phase1_cora_queue.nohup.log` |
| Checklist | `experiments/propagator_eval/EXECUTION_CHECKLIST.md` |

## Snapshot log

### 2026-05-01 18:32 CAT

| Item | Status | Notes |
|---|---|---|
| Resources | safe for small intrinsic run | GPU and RAM headroom available; no downstream launch yet |
| First Cora smoke run | blocked then retried | foreground attempts exposed import + tolerance issues |
| Fixes applied | yes | import path fixed, tolerance type fixed |
| Next action | launch background diffusion smoke run | monitor every 10 minutes via file snapshots |

### 2026-05-01 18:33 CAT

| Item | Status | Notes |
|---|---|---|
| Smoke run process | running | pid 524446 |
| GPU | snapshot | 11429 MiB used / 32303 MiB free / 8% util |
| RAM | snapshot | 9.7Gi used / 28Gi free / 52Gi avail |
| Load | snapshot | 3.29, 5.48, 5.87 |
| Log tail | snapshot | Intrinsic eval — 1 runs [1/1] diffusion / Cora / beta=10000 / seed=0 |

### 2026-05-01 18:43 CAT

| Item | Status | Notes |
|---|---|---|
| Smoke run process | finished | pid 524446 |
| GPU | snapshot | 16341 MiB used / 27390 MiB free / 8% util |
| RAM | snapshot | 17Gi used / 19Gi free / 43Gi avail |
| Load | snapshot | 4.17, 5.17, 5.70 |
| Log tail | snapshot | [1/1] diffusion / Cora / beta=10000 / seed=0   saved → experiments/propagator_eval/results/phase_1_cora_intrinsic/raw/diffusion/cora/beta10000_seed0.json   done in 166.8s  mse=0.0133  recovery=-0.044 |
