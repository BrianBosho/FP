# FedProp Ablation & Research Tracker

This document tracks the systematic investigation of Feature Propagation (FP) strategies.

## 1. Directory Structure

| Type | Path | Description |
|---|---|---|
| **Ablation Configs** | `conf/ablations/` | Dedicated YAMLs for research sweeps. |
| **Core Configs** | `conf/` | Production-ready dataset configurations. |
| **Result Data** | `results/ablations/` | Raw JSON/CSV output from research runs. |
| **Diagnostic Logs** | `results/propagation_stats/` | Step-by-step FP metrics (residuals, energy, etc). |

---

## 2. Implementation Status

| Feature | Status | Implementation Detail |
|---|---|---|
| **Diagnostic Logging** | ✅ Done | Logs Residual, Norm Drift, Variance, Energy in `propagation.py`. |
| **Sweep Runner** | ✅ Done | `run_experiments.py` extended for `iter`, `t`, and `alpha` sweeps. |
| **Multi-Scale Fusion** | ⬜ Planned | Needs `propagate_features` update for weighted averaging. |
| **Structural Regularizer**| ⬜ Planned | Needs `FLClient` and `train.py` update for Laplacian loss. |
| **Adaptive Scale ($t$)** | ⬜ Planned | Needs subgraph diameter estimation in `partitioning.py`. |

---

## 3. Active Research Plan

### Phase 1: Diagnostic Calibration (Current)
**Goal:** Identify if internal FP diagnostics (Residual/Variance) correlate with downstream accuracy.
*   **Config:** `conf/ablations/A1_diagnostic_sweep.yaml`
*   **Sweep:** Iterations $\{5, 20, 50, 100\}$ × $t \{0.01, 0.1, 0.5\}$.
*   **Target:** Cora, Citeseer.

### Phase 2: Multi-Scale Weighted Fusion
**Goal:** Test if combining local and global structural information outperforms the best single-scale pass.
*   **Config:** `conf/ablations/A2_multiscale_fusion.yaml`
*   **Mechanism:** Weighted average of $T=\{5, 20, 50\}$.

### Phase 3: Structural Regularization
**Goal:** Evaluate the impact of the Laplacian consistency loss on non-IID partitions.
*   **Config:** `conf/ablations/A3_consistency_reg.yaml`

---

## 4. Completed Ablations
*None yet. Awaiting Phase 1 execution.*
