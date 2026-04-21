# Federated‑GNN — Performance‑Impacting Issues Checklist

Companion to [`FL_IMPLEMENTATION_REVIEW.md`](./FL_IMPLEMENTATION_REVIEW.md).
This checklist filters the full review down to issues that materially
affect **model performance** (test accuracy, convergence, generalization).
Runtime / compute‑efficiency issues are listed separately at the bottom.

For each item:

- **Impact** = expected effect on reported accuracy numbers.
- **Effort** = relative size of the fix (Trivial / Easy / Medium).
- **Confidence** = how sure we are the issue is actually biting current
  experiments (High / Medium / Low).
- **Status** = `[ ]` open, `[~]` in progress, `[x]` done.

Use this file as the working tracker; link PRs next to each item.

**Current review update (2026-04-21):** Some items have partial code in place,
but none of the high-impact `[~]` items should be treated as publication-ready
until the remaining notes under each item are resolved.

---

## A. Correctness bugs that silently skew accuracy (fix first)

These are bugs where a code path is either wrong or broken, and results
you have already produced may reflect the bug rather than the science.

- [x] **A1. `GAT_Arxiv` missing from `evaluate`/`test` (train.py §2.3)**
  - Symptom: any run with `GAT_Arxiv` crashes on validation or test.
    If no crash observed, confirm which GAT class is actually used on
    `ogbn-arxiv` (config uses `use_unified_model: false` → `GAT_Arxiv`).
  - Impact: **High** — all reported ogbn‑arxiv GAT numbers must be
    re-validated.
  - Effort: Trivial. Add `GAT_Arxiv` to the isinstance tuples in
    `evaluate`, `test`, `evaluate_with_minibatch`, `test_with_minibatch`.
  - Confidence: High.

- [x] **A2. `PubmedGAT` output layer produces `num_classes * 8` logits
      (models.py §7.1)**
  - Symptom: final layer uses `heads=8` with default `concat=True`.
    `log_softmax` spreads probability across 8× as many columns as classes.
  - Impact: **High** on Pubmed accuracy — probability mass bleeds into
    phantom classes.
  - Effort: Trivial. Either `heads=1`, or `heads=8, concat=False`.
  - Confidence: High.

- [x] **A3. `page_rank` propagation mode was crashy and is still not
      publication-scale (propagation_functions.py §5.3)**
  - Current status: the old `current_node` `NameError` is fixed. Added a
    size guard that raises `ValueError` for graphs >50k nodes (use
    `random_walk` or `chebyshev_diffusion` instead).
  - Impact: High if ever used on real graph sizes; it may OOM or be
    prohibitively slow.
  - Effort: Easy for a small-graph guard; Medium for a sparse implementation.
  - Confidence: High.

- [x] **A4. Early‑stopping logic resets patience against stale bests
      (run.py §3.1)**
  - Symptom: `best_eval_loss` is only updated in the `elif`; `best_eval_acc`
    only in the `if`. Patience can reset spuriously, stretching training
    past the real best epoch, or stopping too early.
  - Impact: Medium on reported best accuracy, especially when training
    is long and noisy (ogbn‑arxiv, Pubmed GAT).
  - Effort: Trivial — pick loss as sole criterion, or track both cleanly.
  - Confidence: High.

- [x] **A5. Val/test mask division by zero on highly non‑IID shards
      (train.py §2.4)**
  - Symptom: `acc = int(correct) / int(data.val_mask.sum())` when a
    client has zero val/test nodes → `ZeroDivisionError` or `inf` acc.
  - Impact: Can crash runs at small β, or skew client‑average metrics.
  - Effort: Trivial — guard with `max(sum, 1)` and return NaN.
  - Confidence: Medium (only triggers at low β with many clients).

---

## B. Federated aggregation — first‑order correctness

These directly change the fixed point of the federated training and are
the most likely sources of systematic accuracy gaps vs. literature.

- [~] **B1. Unweighted averaging instead of FedAvg (server.py §1.1)**
  - Current status: `_aggregate_fedavg_weighted` and
    `FLClient.get_num_train_samples()` exist, but `conf/base.yaml` still
    defaults to `aggregation: "mean"`.
  - Symptom: publication configs that do not opt in still use
    `p.data /= self.num_of_trainers` and ignore `n_k`.
  - Impact: **High** under Dirichlet non‑IID (low β) where client sizes
    differ by orders of magnitude. Biases toward small clients.
  - Effort: Easy. Add tests and publication configs/presets that use
    `"fedavg_weighted"` by default. Keep `"mean"` only for legacy
    reproduction.
  - Confidence: High.

- [x] **B2. Naive BatchNorm running‑stats averaging (server.py §1.3)**
  - Fixed: added `bn_fl_strategy: "average"` (legacy) or `"fedbn"` config.
    When `"fedbn"`, BN running-mean/var are skipped during aggregation;
    clients keep their own stats.
  - Impact: **High** for configs using `normalization: batch`
    (ogbn‑arxiv GCN_arxiv / GAT_Arxiv, GraphSAGEProducts).
  - Effort: Medium.
  - Confidence: High.

- [x] **B3. `num_batches_tracked` becomes stale after round 1
      (server.py §1.3)**
  - Fixed: non-float buffers are now always copied from the first client
    each round (tracked via `first_client` flag), not only when `mb.sum()==0`.
  - Impact: Low to Medium.
  - Effort: Trivial.
  - Confidence: Medium.

- [x] **B4. Initial broadcast not synchronized (server.py §1.4)**
  - Fixed: `broadcast_params(-1, sync=True)` in `Server.__init__`.
  - Impact: Low to Medium — can bias round 0 per‑client training when
    `max_concurrent_clients < num_clients`.
  - Effort: Trivial.
  - Confidence: Medium.

- [x] **B5. Silent inclusion of failed clients in aggregation
      (server.py §1.5)**
  - Fixed: `train_client` now returns `(loss, acc, success: bool)`. Failed
    clients are filtered before aggregation with a warning.
  - Impact: Medium when clients OOM intermittently (observed in
    ogbn‑arxiv + diffusion runs).
  - Effort: Easy — return a success flag and filter before averaging.
  - Confidence: High for runs with known OOM errors in logs; Low otherwise.

---

## C. Train / eval mismatch (what the clients learn ≠ what you measure)

This is the biggest subtle source of wrong numbers on FP + PE experiments.

- [x] **C1. Feature propagation applied per‑client, but
      `test_global_model` uses unprocessed `data.x` (run.py §3.3 /
      partitioning.py §4.5)**
  - Fixed: added `global_eval_uses_fp` config. When true, the global graph is
    feature-propagated before `test_global_model`; when false, a warning marks
    the legacy mismatch.
  - Impact: **High** for all FP modes (`adjacency`, `diffusion`,
    `chebyshev_*`, `random_walk`). Likely explains part of the gap
    between client‑average and global accuracy in your tables.
  - Effort: Medium.
  - Confidence: High.

- [x] **C2. Positional encodings computed locally per client AND again
      globally — different PE for the same node (positional_encoding.py
      §6.1, run.py §3.3)**
  - Fixed: added `global_eval_pe_mode`. The legacy `"local"` mode is explicit;
    `"global_compute"` recomputes seeded full-graph RFP for global eval and
    replaces the existing PE block instead of appending a second block.
  - Impact: **High** whenever `use_pe: true`. Clients train on a
    structure‑dependent feature that the global test set does not
    reproduce.
  - Effort: Medium.
  - Confidence: High.

- [x] **C3. `fulltraining_flag=True` causes cross‑client label leakage
      (partitioning.py §4.4)**
  - Fixed: `conf/base.yaml` now labels this as an oracle baseline and
    `partition_data()` prints a runtime warning when enabled.
  - Impact: Inflates accuracy. **High** if used as your "federated"
    condition; acceptable as an oracle baseline if labelled as such.
  - Effort: Medium.
  - Confidence: High.

- [x] **C4. Macro‑average of per‑client test accuracies reported as if
      it were global (run.py §3.2)**
  - Fixed: logs client-local macro accuracy and client-local micro accuracy
    weighted by each client's test-node count. Macro aggregation is now
    `nan`-aware.
  - Impact: Medium — numbers are not "wrong" but they are not what most
    FL papers mean by "test accuracy".
  - Effort: Easy.
  - Confidence: High.

- [x] **C5. Train/eval seed inconsistency masks true variance
      (train.py §2.5)**
  - Fixed: `experiment_seed` reaches partitioning, server model init,
    client training, per-client RFP, and global PE. Repetitions derive
    `experiment_seed + run_idx`, mini-batch eval/test no longer reseed to 42
    unless an explicit seed is provided, and per-repetition seed provenance is
    recorded in result JSON.
  - Note: `experiment_seed: null` remains the legacy default for backward
    compatibility; publication configs should set it explicitly.
  - Impact: **High on reported std**. Reported `std` is training‑noise
    only; partition variance hidden. Given how sensitive FL is to
    partitioning, the true std is likely much larger.
  - Effort: Medium.
  - Confidence: High.

---

## D. Training dynamics knobs that are silently hardcoded

These do not produce a wrong number, but they make fair comparisons
between configs/datasets very hard.

- [x] **D1. Hardcoded gradient clipping at `max_norm=1.0` (train.py §2.7)**
  - Fixed: added `grad_clip_norm` config and threaded it through full-batch and
    mini-batch training. `null`/non-positive values disable clipping.
  - Impact: Under SGD with lr=0.5 (Cora / Citeseer defaults) this
    suppresses gradient magnitude significantly and interacts strongly
    with normalization choices.
  - Effort: Trivial.
  - Confidence: Medium.

- [x] **D2. `patience_threshold=10` hardcoded (run.py)**
  - Fixed: now reads `early_stopping_patience` from config (default 10).
  - Impact: With `num_rounds=200` (Cora) this is fine; with
    `num_rounds=5` (ogbn‑arxiv) it is effectively disabled. Tied to A4.
  - Effort: Trivial. Expose `patience` in config.
  - Confidence: Medium.

- [x] **D3. FP convergence tolerance never triggers on large graphs
      (data_utils.py §5.5)**
  - Fixed: added `feature_prop_relative_tolerance`. When enabled, convergence
    uses `||out - prev_out|| / (||prev_out|| + eps)`.
  - Impact: No accuracy impact, but burns compute that distorts
    efficiency comparisons across propagation modes (see §E below).
  - Effort: Easy.
  - Confidence: High.

- [x] **D4. `diffusion_kernel` first‑order fallback for N>50k is not
      a diffusion (propagation_functions.py §5.4)**
  - Fixed: added `force_chebyshev_for_large_graphs` (default true) and redirect
    large-graph `diffusion`/`diffusion_kernel` requests to Chebyshev with a
    warning.
  - Impact: Medium on ogbn‑arxiv diffusion experiments — the "diffusion"
    numbers do not measure what the name says.
  - Effort: Easy.
  - Confidence: High.

---

## E. Orchestration issues that distort accuracy *comparisons*

These do not change any single run's number, but they make side‑by‑side
comparisons (sweeps across β, propagation mode, etc.) unreliable.

- [x] **E1. `use_pe` list/scalar inconsistency in
      `run_experiments.py` (§9.1)**
  - Fixed: added `use_pe` to the in-memory default config and normalized
    scalar/list sweep axes through a shared `as_list()` helper.
  - Effort: Trivial.
  - Confidence: High.

- [ ] **E2. Ray init + forced `time.sleep(1)` between experiments
      (run_experiments.py §10.2)**
  - Symptom: Long sweeps occasionally crash with CUDA context
    corruption after `diffusion` or `chebyshev_*` runs.
  - Impact: Lost experiment runs → asymmetric sample sizes per config →
    incorrect averages/std in summary tables.
  - Effort: Medium. Subprocess‑per‑experiment harness.
  - Confidence: Medium.

---

## F. Runtime / compute‑efficiency (not accuracy, but blocks ablations)

Listed for completeness — fix only if they block the experiments you
actually need.

- [ ] F1. `monte_carlo_random_walk` is O(V·walks·len) pure Python (§5.2).
- [ ] F2. `compute_dirichlet_energy` per FP iteration under logging (§5.6).
- [ ] F3. QR every RFP iteration on N=169k (§6.2).
- [x] F4. `GradScaler` instantiated even on CPU / `use_amp=false` (§2.6).
      Fixed: full-batch and mini-batch training instantiate/use `GradScaler`
      only when `use_amp` is true.
- [ ] F5. Hardcoded `num_gpus=1/10` per actor, fixed `num_gpus=1` in Ray
      init (§2.1, §3.4).
- [ ] F6. Reproducible environment is not pinned: `requirements.txt` leaves
      critical versions unconstrained and does not explicitly list
      `torch_sparse` / `torch_scatter`, while CLI smoke tests are skipped when
      heavy deps are absent.

---

## Recommended fix order

If time is limited, do these in order; each one independently improves
the accuracy numbers you can trust.

1. **A1, A2, A3** — stop crashes / obvious logit bugs. (<1 hour total)
2. **A4, A5** — fix early stopping and empty-mask metrics before more runs.
3. **C5 (seed plumbing)** — so every subsequent measurement reflects
   real variance. Trivial plumbing but unlocks meaningful error bars.
4. **B1 (FedAvg weighting)** — single most likely source of a clean
   accuracy jump at low β.
5. **B2 (BN strategy)** — biggest win for ogbn‑arxiv / products configs.
6. **C1, C2 (FP/PE consistency at eval time)** — realigns client training
   and global evaluation.
7. **B4, B5, C3, C4** — cleanups that tighten numbers.
8. **D, E, F** — infrastructure and reproducibility polish.

---

## How to use this checklist

- Open a PR per item (or per small cluster), tick the box, and link the
  PR next to the bullet.
- Every fix should follow the project convention: guarded by a config
  flag with a default that reproduces current behavior. Exception:
  obvious crash bugs (A1, A3) — just fix.
- When a fix lands, re‑run the affected experiments and record deltas
  in a follow‑up doc (`FL_FIX_RESULTS.md`) so the science stays auditable.
