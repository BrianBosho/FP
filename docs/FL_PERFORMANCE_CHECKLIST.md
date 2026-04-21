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

---

## A. Correctness bugs that silently skew accuracy (fix first)

These are bugs where a code path is either wrong or broken, and results
you have already produced may reflect the bug rather than the science.

- [ ] **A1. `GAT_Arxiv` missing from `evaluate`/`test` (train.py §2.3)**
  - Symptom: any run with `GAT_Arxiv` crashes on validation or test.
    If no crash observed, confirm which GAT class is actually used on
    `ogbn-arxiv` (config uses `use_unified_model: false` → `GAT_Arxiv`).
  - Impact: **High** — all reported ogbn‑arxiv GAT numbers must be
    re-validated.
  - Effort: Trivial. Add `GAT_Arxiv` to the isinstance tuples in
    `evaluate`, `test`, `evaluate_with_minibatch`, `test_with_minibatch`.
  - Confidence: High.

- [ ] **A2. `PubmedGAT` output layer produces `num_classes * 8` logits
      (models.py §7.1)**
  - Symptom: final layer uses `heads=8` with default `concat=True`.
    `log_softmax` spreads probability across 8× as many columns as classes.
  - Impact: **High** on Pubmed accuracy — probability mass bleeds into
    phantom classes.
  - Effort: Trivial. Either `heads=1`, or `heads=8, concat=False`.
  - Confidence: High.

- [ ] **A3. `page_rank` propagation mode references undefined
      `current_node` (propagation_functions.py §5.3)**
  - Symptom: any run with `data_loading: page_rank` crashes with
    `NameError`.
  - Impact: High if ever used; none if never used. Confirm against logs.
  - Effort: Trivial.
  - Confidence: High.

- [ ] **A4. Early‑stopping logic resets patience against stale bests
      (run.py §3.1)**
  - Symptom: `best_eval_loss` is only updated in the `elif`; `best_eval_acc`
    only in the `if`. Patience can reset spuriously, stretching training
    past the real best epoch, or stopping too early.
  - Impact: Medium on reported best accuracy, especially when training
    is long and noisy (ogbn‑arxiv, Pubmed GAT).
  - Effort: Trivial — pick loss as sole criterion, or track both cleanly.
  - Confidence: High.

- [ ] **A5. Val/test mask division by zero on highly non‑IID shards
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

- [ ] **B1. Unweighted averaging instead of FedAvg (server.py §1.1)**
  - Symptom: `p.data /= self.num_of_trainers` — ignores `n_k`.
  - Impact: **High** under Dirichlet non‑IID (low β) where client sizes
    differ by orders of magnitude. Biases toward small clients.
  - Effort: Easy. Add `aggregation: "mean" | "fedavg"` config flag; for
    `fedavg`, collect `n_k` once via a new remote method and weight the
    sum. Keep `"mean"` as default for backward compatibility.
  - Confidence: High.

- [ ] **B2. Naive BatchNorm running‑stats averaging (server.py §1.3)**
  - Symptom: float buffers (BN running mean/var) are summed across
    clients and divided by `K`. Under non‑IID this is known to hurt.
  - Impact: **High** for configs using `normalization: batch`
    (ogbn‑arxiv GCN_arxiv / GAT_Arxiv, GraphSAGEProducts).
  - Effort: Medium. Options, from cheapest to most principled:
    1. Switch FL models to `layer` / `group` norm (trivial, may lose
       accuracy on arxiv).
    2. Keep running stats per‑client, aggregate only `weight` + `bias`
       (FedBN).
    3. Apply SiloBN / HeteroBN.
  - Confidence: High.

- [ ] **B3. `num_batches_tracked` becomes stale after round 1
      (server.py §1.3)**
  - Symptom: `zero_buffers()` only zeroes float buffers; non‑float
    buffers are copied from the first client only when `mb.sum()==0`,
    which is only true at round 0.
  - Impact: Low to Medium. BN uses `num_batches_tracked` mainly for
    momentum; stale values slowly drift running stats but are not
    catastrophic.
  - Effort: Trivial — explicitly copy from the first contributing client
    each round.
  - Confidence: Medium.

- [ ] **B4. Initial broadcast not synchronized (server.py §1.4)**
  - Symptom: `broadcast_params(-1)` in `__init__` without `sync=True`.
    In batched mode this can race with the first training submission.
  - Impact: Low to Medium — can bias round 0 per‑client training when
    `max_concurrent_clients < num_clients`.
  - Effort: Trivial.
  - Confidence: Medium.

- [ ] **B5. Silent inclusion of failed clients in aggregation
      (server.py §1.5)**
  - Symptom: `train_client` returns `(0.0, 0.0)` on exception; `get_params`
    still returns the pre‑training broadcast, which is averaged in.
  - Impact: Medium when clients OOM intermittently (observed in
    ogbn‑arxiv + diffusion runs).
  - Effort: Easy — return a success flag and filter before averaging.
  - Confidence: High for runs with known OOM errors in logs; Low otherwise.

---

## C. Train / eval mismatch (what the clients learn ≠ what you measure)

This is the biggest subtle source of wrong numbers on FP + PE experiments.

- [ ] **C1. Feature propagation applied per‑client, but
      `test_global_model` uses unprocessed `data.x` (run.py §3.3 /
      partitioning.py §4.5)**
  - Symptom: clients train on propagated features; the global evaluator
    receives raw features. Global test acc systematically
    under‑represents what the model has learned.
  - Impact: **High** for all FP modes (`adjacency`, `diffusion`,
    `chebyshev_*`, `random_walk`). Likely explains part of the gap
    between client‑average and global accuracy in your tables.
  - Effort: Medium. Two options, both behind a flag
    `global_eval_uses_client_preprocessing`:
    1. Recompute FP on the global graph before `test_global_model`.
    2. Stop reporting `test_global_model` when FP is on, or mark it as
       "mismatched" in summaries.
  - Confidence: High.

- [ ] **C2. Positional encodings computed locally per client AND again
      globally — different PE for the same node (positional_encoding.py
      §6.1, run.py §3.3)**
  - Symptom: RFP is computed from the subgraph edge_index on clients
    and from the global edge_index for `data` on the server. Same node,
    different PE.
  - Impact: **High** whenever `use_pe: true`. Clients train on a
    structure‑dependent feature that the global test set does not
    reproduce.
  - Effort: Medium. Options:
    1. Compute RFP once globally, extract per‑client rows (leaks
       structural info — acceptable if the threat model allows it).
    2. Keep local PEs, report only client‑local test metrics.
    3. Document the mismatch explicitly in the paper.
  - Confidence: High.

- [ ] **C3. `fulltraining_flag=True` causes cross‑client label leakage
      (partitioning.py §4.4)**
  - Symptom: k‑hop expanded subgraph keeps all masks from the global
    graph, including masks of nodes owned by other clients. A client
    optimizes over labels it does not own.
  - Impact: Inflates accuracy. **High** if used as your "federated"
    condition; acceptable as an oracle baseline if labelled as such.
  - Effort: Medium. Rename the baseline to `oracle_khop`, disallow it
    under strict FL, or restrict masks to the owning client.
  - Confidence: High.

- [ ] **C4. Macro‑average of per‑client test accuracies reported as if
      it were global (run.py §3.2)**
  - Symptom: `average_results = sum(client_test_results)/len(...)` — a
    macro average biased when client test sizes are imbalanced.
  - Impact: Medium — numbers are not "wrong" but they are not what most
    FL papers mean by "test accuracy".
  - Effort: Easy. Also log `global_test_acc_micro` = weighted by client
    test size.
  - Confidence: High.

- [ ] **C5. Train/eval seed inconsistency masks true variance
      (train.py §2.5)**
  - Symptom: `label_dirichlet_partition` seeds `np.random.seed(123)`,
    every repetition gets the **same partition**. Mini‑batch paths seed
    torch/np/python to 42 on every entry. Full‑batch path never seeds.
    RFP unseeded.
  - Impact: **High on reported std**. Reported `std` is training‑noise
    only; partition variance hidden. Given how sensitive FL is to
    partitioning, the true std is likely much larger.
  - Effort: Medium. Single `experiment_seed`, derive per‑repetition
    `seed + run_idx`, thread into partitioning, RFP, model init, DataLoaders.
    Keep defaults that reproduce current behavior.
  - Confidence: High.

---

## D. Training dynamics knobs that are silently hardcoded

These do not produce a wrong number, but they make fair comparisons
between configs/datasets very hard.

- [ ] **D1. Hardcoded gradient clipping at `max_norm=1.0` (train.py §2.7)**
  - Impact: Under SGD with lr=0.5 (Cora / Citeseer defaults) this
    suppresses gradient magnitude significantly and interacts strongly
    with normalization choices.
  - Effort: Trivial. Expose `grad_clip_norm` in `base.yaml`, default 1.0.
  - Confidence: Medium.

- [ ] **D2. `patience_threshold=10` hardcoded (run.py)**
  - Impact: With `num_rounds=200` (Cora) this is fine; with
    `num_rounds=5` (ogbn‑arxiv) it is effectively disabled. Tied to A4.
  - Effort: Trivial. Expose `patience` in config.
  - Confidence: Medium.

- [ ] **D3. FP convergence tolerance never triggers on large graphs
      (data_utils.py §5.5)**
  - Symptom: `delta = torch.norm(out - prev_out)` on ogbn‑arxiv never
    falls below `tol=1e-6`; FP always runs `num_iterations` even after
    converging numerically per‑node.
  - Impact: No accuracy impact, but burns compute that distorts
    efficiency comparisons across propagation modes (see §E below).
  - Effort: Easy. Normalize the delta: `delta / ||prev_out|| + eps`, or
    use L∞ per‑node.
  - Confidence: High.

- [ ] **D4. `diffusion_kernel` first‑order fallback for N>50k is not
      a diffusion (propagation_functions.py §5.4)**
  - Symptom: `I - tL` for `t≥1` is not PSD — the "diffusion" mode on
    large graphs is actually a single‑step Jacobi smoother.
  - Impact: Medium on ogbn‑arxiv diffusion experiments — the "diffusion"
    numbers do not measure what the name says.
  - Effort: Easy. Force Chebyshev path for N>50k and warn if `t≥1`.
  - Confidence: High.

---

## E. Orchestration issues that distort accuracy *comparisons*

These do not change any single run's number, but they make side‑by‑side
comparisons (sweeps across β, propagation mode, etc.) unreliable.

- [ ] **E1. `use_pe` list/scalar inconsistency in
      `run_experiments.py` (§9.1)**
  - Symptom: scalar `use_pe: true` crashes the sweep loop; silent
    regression when users edit dataset configs.
  - Effort: Trivial. Wrap the same way as `num_clients`, `beta`, etc.
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
- [ ] F4. `GradScaler` instantiated even on CPU / `use_amp=false` (§2.6).
- [ ] F5. Hardcoded `num_gpus=1/10` per actor, fixed `num_gpus=1` in Ray
      init (§2.1, §3.4).

---

## Recommended fix order

If time is limited, do these in order; each one independently improves
the accuracy numbers you can trust.

1. **A1, A2, A3** — stop crashes / obvious logit bugs. (<1 hour total)
2. **C5 (seed plumbing)** — so every subsequent measurement reflects
   real variance. Trivial plumbing but unlocks meaningful error bars.
3. **B1 (FedAvg weighting)** — single most likely source of a clean
   accuracy jump at low β.
4. **B2 (BN strategy)** — biggest win for ogbn‑arxiv / products configs.
5. **C1, C2 (FP/PE consistency at eval time)** — realigns client training
   and global evaluation.
6. **A4, A5, B4, B5, C3, C4** — cleanups that tighten numbers.
7. **D, E, F** — polish.

---

## How to use this checklist

- Open a PR per item (or per small cluster), tick the box, and link the
  PR next to the bullet.
- Every fix should follow the project convention: guarded by a config
  flag with a default that reproduces current behavior. Exception:
  obvious crash bugs (A1, A3) — just fix.
- When a fix lands, re‑run the affected experiments and record deltas
  in a follow‑up doc (`FL_FIX_RESULTS.md`) so the science stays auditable.
