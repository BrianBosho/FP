# Federated-GNN — Fix Implementation Plan

**Date:** 2026-04-21
**Companion to:** [`FL_PERFORMANCE_CHECKLIST.md`](./FL_PERFORMANCE_CHECKLIST.md),
[`FL_IMPLEMENTATION_REVIEW.md`](./FL_IMPLEMENTATION_REVIEW.md)

This document lays out the concrete implementation steps for every open checklist
item, grouped into 6 independent phases that can be landed and tested separately.

**Convention:** Every behavior change is guarded by a config flag whose default
reproduces current behavior. Exception: obvious crash bugs (A1, A3) are just fixed
in place.

**Research-production note:** legacy-compatible defaults are useful for
reproduction, but they should not be the defaults for new publication runs. A
separate publication preset should enable the scientifically correct behavior
once the fixes below land: weighted FedAvg, deterministic non-null seeds, fixed
PubmedGAT output heads, safe empty-mask metrics, explicit evaluation mode, and a
BatchNorm FL strategy.

---

## Current State Summary

Several items from the original review have been **partially addressed** in prior
work but remain unchecked:

| Item | Partial work done | What's left |
|------|-------------------|-------------|
| B1 (FedAvg) | `server.py` has `_aggregate_fedavg_weighted` and clients expose `get_num_train_samples()` | Default is still `"mean"`; publication configs should use `"fedavg_weighted"` |
| B4 (sync broadcast) | `train_clients` uses conditional sync | Initial `__init__` broadcast still unsynced |
| C5 (seeding) | `experiment_seed` reaches partitioning, server model init, client training, and per-client RFP | Defaults to `null`; repetitions do not derive distinct seeds; global PE and minibatch eval still have seed gaps |
| D2 (patience) | `training.default.patience: 10` in config | `run.py` hardcodes `patience_threshold = 10` |

The checklist now distinguishes `[~]` partially implemented items from `[ ]`
open items. Treat `[~]` items as not publication-ready until their remaining
work is done and tested.

## Current Review Findings (2026-04-21)

The implementation still has several blockers before results should be used in
publication tables:

1. `GAT_Arxiv` is still missing from `evaluate()`, `test()`,
   `evaluate_with_minibatch()`, and `test_with_minibatch()`.
2. `PubmedGAT` still emits `num_classes * 8` logits because the output
   `GATv2Conv` uses `heads=8` with default `concat=True`.
3. Full-batch val/test accuracy still divides by zero when a client has no
   val/test nodes.
4. Early stopping still updates best loss and best accuracy through stale
   `if`/`elif` logic.
5. Failed clients still return `(0.0, 0.0)` and are included in aggregation.
6. BatchNorm running stats are still averaged, and non-float BN buffers can go
   stale after round 1.
7. Global evaluation remains mismatched with client-side feature propagation
   and positional encoding.
8. Sweep handling still treats scalar `use_pe` incorrectly and the in-memory
   default config omits `use_pe`.
9. The `src/fedgnn` package currently contains compatibility wrappers over
   legacy `src.*` modules, not migrated implementations.

---

## Phase 1 — Crash bugs and logit bugs

> **Effort:** < 1 hour
> **Impact:** Fixes crashes and wrong logit dimensions

### 1.1 A1 — Add `GAT_Arxiv` to evaluate / test isinstance checks

**File:** `src/train.py`
**Lines:** 110, 130, ~314, ~396

Four functions contain an `elif isinstance(model, ...)` chain that dispatches on
model type. `GAT_Arxiv` is present in `train()` but missing from `evaluate()`,
`test()`, `evaluate_with_minibatch()`, and `test_with_minibatch()`. Any run with
`GAT_Arxiv` crashes with `"Unknown model"` on validation.

**Fix:** Add `isinstance(model, GAT_Arxiv)` to each `elif` chain. `GAT_Arxiv`
uses the same forward signature as `GCN_arxiv`: `model(data.x, data.edge_index)`.

```python
# Before (evaluate, line 110):
elif isinstance(model, GCN) or isinstance(model, GAT) or isinstance(model, GCN_arxiv) or isinstance(model, GraphSAGEProducts) or isinstance(model, PubmedGAT):

# After:
elif isinstance(model, (GCN, GAT, GCN_arxiv, GAT_Arxiv, GraphSAGEProducts, PubmedGAT)):
```

Apply the same change in `test()`, `evaluate_with_minibatch()`, and
`test_with_minibatch()`.

---

### 1.2 A2 — Fix PubmedGAT output layer producing `num_classes * 8` logits

**File:** `src/models.py`, line 464
**Config guard:** `pubmedgat_fix_heads: false` (default = legacy)

`PubmedGAT.__init__` builds the output layer as:

```python
self.convs.append(GATv2Conv(self.dim_h * heads, dim_out, heads=8))
```

`GATv2Conv` defaults to `concat=True`, so the output shape is
`(N, dim_out * 8)`. `log_softmax` then spreads probability across 8× too many
columns.

**Fix:**

```python
# In PubmedGAT.__init__, read from config:
fix_heads = self.cfg.get("pubmedgat_fix_heads", False) if hasattr(self, 'cfg') and self.cfg else False
output_concat = not fix_heads  # legacy=True keeps concat=True

self.convs.append(GATv2Conv(
    self.dim_h * heads, dim_out, heads=8, concat=output_concat
))
```

When `pubmedgat_fix_heads: true`, output is `(N, dim_out)` as intended.

---

### 1.3 A3 — Verify and scale `page_rank` propagation mode

**File:** `src/dataprocessing/propagation_functions.py`, lines 8–62

The review described a `current_node` `NameError` in the page-rank loop. The
current implementation at line 42 uses `node` (the loop variable) correctly:

```python
for node in range(num_nodes):
    neighbors = col[row == node]
```

**Action:** The old `NameError` appears fixed, but this path still builds a
dense `num_nodes x num_nodes` PPR matrix and loops over nodes. Add a small
correctness test, then either restrict `page_rank` to small graphs or replace it
with a sparse power-iteration implementation before using it in publication
runs.

---

## Phase 2 — Early stopping and mask guards

> **Effort:** ~ 1 hour
> **Impact:** Correct convergence detection; no more ZeroDivisionError

### 2.1 A4 — Fix early-stopping logic

**File:** `src/run.py`, lines 370–379

**Problem:** `best_eval_loss` is only updated in the `elif` branch, so it goes
stale once accuracy starts improving. Patience resets against a stale loss value.

**Current code:**

```python
improved_acc = avg_eval_acc > best_eval_acc
improved_loss = avg_eval_loss < best_eval_loss
if improved_acc:
    best_eval_acc = avg_eval_acc
    patience = 0
elif improved_loss:
    best_eval_loss = avg_eval_loss
    patience = 0
else:
    patience += 1
```

**Fix:** Always update both bests and reset patience when either improves:

```python
improved = False
if avg_eval_acc > best_eval_acc:
    best_eval_acc = avg_eval_acc
    improved = True
if avg_eval_loss < best_eval_loss:
    best_eval_loss = avg_eval_loss
    improved = True
patience = 0 if improved else patience + 1
```

---

### 2.2 A5 — Guard val/test mask division by zero

**File:** `src/train.py`, lines 82, 122, 140 (+ minibatch equivalents)

When a client subgraph has zero val/test nodes, `mask.sum()` is 0 and the
division crashes.

**Fix:** Wrap every accuracy computation and make missing metrics explicit:

```python
total = int(mask.sum())
acc = int(correct) / total if total > 0 else float('nan')
```

Apply in: `train()`, `evaluate()`, `test()`, `train_with_minibatch()`,
`evaluate_with_minibatch()`, `test_with_minibatch()`.

Downstream macro averages must use `np.nanmean` and log how many clients
contributed non-NaN metrics. Returning `0.0` for an empty validation/test mask
mixes "missing" with a real zero-accuracy result.

---

### 2.3 D2 — Expose `patience_threshold` from config

**File:** `src/run.py`, line 350

Replace `patience_threshold = 10` with:

```python
patience_threshold = cfg.get("early_stopping_patience", 10)
```

The value already exists in `conf/base.yaml` as `training.default.patience: 10`.

---

## Phase 3 — Aggregation correctness

> **Effort:** ~ 2 hours
> **Impact:** Correct FL convergence, especially under non-IID data

### 3.1 B1 — FedAvg weighting (verify existing implementation)

**Files:** `src/server.py`, `conf/base.yaml`

Already implemented: `_aggregate_fedavg_weighted` method, `get_num_train_samples`
on client, `aggregation: "mean"` config (opt-in to `"fedavg_weighted"`).

**Action:** Run a verification test and add a publication config/preset that
sets `aggregation: "fedavg_weighted"`. Keep `"mean"` only for legacy
reproduction. Do not treat unweighted mean as the default for new FL
publication experiments.

---

### 3.2 B2 — BatchNorm running-stats aggregation strategy

**Files:** `src/server.py`, `conf/base.yaml`
**Config:** `bn_fl_strategy: "average"` (default = legacy) or `"fedbn"`

Under non-IID data, averaging BN running-mean/var across clients degrades accuracy.
FedBN keeps per-client running stats and only aggregates `weight`/`bias`.

**Fix:**

In `_aggregate_mean` and `_aggregate_fedavg_weighted`, check if a buffer belongs
to a BatchNorm layer. When `bn_fl_strategy: "fedbn"`, skip float buffers that are
BN running-mean or running-var:

```python
bn_strategy = self.cfg.get("bn_fl_strategy", "average")

# In the buffer aggregation loop:
for name, b, mb in zip(buffer_names, params_dict['buffers'], self.model.named_buffers()):
    if "running_mean" in name or "running_var" in name:
        if bn_strategy == "fedbn":
            continue  # skip — clients keep their own stats
    # ... existing aggregation logic
```

Note: need to collect buffer names alongside values in `get_params()`.

---

### 3.3 B3 — Fix stale `num_batches_tracked` after round 1

**File:** `src/server.py`, lines 101–107 and 175–180

`zero_buffers()` only zeros float buffers. The `mb.sum() == 0` check for
non-float buffers is only true at round 0, so `num_batches_tracked` is never
updated after the first round.

**Fix:** Track whether we've seen the first client, and always copy non-float
buffers from client 0:

```python
first_client = True
# ... inside the loop:
if not b.dtype.is_floating_point:
    if first_client:
        mb.data = b.to(self.device)
# ... after loop iteration:
first_client = False
```

---

### 3.4 B4 — Sync initial broadcast

**File:** `src/server.py`, line 57

```python
# Before:
self.broadcast_params(-1)

# After:
self.broadcast_params(-1, sync=True)
```

---

### 3.5 B5 — Filter failed clients from aggregation

**Files:** `src/client.py`, `src/server.py`

**client.py** — change return signature:

```python
# train_client: return 3-tuple
return loss, acc, True      # success
return 0.0, 0.0, False      # exception
```

**server.py** — after collecting results, filter:

```python
results = ray.get(train_futures)
successful = [(client, res) for client, res in zip(clients, results) if res[2]]
if len(successful) < len(clients):
    print(f"[Server] {len(clients) - len(successful)} clients failed; "
          f"aggregating {len(successful)}/{len(clients)}")
clients_for_agg = [c for c, _ in successful]
self._aggregate_mean(clients_for_agg)  # or fedavg_weighted
```

Adjust all downstream unpacking of `(loss, acc)` to handle the third element.

---

## Phase 4 — Train / eval consistency

> **Effort:** ~ 3 hours
> **Impact:** Aligns what the model learns with what is measured

### 4.1 C1 — Global test uses unprocessed features when FP is active

**File:** `src/run.py`, around line 440
**Config:** `global_eval_uses_fp: false` (default = legacy)

Clients train on propagated features; `test_global_model(data)` feeds raw
features to the server model.

**Fix when `global_eval_uses_fp: true`:** Before calling
`test_global_model(data)`, apply the same feature propagation to the global
graph:

```python
if cfg.get("global_eval_uses_fp", False) and data_loading_option != "none":
    data.x = propagate_features(
        data.x, data.edge_index,
        mask=torch.ones(data.x.size(0), dtype=torch.bool),
        device=self.device,
        num_iterations=cfg.get("num_iterations", 80),
        mode=data_loading_option,
        alpha=cfg.get("alpha", 0.5),
    )
```

When `false` (default): current behavior, but log a one-time warning.

---

### 4.2 C2 — PE per-client vs global inconsistency

**Files:** `src/run.py`, `src/dataprocessing/positional_encoding.py`
**Config:** `global_eval_pe_mode: "local"` (default) or `"global_compute"`

- `"local"`: current behavior. Document the mismatch.
- `"global_compute"`: compute RFP once on the full graph, then extract per-client
  rows. This aligns train/eval but leaks structural info across clients.

When `"global_compute"` and `use_pe: true`, recompute RFP on the global graph
before `test_global_model`.

---

### 4.3 C3 — Document `fulltraining_flag` as oracle baseline

**File:** `conf/base.yaml`, `src/dataprocessing/partitioning.py`

Already defaults to `false`. Add:

```yaml
fulltraining_flag: false  # WARNING: true = oracle baseline with cross-client label leakage
```

Add a runtime `print()` warning in `partitioning.py` when the flag is `true`.

---

### 4.4 C4 — Report micro-averaged test accuracy alongside macro

**File:** `src/run.py`, around lines 446–450

Current code computes a macro-average over per-client test accuracies. Add a
micro-average weighted by test set size:

```python
# After collecting client_test_results and test_data:
total_correct = 0
total_test_nodes = 0
for acc, td in zip(client_test_results, test_data):
    n = int(td.test_mask.sum())
    if n > 0:
        total_test_nodes += n
        total_correct += acc * n

global_test_acc_micro = total_correct / total_test_nodes if total_test_nodes > 0 else float('nan')

# Log both:
print(f"Local test acc (macro): {sum(client_test_results)/len(client_test_results):.4f}")
print(f"Global test acc (micro): {global_test_acc_micro:.4f}")
```

---

### 4.5 C5 — Complete seed plumbing

**Files:** `src/run.py`, `src/dataprocessing/partitioning.py`, `src/dataprocessing/positional_encoding.py`

Partial work exists. Verify the full chain is wired:

1. `experiment_seed` in config → threaded to `label_dirichlet_partition(seed=...)`
2. Same seed → threaded to `generate_rfp_encoding(seed=...)`
3. Client derives `client_seed = experiment_seed + client_id` → passed to `train(seed=...)`

Current status: partially wired, not done. The seed reaches partitioning, server
model init, client training, and per-client RFP. Remaining gaps:

- `experiment_seed` still defaults to `null`.
- `main_experiment()` does not derive `experiment_seed + repetition_index`, so
  multiple repetitions can reuse the same partition/init when one base seed is
  supplied.
- `evaluate_with_minibatch()` and `test_with_minibatch()` still call
  `set_seed(42)`, mutating global RNG during evaluation.
- Global PE in `loaders.py` is not seeded.
- Seed values are not saved explicitly in the result JSON.

**Action:** Fix the remaining gaps, then mark C5 as done.

---

## Phase 5 — Config-driven training knobs

> **Effort:** ~ 1 hour
> **Impact:** Enables fair hyperparameter comparison across configs

### 5.1 D1 — Expose gradient clipping in config

**File:** `src/train.py`, lines 77 and 258
**Config:** `grad_clip_norm: 1.0` (default = current value)

```python
# In train() and train_with_minibatch(), accept grad_clip_norm parameter:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
```

Thread the value from `cfg.get("grad_clip_norm", 1.0)` through the call chain.

---

### 5.2 D3 — Normalize FP convergence delta

**File:** `src/dataprocessing/data_utils.py`, line 256
**Config:** `feature_prop_relative_tolerance: false` (default = legacy)

On ogbn-arxiv, the absolute L2 delta never drops below `tol=1e-6`, so FP always
runs the full iteration count.

**Fix when `feature_prop_relative_tolerance: true`:**

```python
delta = torch.norm(out - prev_out).item() / (torch.norm(prev_out).item() + 1e-12)
```

---

### 5.3 D4 — Force Chebyshev for large-graph diffusion

**File:** `src/dataprocessing/propagation_functions.py`, lines 212–217
**Config:** `force_chebyshev_for_large_graphs: true` (default)

When `num_nodes > 50000` and mode is `"diffusion"`, the first-order fallback
`I - tL` is not a valid diffusion kernel (non-PSD for `t >= 1`).

**Fix:** When the flag is `true`, redirect to `"chebyshev_diffusion"` with a
warning. When `false`, keep the legacy approximation.

---

## Phase 6 — Experiment infrastructure

> **Effort:** ~ 2 hours
> **Impact:** Reliable sweeps, correct experiment comparisons

### 6.1 E1 — Handle scalar `use_pe` in sweep loop

**File:** `src/experiments/run_experiments.py`

```python
# Before:
use_pe_values = OmegaConf.to_container(cfg["use_pe"], resolve=True)

# After:
use_pe_values = OmegaConf.to_container(cfg["use_pe"], resolve=True)
if not isinstance(use_pe_values, list):
    use_pe_values = [use_pe_values]
```

Also add `use_pe` to the in-memory default config in `run_experiments.py`; today
running without a YAML config can hit a missing-key path.

---

### 6.2 E2 — Ray init cleanup (deferred)

The Ray re-init + `time.sleep(1)` pattern is fragile. Full fix requires a
subprocess-per-experiment harness — out of scope for this round. Document the
issue and increase sleep to 2s as a mitigation.

---

### 6.3 F4 — Guard GradScaler behind `use_amp`

**File:** `src/train.py`, lines 44 and 177

```python
# Before:
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# After:
if use_amp:
    scaler = torch.cuda.amp.GradScaler()
    # ... mixed-precision backward
else:
    loss.backward()
    optimizer.step()
```

---

### 6.4 Config comment fix

**File:** `conf/base.yaml`, line 67

```yaml
# Before:
feature_prop_device: "cuda"  # Memory optimization: Use CPU for preprocessing to save ~9GB GPU memory

# After (align comment with value):
feature_prop_device: "cuda"  # Set to "cpu" to save ~9GB GPU memory during preprocessing
```

---

## Verification Checklist

After each phase, run:

```bash
# Smoke test with Cora (fast, small graph):
python -m src.run --config-name=base dataset=cora num_rounds=3 num_clients=2

# Test GAT_Arxiv doesn't crash (Phase 1):
python -m src.run --config-name=base dataset=arxiv model_type=GAT_Arxiv num_rounds=1

# Test FedAvg weighting differs from mean (Phase 3):
python -m src.run --config-name=base dataset=cora aggregation=fedavg_weighted beta=0.1
```

After all phases:

- [ ] All 23 checklist items ticked `[x]`
- [ ] No new crashes on Cora, Citeseer, Pubmed, ogbn-arxiv configs
- [ ] `aggregation: "fedavg_weighted"` produces different accuracy than `"mean"`
  at `beta=0.1`
- [ ] Early stopping patience increments monotonically and resets correctly
- [ ] Global test accuracy reported when FP is active (Phase 4)
- [ ] Update `FL_PERFORMANCE_CHECKLIST.md` with `[x]` and commit SHAs

---

## Effort Summary

| Phase | Items | Est. effort | Priority |
|-------|-------|-------------|----------|
| 1 — Crash/logit bugs | A1, A2, A3 | < 1 h | **Critical** |
| 2 — Early stopping + masks | A4, A5, D2 | ~ 1 h | High |
| 3 — Aggregation | B1–B5 | ~ 2 h | High |
| 4 — Train/eval consistency | C1–C5 | ~ 3 h | High |
| 5 — Training knobs | D1, D3, D4 | ~ 1 h | Medium |
| 6 — Infrastructure | E1, E2, F4 | ~ 2 h | Low |
| **Total** | **19 distinct fixes** | **~ 10 h** | |
