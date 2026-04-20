# Federated‑GNN Implementation Review

**Scope:** `/home/brian_bosho/FP/FP/federated-gnn/src` (plus `conf/`)
**Files reviewed:** `run.py`, `server.py`, `client.py`, `train.py`, `models.py`,
`dataprocessing/partitioning.py`, `dataprocessing/loaders.py`,
`dataprocessing/data_utils.py`, `dataprocessing/propagation_functions.py`,
`dataprocessing/positional_encoding.py`, `dataprocessing/datasets.py`,
`utils/run_utils.py`, `utils/utils.py`, `experiments/run_experiments.py`,
`conf/base.yaml` and dataset overrides.

This document is a rigorous, section‑by‑section review ordered by severity.
Concrete fixes are suggested at the end; they are designed to be
backward‑compatible per project convention.

---

## 1. Federated Aggregation (server.py) — critical issues

### 1.1 Non‑standard FedAvg (unweighted averaging)

`server.Server.train_clients` aggregates with `mp.data += p` and then
`p.data /= self.num_of_trainers`. This is **simple averaging**, not FedAvg.
Canonical FedAvg weights each client by its local training set size:

\[
\theta^{t+1} \;=\; \sum_{k=1}^{K} \frac{n_k}{\sum_j n_j}\,\theta_k^{t+1}
\]

With Dirichlet non‑IID partitioning where client training counts differ by
orders of magnitude (especially for low β), unweighted averaging is biased
and can significantly degrade convergence.

```90:127:src/server.py
        params = [client.get_params.remote() for client in clients]
        self.zero_params()
        self.zero_buffers()

        while True:
            ready, left = ray.wait(params, num_returns=1, timeout=None)
            if ready:
                for t in ready:
                    params_dict = ray.get(t)
                    # Aggregate parameters
                    for p, mp in zip(params_dict['params'], self.model.parameters()):
                        mp.data += p.to(self.device)
                    ...
        # Average parameters
        for p in self.model.parameters():
             p.data /= self.num_of_trainers
```

**Fix:** expose each client's `n_k = train_mask.sum()` via a new remote
method, then do a weighted sum and divide by total `n`. Keep an
`aggregation: "mean" | "fedavg"` toggle to preserve backward compatibility.

### 1.2 `@torch.no_grad()` on the orchestration function

`train_clients` is decorated with `@torch.no_grad()` even though it launches
`client.train_client.remote()` (which does need autograd). Ray worker
processes are separate, so the decorator does not reach them — but it is
confusing and fragile. It also silently protects the aggregation block,
making the `mp.data +=` pattern redundant.

**Fix:** remove the decorator and put `torch.no_grad()` explicitly around
the aggregation block.

### 1.3 BatchNorm running stats aggregation is naive

```105:128:src/server.py
                    for b, mb in zip(params_dict['buffers'], self.model.buffers()):
                        if b.dtype.is_floating_point:
                            mb.data += b.to(self.device)
                        else:
                            # For non-float buffers (like num_batches_tracked), just copy from first client
                            if self.num_of_trainers == 1 or mb.sum() == 0:
                                mb.data = b.to(self.device)
```

1. **Running mean/variance averaging is not statistically correct**. This is
   the exact problem addressed by `FedBN` / `SiloBN` / `HeteroFedBN`. Under
   non‑IID data, averaging running stats cross‑client hurts accuracy. For
   the arxiv/products configs with `normalization: batch`, this matters a lot.
   Recommend either (a) switching to `layer` / `group` norm for FL
   (trivial), or (b) keeping running stats per‑client and averaging only
   the affine `weight`/`bias`.
2. **`num_batches_tracked` becomes stale after round 1:** `zero_buffers`
   only zeros *float* buffers, so `mb.sum() == 0` is true only at round 0.
   From round 1 onwards, no non‑float buffer is ever updated.

### 1.4 Initial broadcast is not synchronized

In `Server.__init__`: `self.broadcast_params(-1)` is called without
`sync=True`. In parallel mode Ray dependencies happen to serialize this
correctly because `train_client.remote()` is submitted after, but in
batched mode there is no such guarantee unless the first training round
also syncs.

**Fix:** always `sync=True` for the initial broadcast.

### 1.5 Aggregation not idempotent w.r.t. client failures

If one client raises, `FLClient.train_client` returns `0.0, 0.0` and
`get_params` returns the still‑initialized params (i.e. the pre‑training
broadcast). Those are then averaged in as if that client trained. There
is no "drop failed client" path — this silently degrades aggregation.

**Fix:** return a success flag and exclude failed clients from the average.

---

## 2. Client & Training (client.py, train.py)

### 2.1 Hardcoded `1/10` GPU fraction per actor

```24:25:src/client.py
@ray.remote(num_gpus=1/10)
class FLClient:
```

This is a Ray scheduling hint only (does not enforce memory). Hardcoding
`1/10` breaks when `num_clients != 10` and conflicts with
`max_concurrent_clients`.

**Fix:** make it a config field or compute as `1 / max_concurrent_clients`.

### 2.2 `train_with_minibatch` vs `train` return different metrics

- `train` returns `(final_val_loss, training_acc, ...)` where
  `training_acc` is the **last** epoch's training accuracy (variable
  overwritten each epoch).
- `train_with_minibatch` returns `(final_val_loss, training_accuracies[-1], ...)`.

`FLClient.train_client` does `loss, acc, loss_list, acc_list = train(...)`
and logs `acc` as "training accuracy". The two code paths disagree on
what `acc` means in edge cases, and `loss` carries a final validation
loss under a misleading name.

### 2.3 `evaluate` / `test` missing `GAT_Arxiv` in isinstance

```104:109:src/train.py
        elif isinstance(model, GCN) or isinstance(model, GAT) or isinstance(model, GCN_arxiv) or isinstance(model, GraphSAGEProducts) or isinstance(model, PubmedGAT):
            output = model(data.x, data.edge_index)
```

`GAT_Arxiv` is supported in `train()` (line 58) but missing from
`evaluate()`, `test()`, `evaluate_with_minibatch`, and
`test_with_minibatch`. **Any GAT_Arxiv run will crash with "Unknown
model" on validation/test.**

### 2.4 Val/test mask division by zero

```113:117:src/train.py
        loss = criterion(out[data.val_mask], data.y[data.val_mask])
        _, pred = torch.max(out[data.val_mask], dim=1)
        correct = (pred == data.y[data.val_mask]).sum()
        acc = int(correct) / int(data.val_mask.sum())
```

With strong Dirichlet sharding (e.g. β=0.1 and large K), a client subgraph
may have 0 val (or test) nodes → `ZeroDivisionError`.

**Fix:** guard with a safe default (e.g. `0.0` or NaN) when the mask sum is 0.

### 2.5 Inconsistent seeding

- `train_with_minibatch`, `evaluate_with_minibatch`, `test_with_minibatch`
  each call `set_seed(42)` at entry, mutating global torch/numpy/python
  RNG. Evaluating a model therefore *resets training RNG*, causing
  non‑determinism in unexpected places.
- `train()` (full‑batch) never seeds.
- `label_dirichlet_partition` uses a hardcoded `np.random.seed(123)`, so
  partitioning is identical across all `repetitions`.
- Random feature matrix in `generate_rfp_encoding` is not seeded.

Net effect: repetitions are partially deterministic and the reported `std`
underestimates real variance.

**Fix:** single `experiment_seed` in config, derived per‑repetition as
`seed + run_idx`, threaded into partitioning, DataLoader workers, model
init, and RFP.

### 2.6 GradScaler used unconditionally

`scaler = torch.cuda.amp.GradScaler(enabled=use_amp)` is instantiated even
with CPU or `use_amp=False`. With `enabled=False` it is a no‑op but still
adds API quirks.

**Fix:** wrap the backward block in an `if self.use_amp:` branch.

### 2.7 Hardcoded gradient clipping at `max_norm=1.0`

For GAT/PubmedGAT with learning rates up to 0.5 and SGD, clip=1.0 can
suppress learning.

**Fix:** expose as config (`grad_clip_norm`).

---

## 3. Orchestration / Run flow (run.py)

### 3.1 Early‑stopping logic is broken

```326:335:src/run.py
            if avg_eval_acc > best_eval_acc:
                best_eval_acc = avg_eval_acc
                patience = 0
            elif avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                patience = 0
            else:
                patience += 1
```

- `best_eval_loss` is only updated in the `elif`, so it becomes stale
  once accuracy starts improving (common case).
- `best_eval_acc` is only updated in the `if`. Entering the `elif` with a
  *worse* accuracy but better loss and then bumping accuracy next round
  resets patience against stale `best_eval_acc`.

**Fix:** track each metric independently, or pick one (loss) as the
definitive criterion.

### 3.2 `test_data = clients_data` then per‑client test accuracy averaging

```287:289:src/run.py
    test_data = clients_data
```

`client.test(test_data[i])` evaluates each client on its own subgraph's
`test_mask`. Because Dirichlet partitioning is disjoint over nodes, the
union of client test sets equals the global test set, but:

- The **macro‑average** over clients is *not* the same as the **global
  accuracy** (micro‑average). The code reports the macro‑average
  (`average_results = sum(client_test_results) / len(client_test_results)`)
  as if it were a validation of client models. With very imbalanced test
  counts per client (common at β < 1), macro‑avg is biased.
- A client's subgraph does not contain neighbors of test nodes owned by
  other clients, so GNN predictions on boundary nodes use a strictly
  smaller receptive field than global inference.

**Fix:** rename to make the distinction explicit
(`local_test_acc_macro` vs `global_test_acc`) and add a
`global_test_acc_from_clients_micro` weighted by test nodes per client.

### 3.3 "Full graph to CPU" hides a subtle correctness trap

```270:276:src/run.py
    data = data.to(torch.device("cpu"))
```

When `use_pe=True` in `load_and_split_with_khop`, the global `data.x`
gets RFP encoding appended on GPU, then moved to CPU here. The RFP on the
global graph is computed from the *global* edge_index whereas each
client's RFP is computed from its *subgraph* edge_index. **The same node
has a different PE in the client subgraph and in the global graph.** When
the server model is tested on `data` (global), it sees a different input
distribution than any client was trained on.

Options:

- Compute RFP once on the global graph, then extract rows per client
  (leaks structure across clients).
- Accept local PEs and do not report `test_global_model`.
- Document the mismatch explicitly in the paper / code.

### 3.4 Ray init with fixed `num_gpus=1`

```449:457:src/run.py
        ray.init(
            num_gpus=1,
            ignore_reinit_error=True,
            object_store_memory=10 * 1024 * 1024 * 1024,
```

Hardcoded single‑GPU. Breaks multi‑GPU boxes and ignores
`CUDA_VISIBLE_DEVICES`.

**Fix:** `num_gpus=torch.cuda.device_count()`.

### 3.5 `main_experiment` control‑flow oddity

`repetitions = cfg.get("repetitions", 1),` — the trailing comma makes
`repetitions` a one‑element tuple, not an int. It is shadowed/unused
later. Dead misleading code — remove.

---

## 4. Data partitioning (partitioning.py)

### 4.1 `np.random.seed(123)` hardcoded in `label_dirichlet_partition`

Every repetition, every β, every `num_clients` gets the same partition.
This is the dominant reproducibility problem: reported std across
repetitions is training‑noise only.

**Fix:** thread a `seed` argument through, controlled from config.

### 4.2 `while min_size < min_require_size:` loop

Dirichlet resampling can loop many times for very large K or small β.

**Fix:** add a `max_tries` guard.

### 4.3 `get_in_comm_indexes` references undefined `DEVICE`

```134:136:src/dataprocessing/partitioning.py
        communicate_index = communicate_index.to(DEVICE)
        communicate_indexes.append(communicate_index)
```

Module‑level `DEVICE` is commented out. The function appears unused, but
if ever called will raise `NameError`.

**Fix:** delete the function, or accept `device` as an argument.

### 4.4 `fulltraining_flag=True` causes training data leakage across clients

In `reset_subgraph_features2` with `fulltraining_flag=True`, the k‑hop
expanded subgraph keeps *all* masks from the global graph, including
masks of nodes owned by other clients. A client trained with
`fulltraining_flag=True` is therefore optimizing over labels that do not
belong to it. Either it is data leakage or it is an oracle baseline —
must be called out explicitly.

### 4.5 Feature‑propagation per‑client vs global inconsistency

Propagation is applied independently on each client subgraph. For a
boundary node that appears in one client's k‑hop, the propagated feature
is a function of the *local* subgraph only. The global `data.x` is **not**
reprocessed (except for PE). Consequence: the server model during
`test_global_model(data)` sees original, unpropagated features, while
clients were trained on propagated features.

**Fix:** either apply the same FP globally to `data.x` for the global
eval, or stop reporting `test_global_model` as the primary metric when
FP is on.

### 4.6 Facebook masks are not boolean tensors

`_load_facebook` sets masks to `range(...)` objects. `create_subgraph`
does `data.train_mask.cpu()[node_mask]`, which will fail on `range`.

**Fix:** convert to boolean tensors in the dataset loader.

---

## 5. Feature Propagation (data_utils.py, propagation_functions.py)

### 5.1 Two nearly identical propagation implementations

- `propagate_features` (data_utils.py) — production path.
- `propagate_features_efficient` (propagation_functions.py) — unused.

**Fix:** consolidate or mark one deprecated.

### 5.2 `monte_carlo_random_walk` is O(V × num_walks × walk_length) in Python

Unused in practice.

**Fix:** drop it, or guard behind a size threshold.

### 5.3 `get_personalized_pagerank_matrix` bug — `page_rank` mode is broken

```37:46:src/dataprocessing/propagation_functions.py
    for i in range(max_iter):
        next_ppr = torch.zeros_like(ppr)

        for node in range(num_nodes):
            neighbors = col[row == current_node]  # NameError: `current_node`
```

`current_node` is undefined. Any call through `mode="page_rank"` in
`propagate_features` will crash.

**Fix:** replace with a power‑iteration on the normalized adjacency (see
`propagate_features_efficient` for a correct sketch).

### 5.4 `diffusion_kernel` large‑graph fallback is not a diffusion

```212:217:src/dataprocessing/propagation_functions.py
    if num_nodes > 50000:
        identity_approx = SparseTensor.eye(num_nodes).to(device)
        diffusion = identity_approx - sparse_scalar_mul(laplacian, t)
```

First‑order Taylor `I - tL` is not a diffusion kernel for t ≥ 1 (non‑PSD,
can have negative eigenvalues). Misleading name.

**Fix:** force Chebyshev path for large graphs, or warn loudly.

### 5.5 Convergence tolerance on L2 norm of full feature matrix

```255:267:src/dataprocessing/data_utils.py
        if prev_out is not None:
            delta = torch.norm(out - prev_out).item()
            ...
            if delta < tol:
```

For ogbn‑arxiv (~169K nodes × ~192 dims), per‑entry changes of `1e‑5`
give total L2 ≈ 0.06, so `tol=1e-6` **never triggers**. Iterations
always run to `num_iterations`.

**Fix:** normalize the delta: `delta / torch.norm(prev_out).clamp_min(1e-12)`
or use L∞.

### 5.6 `compute_dirichlet_energy` per iteration is expensive

On arxiv this is ~1.2M edges × hidden_dim per iteration, done
`num_iterations` times.

**Fix:** guard behind `log_energy: bool`, or subsample edges.

### 5.7 FP device default contradicts comment

```40:40:conf/base.yaml
feature_prop_device: "cuda"  # Memory optimization: Use CPU for preprocessing to save ~9GB GPU memory
```

Comment says CPU, value is CUDA.

---

## 6. Positional encoding (positional_encoding.py)

### 6.1 Per‑client vs global RFP inconsistency

Same root cause as §3.3 / §4.5: PE is a function of (node, subgraph), so
clients and the global evaluator see different PEs.

### 6.2 QR on every iteration is expensive

With `pe_r=64`, `pe_P=16`, `N=169K`, this is 16 × O(N × r²) ≈ 1.1e10 ops.

**Fix:** offer `normalize="l2"` for large graphs.

### 6.3 Random seed for RFP not set

`torch.randn(num_nodes, r, device=device)` — never seeded. Combined with
`seed(123)` in partitioning and `seed(42)` in mini‑batch, this is the
third uncontrolled RNG layer.

---

## 7. Models (models.py)

### 7.1 PubmedGAT final layer has wrong output shape — likely bug

```463:465:src/models.py
        # Output layer (8 heads for Pubmed, as in original)
        self.convs.append(GATv2Conv(self.dim_h * heads, dim_out, heads=8))
```

`GATv2Conv` default is `concat=True`, so with `heads=8` and
`out_channels=dim_out`, the output tensor has shape `[N, dim_out * 8]`.
Then:

```475:477:src/models.py
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
```

`log_softmax` is applied over `dim_out * 8` "classes". NLLLoss against
integer labels in `[0, dim_out)` still runs (indices < `dim_out * 8`) but
probability mass bleeds to the extra columns.

**Fix:** `heads=1` in the output layer, or `heads=8, concat=False`.

### 7.2 Mixed use of `GCN` vs `GCN_arxiv`, `GAT` vs `GAT_Arxiv`

`use_unified_model` flag picks between two near‑duplicate class families
with slightly different default normalizations / layer patterns. Four
code paths per model type is more than necessary.

**Fix:** consolidate.

### 7.3 Dead model code in hot path

`VanillaGNN` / `MLP` / `SparseVanillaGNN` are referenced by
`train`/`evaluate`/`test` but never instantiated by `run.py`. The
`VanillaGNN` branch calls `to_dense_adj` which OOMs on arxiv.

**Fix:** remove the dead `isinstance` branches.

---

## 8. Datasets (datasets.py)

### 8.1 Facebook masks are Python `range` objects

See §4.6. Will crash the partitioning pipeline.

### 8.2 OGB path reuse

```72:78:src/dataprocessing/datasets.py
        elif name in self.SUPPORTED_DATASETS["ogb"]:
            return self.config['paths']['ogbn']['arxiv']
```

`ogbn-products` is routed to the `arxiv` path. OGB handles subfolder
naming internally, but this is misleading.

### 8.3 Amazon mask fallback

`transforms.AddTrainValTestMask` was removed from PyG long ago. The
`except AttributeError` handles it, but different PyG versions raise
different errors.

**Fix:** version check or `hasattr`.

---

## 9. Configuration (base.yaml and derivatives)

### 9.1 `use_pe` list/scalar inconsistency

`base.yaml` has `use_pe: [false]`, some dataset configs have
`use_pe: true` (scalar). `run_experiments.py` iterates over `use_pe`
values but does not wrap scalars — a scalar crashes the loop.

**Fix:** add `use_pe` to the list of auto‑wrapped keys.

### 9.2 Magic numbers not exposed

Gradient clip (`max_norm=1.0`), early‑stopping `patience_threshold=10`,
seed values are all hardcoded. Lift into `base.yaml`.

### 9.3 `training:` section is dead config

`base.yaml` defines a `training.default` block with `lr`, `optimizer`,
`weight_decay`, `epochs`, `patience`. `client.py` / `run.py` read these
at the top level. The `training:` block is never used.

---

## 10. Experiment orchestration (run_experiments.py)

### 10.1 Monkey‑patching `save_results_to_csv`

```316:323:src/experiments/run_experiments.py
                            def patched_save_func(results, filename=None):
                                ...
                            run_utils.save_results_to_csv = patched_save_func
```

Monkey‑patching a module global per experiment is fragile (race
conditions with parallelism, leaks between runs).

**Fix:** inject the target CSV path as an argument.

### 10.2 Double Ray init + forced `time.sleep(1)`

The "outer shutdown/init per experiment" + "inner `main_experiment`
`ray.init`" pattern needs a `time.sleep(1)` to avoid CUDA corruption — a
symptom of `torch_sparse` SparseTensor holding internal CUDA state.
Fragile under long sweeps.

**Fix:** subprocess‑per‑experiment harness instead of in‑process Ray
re‑init.

### 10.3 Summary directory resolution

```272:275:src/experiments/run_experiments.py
    results_dir = os.path.abspath(cfg["results_dir"])
    summary_dir = os.path.join(os.path.dirname(results_dir), "../results_summary", os.path.basename(results_dir))
```

Relative `../results_summary` depends on the absolute layout of the run
directory. Brittle.

---

## 11. Cross‑cutting scientific concerns — severity table

| # | Issue | Severity | Fix difficulty |
|---|-------|----------|----------------|
| 1.1 | Unweighted averaging instead of FedAvg | High | Easy |
| 1.3 | BatchNorm running stats averaged across clients | High | Medium |
| 3.3 / 4.5 | FP/PE computed locally per client but global test uses untouched graph | High | Medium |
| 4.1 | Partitioning seeded with `np.random.seed(123)`, hiding partition variance | High | Trivial |
| 2.3 | GAT_Arxiv missing from evaluate/test → crash | High | Trivial |
| 5.3 | `page_rank` mode references undefined `current_node` → crash | High | Trivial |
| 7.1 | PubmedGAT output has `dim_out * 8` classes | High | Trivial |
| 3.1 | Early‑stopping `best_eval_loss` update bug | Medium | Trivial |
| 1.4 | Initial broadcast not synced | Medium | Trivial |
| 5.4 | `diffusion_kernel` first‑order fallback is not a diffusion | Medium | Easy |
| 5.5 | FP tolerance effectively never triggers on large graphs | Medium | Easy |
| 4.4 | `fulltraining_flag=True` leaks other clients' labels | Medium | Document or forbid |
| 2.5 | Inconsistent seeding across train paths + RFP unseeded | Medium | Medium |
| 3.2 | Macro‑average of per‑client test conflated with global | Medium | Easy |
| 4.6 / 8.1 | Facebook masks aren't tensors | Medium | Trivial |

---

## 12. Concrete suggested next actions (backward‑compatible)

Per project convention, all new behavior should be guarded by flags and
leave existing behavior reachable.

1. **Weighted‑FedAvg path.** Add `cfg["aggregation"]` with values
   `"mean"` (current) or `"fedavg"`. In `server.train_clients`, when
   `"fedavg"`, collect `client.get_train_size.remote()` once at init and
   weight the sum. Keep `"mean"` as default.
2. **Page‑rank bugfix as `page_rank_v2`.** Keep the old broken entry
   reachable only behind a flag so existing (presumably unused) calls do
   not change behavior silently.
3. **Thread seed through.**
   `label_dirichlet_partition(..., seed=cfg.get("partition_seed", 123))`,
   `generate_rfp_encoding(..., seed=cfg.get("pe_seed", None))`,
   `set_seed(cfg.get("train_seed", 42))`. Defaults preserve current
   behavior.
4. **`evaluate_v2` / `test_v2`** that include `GAT_Arxiv` and guard
   against empty masks. Keep the originals.
5. **Fix PubmedGAT** behind `pubmedgat_fix_heads: true` (default
   `false`, so existing checkpoints reproduce).
6. **Expose `grad_clip_norm`, `patience`, `aggregation`,
   `bn_strategy`** in `base.yaml`. Defaults mirror current hardcoded
   values.
7. **Add `global_eval_uses_client_preprocessing`.** When `true`, apply
   the same FP+PE to the global graph before `test_global_model`. When
   `false`, current behavior.

---

## Appendix: severity summary (highest impact first)

The issues most likely to change reportable accuracy numbers:

1. §1.1 — Unweighted aggregation instead of FedAvg.
2. §1.3 — Naive BatchNorm running‑stats averaging.
3. §3.3 / §4.5 — FP/PE applied per‑client, global test uses unprocessed
   graph.
4. §4.1 — Hardcoded partition seed, repetitions share one partition.
5. §2.3 — `GAT_Arxiv` missing from evaluate/test (will crash on val).
6. §5.3 — `page_rank` propagation mode references an undefined variable.
7. §7.1 — PubmedGAT final layer produces `num_classes * 8` outputs.
