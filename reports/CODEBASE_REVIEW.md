# FP Codebase Review

**Date:** 2026-04-28
**Scope:** Full review of the federated GNN training stack — data partitioning, feature propagation, models, FL aggregation, experiment runner.
**Goal:** Identify bugs, latent issues, inefficiencies, and reproducibility risks. Findings are prioritized by severity.

---

## Setup terminology

| Term | Meaning |
|------|---------|
| `data_loading: full` | **Federated upper bound.** Every client owns its k-hop subgraph with all feature values present (including those of nodes assigned to other clients in the partition). Train/val/test masks remain partition-restricted. This is the FL ceiling — it isolates the effect of partition + aggregation from the effect of feature missingness. |
| `data_loading: zero_hop` | Each client sees only its own partition's nodes; no cross-client feature access. |
| `data_loading: adjacency / diffusion / chebyshev_diffusion` | k-hop subgraphs with cross-client features imputed by feature propagation. The methods under study. |
| `aggregation: mean` | Server averages client params with uniform weight `1/K`. |
| `aggregation: fedavg_weighted` | Server weights each client by `|D_k| / sum_k |D_k|` (McMahan FedAvg). |

The numbers reviewed in this report (81% vs 70%) both come from `data_loading: full` runs, so they reflect the upper-bound regime. Differences in absolute accuracy in the upper bound are about FL dynamics, not about feature missingness.

---

## 1. Real bugs (correctness)

### 1.1 GAT dropout never applied to attention coefficients

`src/fedgnn/models/core.py:386, 398, 409` — `GATv2Conv(...)` is constructed without `dropout=`. The `self.dropout` field is only used for input dropout in `forward()`. The original GAT paper applies dropout to attention coefficients as well. That regularization path is dead.

A previous attempt to enable it dropped accuracy to ~41%, which suggests the federated regime is fragile to attention dropout. Worth revisiting with retuned hyperparameters once aggregation is settled.

### 1.2 Undefined `DEVICE` reference

`src/fedgnn/data/partitioning.py:156` — `communicate_index = communicate_index.to(DEVICE)`. The module-level `DEVICE` global is commented out at line 13. `get_in_comm_indexes` is currently unused; this would `NameError` if anyone wired it up.

### 1.3 `train()` returns mislabeled values

`src/fedgnn/fl/train.py:153-155`:

```python
final_val_loss, final_val_acc = evaluate(model, data, criterion)
return final_val_loss, training_acc, training_losses, training_accuracies
```

Returns validation loss but **training** accuracy (last epoch). Caller at `client.py:319` unpacks as `loss, acc = train(...)`, so all "client training accuracy" reporting is post-overfit and meaningless. Early stopping uses `eval_results` separately, which is correct, so this doesn't break training — only logging.

The minibatch path has the same shape inconsistency at line 358.

### 1.4 Struct-reg shape mismatch (latent)

`src/fedgnn/fl/train.py:113`: `struct_loss = torch.mean((h_first - x_fp) ** 2)`. For GAT, `h_first` has shape `[N, dim_h * heads]` and `x_fp = data.x` has shape `[N, input_dim]`. Unless `dim_h*heads == input_dim`, this raises a `RuntimeError`. Default `struct_reg_lambda=0.0` so it's never executed. Latent bug.

### 1.5 Default-value drift between `base.yaml` and code fallbacks

| Key | `base.yaml` | Code fallback | Files |
|-----|-------------|---------------|-------|
| `feature_prop_tolerance` | `1e-6` | `1e-3` | `partitioning.py:309` |
| `num_iterations` | `80` | `50` | `partitioning.py:308, 322` |
| `diffusion_t` | `1.0` | `0.1` | `propagation.py:247` |
| `repetitions` | `5` | `1` (in two places) and `5` (in one) | `run.py:671, 720`; `run_experiments.py:400` |

If any code path bypasses OmegaConf merge (e.g., a wandb sweep, a shallow-copied dict), behavior silently shifts. **High priority** because this is the single most common cause of "I ran the same config and got different numbers."

### 1.6 `Dirichlet` partition retry doesn't reseed

`partitioning.py:50` — `while min_size < min_require_size` re-samples without resetting RNG. With Cora-sized graphs and `beta=10000`, converges in 1 iter; with adversarial inputs, can produce different partitions even at the same seed.

### 1.7 `compute_dirichlet_energy` is O(E·F) and computed unconditionally

`partitioning.py:354` calls it on the full graph regardless of `log_feature_prop_energy`. For ogbn-arxiv this materializes `[1.3M, 128]` of edge differences just to write a JSON field. Gate behind the flag.

### 1.8 GCN input-layer dropout is `p=0.0`

`models/core.py:117` — `F.dropout(x, p=0.0)` is a no-op. Probably a leftover from a refactor.

### 1.9 PubmedGAT and GAT output layers diverge silently

`models/core.py:464` uses `heads=8, concat=False` for the output; `core.py:409` uses `heads=1`. Different model behavior depending on dataset name. Consolidate to a single class with `heads_out` config.

---

## 2. FL aggregation

### 2.1 Two aggregation strategies, two failure modes

```
_aggregate_mean             — uniform 1/K
_aggregate_fedavg_weighted  — |D_k| / sum_k |D_k|
```

For GAT on Cora (`hop=2, beta=10000`), `mean` reaches 81% and `fedavg_weighted` reaches 63-71%. See `GAT_ACCURACY_GAP_ANALYSIS.md` for the hypothesis trace.

### 2.2 `broadcast_params(sync=False)` is fire-and-forget

`server.py:328-336`. Currently safe because Ray's per-actor FIFO orders updates before the next training call, but it's a latent race if anyone adds an asynchronous evaluation path. Default to `sync=True`; the cost is negligible for parameter-only payloads.

### 2.3 Per-round all-clients evaluation is expensive

`server.py:238` — `evaluate_clients()` runs every round on every client. For 200 rounds × 10 clients this is the second-largest GPU cost after training. Add `eval_every_n_rounds`.

### 2.4 No client sampling

All clients participate every round. This is FedSGD over all clients, not McMahan-style FedAvg with sample fraction `C`. Not a bug — but inconsistent with how the FedAvg name is typically used in the literature.

### 2.5 `compare_model_parameters` uses strict equality

`utils/run.py:131-134` — `np.array_equal` on float tensors. Works because broadcast is bit-exact. Switch to `np.allclose(atol=0)` if any non-deterministic op is later introduced.

### 2.6 `ray.kill(client)` skips actor destructors

`fl/run.py:651`. With `keep_data_on_gpu=True`, GPU tensors held by the actor leak until the Python process exits. Use `client.__ray_terminate__.remote()`.

---

## 3. Models

### 3.1 GAT: see 1.1 (attention dropout never applied) and 1.4 (struct-reg shape mismatch).

### 3.2 Dead/duplicate code in `models/core.py`

- Duplicate `import torch` at lines 130-138.
- Commented-out `GCN_arxiv` at lines 298-323.
- `VanillaGNN`, `MLP`, `SparseVanillaGNN` (lines 481-525) referenced by `train.py` isinstance ladders but never instantiated by `instantiate_model` or `FLClient.__init__`. Dead.

### 3.3 No `reset_parameters` between repetitions

`run.py` rep loop (line 720) doesn't call `model.reset_parameters()` between reps. For GCN/GAT the model is **re-instantiated** per rep (via `run_with_server` → `instantiate_model`), so this is fine in practice. For `BatchNorm` running stats this would matter, but Cora/Pubmed/Citeseer all use `normalization=none`. Affects only ogbn-arxiv runs.

### 3.4 ogbn-products device is hardcoded by name

`fl/client.py:87, 120` — `if dataset_name == "ogbn-products": _init_device = cpu`. Use a node-count threshold so future large graphs don't need string-matching.

---

## 4. Data pipeline

### 4.1 PE generation is asymmetric across `data_loading` modes

- `loaders.py:142-173` generates **global** RFP for `load_and_split_with_khop` (used by `full`, `adjacency`, `diffusion`, etc.).
- `loaders.py:36-56` does **not** generate global RFP for `load_and_split` (`zero_hop`).

Run `use_pe=True, data_loading=zero_hop` and clients train with PE-augmented features but the global test surface uses raw features. Distribution mismatch.

### 4.2 `partition_data` silently disables PE if FP is off

`partitioning.py:368` — `use_pe = use_feature_prop and _as_bool(use_pe)`. Documented in `data_loading_uses_pe`, but the user gets no warning if their config asked for PE. Add a print warning.

### 4.3 Adaptive-`t` BFS is pure-Python O(V·E)

`partitioning.py:184-216` — builds adjacency lists in Python and BFS-samples 20 source nodes per client. Milliseconds for Cora; minutes for ogbn-arxiv. Gate by node count or rewrite with sparse ops.

### 4.4 Sequential per-client subgraph creation copies full features each time

`partitioning.py:358-364` and `create_subgraph` (line 90) — `data.x.cpu()[subset]` runs num_clients times. Hoist `data.cpu()` once outside the loop.

### 4.5 `_select_or_create_node_split` for WebKB ignores `experiment_seed`

`datasets.py:200-237` uses its own `split_seed`. C5 reproducibility doesn't propagate to WebKB.

---

## 5. Experiment runner

### 5.1 Nine-deep nested for loops

`run_experiments.py:346-406`. Cartesian product over PE × dataset × loading × model × beta × hop × clients × num_iter × diffusion_t × alpha. Replace with `itertools.product` and a flat loop.

### 5.2 Monkey-patching `save_results_to_csv` per experiment

`run_experiments.py:368-377` rebinds the imported function each iteration. Global side effect — if two `run_experiments` calls run in the same process, the patches race. Pass the filename via cfg instead.

### 5.3 `save_results_to_csv` writes "results.csv" twice

`utils/run.py:79-91` — when env-var path is set, also writes to a fixed `"results.csv"` for "backward compatibility". With parallel experiments dumping to the same cwd this is a data race that corrupts the file.

### 5.4 OOM handling is not adaptive after rep 0

`run.py:806-809` re-raises hard. The `adaptive_device` flag promises GPU→CPU fallback but only checks **time** after rep 0; an OOM during rep 0 crashes the whole run. Should retry the rep on CPU.

### 5.5 Wandb config override happens twice

`run_experiments.py:281-282` and `run.py:783-784` both walk `wandb.config` and copy into cfg. The second runs per-rep and may mutate `current_cfg["experiment_seed"]` between reps. Breaks C5 seeding under wandb sweeps.

### 5.6 Ray lifecycle ownership is split

`run.py:705` initializes Ray inside `main_experiment`; `run_experiments.py:430` shuts it down outside. Caller and callee both touch the lifecycle. Move both to one owner.

---

## 6. Performance / efficiency

### 6.1 `torch.cuda.empty_cache()` on hot paths

~12 calls per round through various memory utilities. Each forces a synchronization. Cora cost: ~100ms × 200 rounds ≈ 20s wasted per rep. Restrict to OOM-recovery paths.

### 6.2 `gc.collect()` × 3 loops

`propagation.py:265, 469`, `run_experiments.py:435`. Cargo-culted from a real torch_sparse leak fix and applied broadly. One `gc.collect()` is enough except in the diffusion teardown.

### 6.3 `prev_out = out.clone()` per FP iteration

`propagation.py:379`. For Cora this is 30 MB per iter × num_iterations × num_clients. Track delta in-place against the previous iter without a clone.

### 6.4 Multi-scale FP rebuilds the diffusion kernel every call

`propagation.py:461-465`. Across reps with the same `(num_clients, beta, partition_seed, hop)`, the kernel is identical. Cache it.

### 6.5 Parallel FP threads share a CUDA context

`partitioning.py:438`. `ThreadPoolExecutor` with GPU work serializes on the same stream. Default `fp_max_concurrent: 1`, so safe — but the option misleads.

---

## 7. Reproducibility

### 7.1 `experiment_seed` defaults to `null` (unseeded)

`base.yaml:35`. Default reproducibility is **off**. For paper-quality benchmarking this is the wrong default.

### 7.2 cuDNN determinism

`fl/train.py:23` sets `cudnn.deterministic=True` only when `set_seed` is called, which is gated on `seed is not None`. Default unseeded runs are non-reproducible across hardware.

### 7.3 Per-client seed only changes dropout patterns

`client.py:300` — `client_seed = experiment_seed + client_id`. But `train()` re-seeds *after* model construction, so per-client weight inits are identical (server broadcasts the same params anyway). Per-client seed only varies dropout/sampling. Document this.

---

## 8. Prioritized action list

| # | Severity | Action |
|---|----------|--------|
| 1 | **High** | Align defaults: `feature_prop_tolerance`, `num_iterations`, `diffusion_t`, `repetitions` in code fallbacks vs base.yaml |
| 2 | **High** | Make `experiment_seed: 0` the default; document per-client seeding semantics |
| 3 | **High** | Fix `data_loading=zero_hop` to generate global PE for symmetry with k-hop modes |
| 4 | **Med** | Fix GAT attention dropout (test carefully; previous attempt collapsed accuracy) |
| 5 | **Med** | Fix `train()` return signature to be unambiguously `(val_loss, val_acc)` |
| 6 | **Med** | `eval_every_n_rounds` config; default 1 for backward compatibility |
| 7 | **Med** | OOM fallback to CPU on rep 0 (`adaptive_device` should cover this case) |
| 8 | **Med** | Replace `ray.kill` with graceful actor termination |
| 9 | **Med** | Single-source `repetitions` default |
| 10 | **Low** | Delete dead model code (commented-out `GCN_arxiv`, unused vanilla MLPs) |
| 11 | **Low** | Consolidate `GAT` and `PubmedGAT` into one class with `heads_out` config |
| 12 | **Low** | Replace 9-deep nested loops in `run_experiments.py` with `itertools.product` |
| 13 | **Low** | Remove `torch.cuda.empty_cache()` from hot paths; keep only at OOM-recovery |
| 14 | **Low** | Gate `compute_dirichlet_energy` behind `log_feature_prop_energy` |
| 15 | **Low** | Fix `partitioning.py:156` undefined `DEVICE` (delete `get_in_comm_indexes` if dead) |

---

## 9. What's NOT broken

To balance the above: the core FL loop is sound, parameter aggregation arithmetic is correct, the C2 fix for global eval surface (testing the aggregated model on per-client preprocessed test graphs) is the right design, the C5 experiment-seed plumbing through partition + RFP + per-client training is consistent, the FedBN strategy is symmetric on client and server, FP convergence checks (delta + tolerance) work, and the PE generation with QR normalization is reasonable for the graph sizes involved.

The known gaps in the FL setup (no client sampling, no DP, no comm-cost instrumentation) are scope decisions, not bugs.

---

*Report generated as part of the GAT baseline investigation, 2026-04-28.*
