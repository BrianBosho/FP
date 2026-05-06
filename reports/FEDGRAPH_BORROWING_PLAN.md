# FedGraph Borrowing Plan for FP

This document summarizes what FP can borrow from `/home/bosho/fedgraph`, especially for large-graph execution and experiment management. The goal is not to copy FedGraph wholesale. FP already has richer propagation, model, and Ray lifecycle logic. The useful lesson from FedGraph is a cleaner path from configuration to execution to parseable results.

## Executive Summary

FedGraph is valuable as a reference for:

- Per-client graph shard caching.
- Simple config-to-run entry points.
- Fixed benchmark result schemas.
- Phase-level timing and communication accounting.
- Client scalability experiments.
- Ray cluster and Prometheus/Grafana monitoring patterns.

FP should not copy FedGraph's dense FedGCN feature pretraining path for large graphs. That path materializes dense `num_nodes x feature_dim` tensors and is not suitable for products or papers100M-scale workloads.

## What To Borrow

### 1. One Config-Driven Entry Point

FedGraph's common pattern is simple:

```python
config = attridict({...})
run_fedgraph(config)
```

FP should keep its richer YAML system but converge on one obvious command:

```bash
python -m src.fedgnn.experiments.run_experiments --config experiments/configs/<experiment>.yaml
```

Shell scripts should remain only as thin launchers for environment setup, logging, CUDA flags, and Ray cleanup. They should not encode datasets, models, betas, propagation modes, or algorithm choices.

### 2. Per-Client Shard Cache

FedGraph can save each client's partitioned tensors to Hugging Face and reload them later. The useful idea is the shard boundary, not Hugging Face itself.

FedGraph saves per client:

- `local_node_index.pt`
- `communicate_node_index.pt`
- `adj.pt`
- `train_labels.pt`
- `test_labels.pt`
- `features.pt`
- `idx_train.pt`
- `idx_test.pt`

FP should implement and standardize a local/shared filesystem cache first:

```text
artifacts/shard_cache/<dataset>_<mode>_c<num_clients>_b<beta>_h<hop>_<hash>/
  metadata.json
  client_0000.pt
  client_0001.pt
  ...
```

The metadata should include dataset, preprocessing mode, beta, hop, seed, feature dimensions, config hash, code version, and compatibility checks.

### 3. Explicit Communication Index Algebra

FedGraph makes the client boundary explicit:

- local nodes owned by the client
- communication nodes visible to the client
- local train/test indices inside the communication block
- per-client edge index

FP currently often materializes full PyG `Data` subgraphs. For large graphs, FP should preserve explicit index structures so large client data can be loaded lazily or reconstructed inside actors only when needed.

Proposed FP direction:

- Keep `Data`-based paths for existing behavior.
- Add an index/shard-backed path for large graphs.
- Let Ray actors load their own shard from disk instead of receiving large `Data` objects through the object store.

### 4. Fixed Benchmark Result Block

FedGraph prints a stable parseable block at the end of each run:

```text
CSV FORMAT RESULT:
DS,IID,BS,TotalTime[s],PureTrainingTime[s],CommTime[s],FinalAcc[%],CommCost[MB],PeakMem[MB],AvgRoundTime[s],ModelSize[MB],TotalParams
...
```

FP should print a similar block for every experiment, even when JSON results are also written. This makes raw logs useful after partial failures, SIGTERM, OOM, or interrupted runs.

Recommended FP schema:

```text
FP CSV FORMAT RESULT:
Dataset,Model,DataLoading,Beta,Clients,Hop,UsePE,Seed,Status,TotalTime[s],LoadTime[s],ShardTime[s],PartitionTime[s],PropTime[s],ActorInitTime[s],TrainTime[s],EvalTime[s],CommTime[s],FinalAcc,BestAcc,PeakCPU[MB],PeakGPU[MB],CommCost[MB],ModelSize[MB],TotalParams,ResultPath
```

### 5. Phase-Level Timing

FedGraph tracks initialization, pretrain, training, and communication. FP should use more FP-specific phases:

- dataset load
- shard cache lookup/load
- shard cache build
- partition/subgraph build
- feature propagation
- positional encoding
- Ray initialization
- actor initialization
- training
- evaluation
- aggregation
- broadcast
- result writing

These should be saved to JSON and printed in the final CSV block.

### 6. Communication Cost Accounting

FedGraph estimates theoretical communication from model size, number of clients, and upload/download direction.

FP should record at least:

- model size in MB
- parameter upload MB per round
- parameter broadcast MB per round
- total theoretical communication MB
- optional Ray object-store bytes, if available
- shard cache read/write size
- actual network usage when running in cluster mode

This is especially useful for comparing full-batch, mini-batch, shard-backed, and propagation-heavy runs.

### 7. Client Scalability Runs

FedGraph keeps client-count logs such as `NC5.log`, `NC10.log`, `NC15.log`, and `NC20.log`, then parses them into scalability plots.

FP should formalize a scalability suite:

```yaml
sweep:
  datasets: [ogbn-arxiv]
  models: [GCN]
  data_loading: [full]
  beta: [10000]
  hop: [2]
  num_clients: [2, 5, 10, 15, 20]
```

Metrics should include:

- final accuracy
- best accuracy
- total time
- training time
- preprocessing time
- peak CPU/GPU memory
- communication cost
- average round time
- failed client count
- completed rounds before failure, if any

### 8. Raw Log Parsers

FedGraph's plotting scripts parse raw logs directly. FP already has better JSON consolidation, but raw-log parsing is valuable when JSON is not written due to failure.

Add:

```text
experiments/parse_logs.py
experiments/parse_scalability_logs.py
```

These should extract:

- experiment config line/block
- completed round metrics
- exception summaries
- final CSV block, if present
- failure type: OOM, CUDA driver, Ray actor death, SIGTERM, timeout

### 9. Cluster Monitoring Pattern

FedGraph's `Monitor` polls Prometheus/Ray metrics for memory and network in cluster mode.

FP should borrow this as an optional backend:

- local mode: `psutil`, `resource`, CUDA memory, Ray state if available
- cluster mode: Prometheus/Ray metrics

This should be a telemetry layer, not mixed deeply into training logic.

### 10. Benchmark-Oriented Plotting

FedGraph has simple scripts for:

- communication cost plots
- client scalability plots
- accuracy curves
- framework comparisons

FP should keep its current result consolidation but add canonical output tables:

```text
experiments/output/scalability.csv
experiments/output/main_results.csv
experiments/output/propagator_ablation.csv
experiments/output/failure_summary.csv
```

Plotting should consume these canonical CSVs, not scrape many ad hoc result shapes.

### 11. Run Ledger And Queue Discipline

FP already has scheduler pieces under `experiments/scheduler/`. Treat that as a first-class experiment ledger rather than a side utility.

Each launched condition should have a durable run packet with:

- requested config
- effective config
- config hash
- run id
- attempt id
- status
- start/end timestamps
- result directory
- log path
- failure summary
- cache hit/miss state

This makes `resume_completed` stronger: the runner should skip only conditions with a valid success packet, and should be able to relaunch failed or interrupted attempts without guessing from result directories.

### 12. Preflight Cost And Feasibility Checks

Before launching expensive experiments, FP should run a cheap preflight pass that estimates:

- dataset size
- client shard sizes
- feature dimension after propagation or PE
- expected model parameter count
- expected model communication per round
- expected Ray object-store pressure
- GPU memory budget per client
- cache hit/miss status

The preflight should produce a small table and fail fast on configurations that are likely to OOM or collide with current resource limits. This matters more for efficiency than any individual training optimization, because it avoids multi-hour failed runs.

### 13. Requested Vs Effective Configuration

FP should record both the user-requested config and the actual config used at runtime. This is especially important when the code adapts settings for safety, such as reducing `ogbn-products` clients or changing device placement.

Recommended fields:

- `requested_num_clients`
- `effective_num_clients`
- `requested_device`
- `effective_device`
- `requested_data_loading_device`
- `effective_data_loading_device`
- `requested_max_concurrent_clients`
- `effective_max_concurrent_clients`
- `adaptive_changes`

Any automatic change should be printed in the final CSV/JSON result. Silent adaptation makes experiments faster in the moment but harder to trust later.

### 14. Cache Locking And Cache Lifecycle

The shard cache should use a lock or atomic build directory so two concurrent runs do not build the same expensive cache at once.

Recommended cache states:

- `building`
- `ready`
- `failed`
- `stale`

Recommended files:

```text
manifest.json
build.lock
build.log
cache_stats.json
```

Add a cache maintenance command later:

```bash
python -m src.fedgnn.data.shard_cache --list
python -m src.fedgnn.data.shard_cache --verify artifacts/shard_cache/<cache_id>
python -m src.fedgnn.data.shard_cache --gc --max-size-gb 500
```

### 15. Partial Result Durability

Long runs should write useful results before the final summary. FP already has optional per-round history; the next step is flushing it to disk during the run.

Write append-only files such as:

```text
per_round.jsonl
per_repetition.jsonl
telemetry.jsonl
events.jsonl
```

This gives usable convergence curves after SIGTERM, OOM, Ray actor death, or manual interruption. Final JSON can still exist, but it should be a consolidation artifact rather than the only source of truth.

### 16. Statistical Efficiency

To run experiments efficiently, FP should not always spend the full seed budget on every condition.

Use a staged policy:

- run smoke checks first
- run one seed for every condition
- eliminate obviously broken or dominated settings
- run the full seed budget only for promising or ambiguous comparisons
- add extra seeds only when confidence intervals overlap

Canonical summaries should include confidence intervals or standard errors, not only means and standard deviations. This helps distinguish real effects from seed noise without wasting compute.

### 17. Adaptive Scheduling Policy

FedGraph's simple client-count experiments are useful, but FP can go further because it already has resource-aware scheduling.

The scheduler should choose launch slots using:

- free GPU memory
- CPU load
- RAM pressure
- Ray object-store pressure
- dataset size
- model size
- configured client concurrency
- historical memory/time from previous attempts

The run ledger should feed back into scheduling. For example, if `ogbn-arxiv, GAT, clients=20` previously used 38 GB VRAM, the scheduler should not launch another similarly sized job unless enough headroom exists.

### 18. Experiment Ladder

Every major suite should have an explicit ladder:

1. import/config validation
2. dataset-load only
3. shard-cache warmup
4. one-client smoke
5. one-round multi-client smoke
6. one-seed short run
7. full run

This avoids burning time on full sweeps when a lower-level assumption has already failed.

### 19. Provenance Bundle

Every result directory should include a reproducibility bundle:

```text
config.yaml
effective_config.yaml
provenance.json
requirements.freeze.txt
git_status.txt
```

`provenance.json` should include git commit, dirty tree status, Python version, PyTorch/PyG/Ray versions, CUDA version, GPU name, dataset path, dataset checksums where feasible, hostname, and command line.

### 20. Failure Taxonomy

Raw log parsing should normalize failures into a small taxonomy:

- `oom_gpu`
- `oom_cpu`
- `ray_actor_died`
- `ray_object_store_full`
- `cuda_driver_error`
- `nan_metric`
- `timeout`
- `sigterm`
- `config_invalid`
- `dataset_missing`
- `unknown_exception`

This makes failed runs analyzable. Failure rate is itself an experiment-management metric when comparing loaders, propagation modes, and client counts.

## What Not To Borrow

### Dense FedGCN Pretrain

FedGraph's FedGCN path computes local feature sums by creating dense tensors shaped like:

```text
global_node_num x feature_dim
```

This is not viable for truly large graphs. FP should avoid this design for `ogbn-products` and larger.

### Hugging Face As The Default Cache

Hugging Face upload is useful for public/reproducible datasets, but not ideal as FP's default. Local or shared filesystem shard cache should come first.

Reasons:

- avoids leaking private partitions
- avoids authentication friction
- faster iteration on local machines
- easier cache invalidation

### Flat Configs Everywhere

FedGraph uses flat configs because its experiment surface is smaller. FP has model-specific, dataset-specific, and propagation-specific parameters. Fully flattening everything would make large YAMLs harder to read.

Use a hybrid schema:

```yaml
sweep:
  datasets: [Cora, Pubmed]
  models: [GCN, GAT]
  beta: [1, 10000]
  hop: [1, 2]
  data_loading: [full, diffusion]

model_architecture:
  GCN:
    hidden_dim: [64, 128]
    num_layers: [2]
  GAT:
    hidden_dim: [64]
    num_heads: [4, 8]
```

The sweep engine should understand nested model-specific parameters.

### Demo/Fake Data Fallbacks

FedGraph's framework comparison script can synthesize missing framework data for plotting. FP should not borrow this. Missing results should be marked missing.

### Reinitializing Ray For Every Condition By Default

FedGraph often shuts down Ray between benchmark conditions. FP's current reusable Ray lifecycle is better for sweeps. However, FP should support an isolation mode for crash-prone large-graph validation:

```yaml
ray_isolation_per_condition: true
```

## Proposed Config Cleanup

### Canonical Layout

Use `experiments/configs/` as the primary experiment config root:

```text
experiments/configs/
  smoke/
  scalability/
  r1/
  r1b/
  ablation/
  propagator_eval/
  legacy/
```

Keep `conf/base.yaml` for defaults and shared paths. Move old one-off configs into `legacy/` or mark them deprecated.

### Explicit Sweep Variables

The following should be explicit and sweepable:

- `num_clients`
- `num_rounds`
- `epochs`
- `beta`
- `aggregation`
- `bn_fl_strategy`
- `lr`
- `optimizer`
- `decay`
- `grad_clip_norm`
- `models`
- `hidden_dim`
- `num_layers`
- `dropout`
- `normalization`
- `num_heads`
- `datasets`
- `data_loading`
- `hop`
- `use_pe`
- `num_iterations`
- `diffusion_t`
- `chebyshev_k`
- `alpha`
- `use_minibatch`
- `batch_size`
- `num_neighbors`
- `repetitions`
- `max_concurrent_clients`
- `fp_max_concurrent`
- `feature_prop_device`
- `keep_data_on_gpu`
- `adaptive_device`
- `debug`
- `early_stopping_patience`
- `experiment_seed`
- `resume_completed`

Hidden Python defaults should be moved into `conf/base.yaml` or the experiment config.

### Shell Script Policy

Shell scripts may set:

- conda Python path
- `CUDA_VISIBLE_DEVICES`
- `LD_LIBRARY_PATH`
- `PYTORCH_CUDA_ALLOC_CONF`
- log directory
- `ray stop --force` before isolated runs

Shell scripts should not set:

- dataset
- model
- beta
- hop
- propagation mode
- rounds
- epochs
- architecture

## Proposed Implementation Phases

### Phase 1: Telemetry And Result Schema

- Add a lightweight telemetry module.
- Track phase timers.
- Track peak CPU/GPU memory.
- Compute model size and communication estimates.
- Print final FP CSV block.
- Save telemetry into result JSON.

### Phase 2: Config Schema Cleanup

- Define canonical experiment YAML schema.
- Move hidden defaults into config.
- Add support for nested model-architecture sweeps.
- Add config hash to all results.
- Mark old configs/scripts as legacy.

### Phase 3: Shard Cache Standardization

- Standardize shard cache metadata.
- Validate cache compatibility by config hash and dataset stats.
- Load client shards inside Ray actors.
- Keep old full `Data` object path for backward compatibility.

### Phase 4: Scalability Harness

- Add client-count sweep configs.
- Add raw-log parser for partial failures.
- Add canonical scalability CSV.
- Add plots for time, memory, communication, and accuracy vs clients.

### Phase 5: Optional Cluster Monitoring

- Add local telemetry backend.
- Add Prometheus/Ray backend for cluster mode.
- Keep monitoring optional and independent of training logic.

## Priority Recommendations

1. Add FedGraph-style final CSV blocks and phase timing first. This improves every run immediately.
2. Standardize the shard cache we already started using.
3. Clean experiment configs around one entry point and explicit sweep variables.
4. Add raw log parsing so failed scalability runs still produce useful records.
5. Add client scalability configs and canonical output CSVs.

## Bottom Line

Borrow FedGraph's experiment discipline:

- simple entry point
- explicit config
- shard reuse
- fixed parseable output
- phase timing
- communication accounting
- scalability plots

Do not borrow its dense large-graph FedGCN pretrain path or its overly flat config style. FP should keep its richer architecture, but make the experiment surface cleaner, more reproducible, and easier to parse after long-running large-graph jobs.

---

## Remaining Implementation Checklist

Status as of 2026-05-06.

### 1. Config Schema Cleanup
- [x] `conf/base.yaml` holds canonical defaults (model_architecture, training, paths)
- [x] `experiments/configs/smoke/cora_smoke.yaml` — fast end-to-end sanity check
- [x] Config validation warns on unknown/misspelled top-level keys (`utils/config.py`)
- [ ] Nested model sweep support via `model_architecture.<Model>.hidden_dim` in sweep YAMLs
- [ ] Move legacy one-off `conf/test_*.yaml` files to `experiments/configs/legacy/`
- [ ] Shell scripts must not set dataset/model/beta/hop/rounds/epochs/architecture

### 2. Preflight Feasibility Checks  ✅ DONE
- [x] `src/fedgnn/experiments/preflight.py` — estimates dataset size, feature dim,
      model params & MB, comm MB, shard cache state, RAM/GPU/object-store pressure,
      Ray concurrency feasibility
- Usage: `python -m src.fedgnn.experiments.preflight --config <yaml>  [--json]`

### 3. Partial Result Durability  ✅ DONE
- [x] `src/fedgnn/utils/durability.py` — `DurabilityBundle` writes four JSONL streams:
      `per_round.jsonl`, `per_repetition.jsonl`, `telemetry.jsonl`, `events.jsonl`
- [x] `fl/run.py` — flushes per-round and per-repetition records as they complete;
      telemetry snapshot and completion event written at end
- [x] `run_experiments.py` — opens a bundle per condition, passes to `main_experiment`

### 4. Run Ledger / Scheduler Integration  ✅ DONE
- [x] `src/fedgnn/experiments/ledger.py` — append-only JSONL ledger with `RunPacket`,
      `make_condition_key`, status update records
- [x] `run_experiments.py` — appends a packet at start, updates on success/failure/skip
- [x] `resume_completed` checks ledger first (fast O(1) key lookup) before directory scan
- [ ] Scheduler (`experiments/scheduler/`) should read ledger instead of result dirs

### 5. Cache Metadata + GC  ✅ DONE
- [x] `--gc` implemented: evicts corrupted/incomplete first, then oldest ready under budget
- [x] `--list` now shows per-entry sizes and total
- [x] `--verify` checks `schema_version` compatibility (not just file existence)
- [ ] Add dataset stats/checksums and code hash to `build_cache_payload`

### 6. Statistical Efficiency  ✅ DONE
- [x] `src/fedgnn/experiments/staged_policy.py` — `smoke_overrides`, `pilot_overrides`,
      `should_promote_to_full`, `ci_95`, `format_ci`, `enrich_summary_with_ci`
- [x] `run_experiments.py` — enriches each result summary with 95 % CI fields
      (`global_ci95_str`, `client_ci95_str`, …)
- [ ] Wire staged policy into scheduler: smoke → pilot → full promotion logic

### 7. Optional Cluster Monitoring
- [ ] Local backend: psutil + nvidia-smi + Ray dashboard metrics
- [ ] Cluster backend: Prometheus / Ray state API
- [ ] Network and object-store metrics
- Not urgent unless running multi-node or remote Ray clusters
