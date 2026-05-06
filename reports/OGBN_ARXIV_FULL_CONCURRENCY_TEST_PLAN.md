# OGBN-Arxiv Full-Mode Concurrency Test Plan

## Purpose

Determine the highest safe `max_concurrent_clients` for **FedProp full data loading** on the current machine without triggering:

- host RAM out-of-memory conditions,
- Ray worker kills due to memory pressure,
- CUDA out-of-memory errors,
- persistent unschedulable actor placement,
- or unusably degraded throughput.

The goal is not just to find a setting that launches, but to find the highest concurrency that is both **stable** and **operationally useful**.

---

## Primary Question

> For `data_loading=full` on `ogbn-arxiv`, how many clients can be run concurrently on the current node before resource pressure makes the configuration unsafe or impractical?

---

## Scope

This test plan is intentionally narrow.

Included:
- Dataset: `ogbn-arxiv`
- Propagation/data loading mode: `full`
- Model: `GCN`
- Client count in the federated experiment: `10`
- Single-node execution on the current host
- Resource and stability testing under short and medium-duration runs

Excluded:
- `zero_hop`, `adjacency`, `diffusion`
- multi-node scaling
- final paper-quality repetitions during the sweep
- accuracy benchmarking against other methods

---

## System Under Test

### Host
- Workspace: `/home/bosho/FP`
- OS: Linux 5.15.0-1084-aws
- CPU cores: `24`
- RAM: `~62 GiB`
- GPU: `NVIDIA L40-48Q`
- GPU memory: `49152 MiB`

### Dataset
- Name: `ogbn-arxiv`
- Disk path: `/home/shared-space/fedgnn-datasets/ogbn_arxiv/`
- Size on disk: `183 MB`
- Dataset status: already downloaded and previously loaded successfully

---

## Known Prior Evidence

The plan starts from prior observations already seen in this workspace:

1. **Full mode has previously OOM'd at higher concurrency.**
   - Existing notes indicate `full` mode failed at `max_concurrent=5` and `max_concurrent=10`.

2. **Current smoke run uses a safer concurrency value.**
   - `S1_smoke_full.yaml` is currently set to `max_concurrent_clients: 2`.

3. **The current run also shows GPU scheduling pressure.**
   - Ray has emitted repeated warnings that resource requests like `{'CPU': 1.0, 'GPU': 0.5}` cannot always be scheduled immediately.

4. **Host RAM is likely the main bottleneck.**
   - GPU fraction can affect scheduling, but it does not reduce real RAM or VRAM usage per client.

This means the experiments should test both:
- whether higher concurrency can be admitted by Ray, and
- whether it remains stable once admitted.

---

## Experimental Principle

For a target concurrency `N`, we will normally pair:

- `max_concurrent_clients = N`
- `gpu_fraction_per_client ≈ 1 / N`

Examples:
- `N=2` -> GPU fraction `0.5`
- `N=5` -> GPU fraction `0.2`
- `N=10` -> GPU fraction `0.1`

This pairing is a **scheduling hypothesis**, not a guarantee of safety.
It only tells Ray how many actors it may try to place on one GPU.
Actual safety still depends on:
- host RAM,
- GPU VRAM,
- object store overhead,
- CUDA context overhead,
- and runtime behavior of full-mode client training.

---

## Fixed Experimental Settings

Unless a specific ablation says otherwise, all concurrency tests should hold these constant.

### Core experiment settings
- `datasets: [ogbn-arxiv]`
- `data_loading: [full]`
- `num_clients: [10]`
- `models: [GCN]`
- `use_pe: [false]`
- `hop: 2`
- `num_iterations: 50`
- `diffusion_t: 0.1`
- `aggregation: fedavg_weighted`

### Training settings
- `epochs: 3`
- `training.lr: 0.01`
- `training.optimizer: Adam`
- `training.weight_decay: 0`
- `early_stopping_patience`: small for the short sweep; larger for confirmation runs
- `repetitions: 1` during the sweep

### Model architecture
- hidden dimension: `256`
- layers: `3`
- dropout: `0.5`
- normalization: `batch`

### Device / memory settings
- `device: cuda`
- `feature_prop_device`: test-specific; default should be explicit
- `keep_data_on_gpu: false`
- `adaptive_device: true`
- `use_amp: true`
- `prop_dtype: bfloat16`
- `convergence_check_interval: 5`
- `ray_num_gpus: 1`
- `ray_object_store_memory_bytes: 4294967296` where applicable

---

## Knobs We Will Vary

### Primary sweep knobs
1. **Target concurrency**
   - `max_concurrent_clients`

2. **Per-client GPU fraction**
   - target heuristic: `1 / max_concurrent_clients`

### Secondary knobs (only after the primary sweep if needed)
3. `feature_prop_device`
   - `cpu` vs `cuda`

4. `use_amp`
   - `true` vs `false`

5. `prop_dtype`
   - `bfloat16` vs fallback precision if supported

6. `keep_data_on_gpu`
   - usually keep `false`

7. `convergence_check_interval`
   - default `5`; may increase if clone/sync overhead matters

8. Ray object-store cap
   - verify whether limiting object-store usage improves stability

These secondary knobs are for explaining the limit or improving it after baseline characterization.

---

## Resources to Monitor

### 1. Host RAM
Monitor:
- total RAM
- used RAM
- available RAM
- per-process RSS for the runner and client actors
- Ray memory-pressure warnings
- worker kills from the Ray memory monitor

Why it matters:
- this is the most likely failure mode for full data loading.

### 2. GPU VRAM
Monitor:
- total GPU memory used
- per-process GPU memory usage
- growth across concurrency levels
- CUDA OOM messages

Why it matters:
- lower GPU reservation fractions do not reduce actual VRAM demand.

### 3. GPU scheduling / actor placement
Monitor:
- whether Ray can place all requested client actors
- repeated `cannot be scheduled right now` warnings
- actual number of simultaneously active clients

Why it matters:
- a configuration may be memory-safe but still fail to realize the target concurrency.

### 4. CPU and system pressure
Monitor:
- CPU utilization
- system load
- process count / general thrashing symptoms

Why it matters:
- not expected to be the primary ceiling, but useful to confirm.

### 5. Training progress
Monitor:
- whether rounds complete
- per-round wall time
- whether metrics/logs continue updating
- whether early stopping behaves normally

Why it matters:
- a run that technically avoids OOM but makes poor progress is not a good operating point.

---

## Metrics to Record for Every Run

For each run we should capture:

- run ID
- config path / config diff
- target concurrency `N`
- GPU fraction per client
- number of clients actually placed concurrently
- peak host RAM used
- minimum available RAM observed
- peak GPU VRAM used
- per-process GPU memory snapshot
- per-round wall time
- total run duration
- success/failure outcome
- failure mode, if any:
  - host OOM / Ray memory pressure
  - Ray worker kill
  - CUDA OOM
  - unschedulable actors
  - stall / hang
  - NaN/empty result artifacts

---

## Experimental Stages

## Stage 0 — Instrumented baseline

### Goal
Establish the current safe baseline and verify monitoring works.

### Settings
- `max_concurrent_clients = 2`
- `gpu_fraction = 0.5`
- short run: enough rounds to exercise placement and at least one full training cycle
- `repetitions = 1`

### Success criteria
- all requested actors place as expected
- no Ray memory-pressure kills
- no CUDA OOM
- normal training progress
- resource logging collected successfully

### Output
Baseline measurements for RAM, VRAM, scheduling, and round time.

---

## Stage 1 — Primary concurrency sweep

### Goal
Find the highest stable concurrency under the default full-mode configuration.

### Ladder
Test sequentially:
- `N = 2`
- `N = 3`
- `N = 4`
- `N = 5`
- if `5` succeeds, continue with:
  - `N = 6`
  - `N = 7`
  - `N = 8`
  - `N = 9`
  - `N = 10`

### Coupled GPU fractions
- `N=2` -> `0.5`
- `N=3` -> `0.333...`
- `N=4` -> `0.25`
- `N=5` -> `0.2`
- `N=6` -> `0.1667`
- `N=7` -> `0.1429`
- `N=8` -> `0.125`
- `N=9` -> `0.1111`
- `N=10` -> `0.1`

### Run design
- keep all other settings fixed
- one repetition each
- short duration first
- abort escalation when a run is clearly unsafe or unplaceable

### What we learn
- the safe boundary,
- the first failure point,
- and the failure mode at each level.

---

## Stage 2 — Explain the bottleneck

Only needed if Stage 1 shows ambiguity.

### Goal
Separate **scheduler limitations** from **real memory limitations**.

### Example cases
- If a run fails because actors never place, GPU fraction may be the issue.
- If actors place but workers are killed, host RAM is the issue.
- If CUDA OOM occurs, GPU VRAM is the issue.

### Optional targeted tests
- keep the same concurrency but adjust GPU fraction slightly
- compare `feature_prop_device=cpu` vs `cuda`
- compare AMP on/off only if required to explain behavior

### What we learn
Why the concurrency limit appears where it does.

---

## Stage 3 — Confirmation run

### Goal
Verify that the best candidate remains stable in a more realistic run.

### Settings
- use the highest concurrency classified as provisionally safe
- longer run
- early stopping enabled
- still single repetition initially

### Success criteria
- no OOM or worker kills over a longer window
- stable rounds
- continued metric updates
- no artifact corruption / NaNs caused by dropped workers

### What we learn
Whether the chosen operating point is truly usable for real experiments.

---

## Stage 4 — Optional safety-margin check

### Goal
Validate the boundary by testing one step above the chosen setting.

### Example
If `N=4` looks safe, test `N=5` once more with the same best settings.

### What we learn
Whether the recommended concurrency has real headroom or sits on a knife edge.

---

## Candidate Settings Table

| Target concurrent clients | GPU fraction per client | Test priority |
|---:|---:|---|
| 2 | 0.5 | baseline |
| 3 | 0.3333 | high |
| 4 | 0.25 | high |
| 5 | 0.2 | high |
| 6 | 0.1667 | conditional on 5 passing |
| 7 | 0.1429 | conditional on 6 passing |
| 8 | 0.125 | conditional on 7 passing |
| 9 | 0.1111 | conditional on 8 passing |
| 10 | 0.1 | conditional on 9 passing |

---

## Failure Criteria

A run should be classified as **failed** if any of the following occur:
- Ray kills workers due to memory pressure
- CUDA OOM appears
- the process hangs without useful progress
- requested concurrency is not actually realized and cannot be stabilized
- result artifacts are missing, empty, or NaN due to dropped workers

A run should be classified as **borderline** if:
- it completes but with repeated severe scheduler pressure,
- available RAM drops too close to the kill threshold,
- or throughput becomes poor enough to make the setting unattractive.

A run should be classified as **safe** only if:
- it completes cleanly,
- actual concurrency is realized,
- no worker-kill / OOM warnings appear,
- and throughput remains operationally reasonable.

---

## Expected Deliverables

1. A run matrix with all attempted concurrency levels
2. A resource summary table for each run
3. A final classification:
   - safe
   - borderline
   - unsafe
4. A recommended production setting for `full` mode
5. Notes on which resource becomes the true bottleneck

---

## Questions We Must Answer at the End

### Primary conclusion questions
1. What is the **highest safe concurrency** for `data_loading=full` on this node?
2. What is the **highest practical concurrency** that is both stable and fast enough to use?
3. Are the safe and practical values the same?

### Bottleneck questions
4. What fails first as concurrency rises: **host RAM**, **GPU VRAM**, or **Ray scheduling**?
5. Does lowering GPU fraction actually help, or does it only let Ray oversubscribe the device?
6. Is host RAM the dominant limit across all failed runs?

### Operational questions
7. Does the runner actually achieve the requested concurrency at each level?
8. At what point do scheduler warnings become operationally unacceptable?
9. Does the best concurrency remain stable long enough to train until early stopping?

### Configuration questions
10. Is `feature_prop_device=cpu` or `feature_prop_device=cuda` better for stable full-mode execution here?
11. How much do `use_amp` and `bfloat16` matter for safe concurrency?
12. Should the production config use the same settings as the sweep, or does it need a more conservative confirmation setting?

### Recommendation question
13. What exact production configuration should we recommend for the full `ogbn-arxiv` run?

---

## Proposed Decision Format

The final report should summarize the outcome in plain terms, e.g.:

- **Safe:** `N = X`
- **Borderline:** `N = Y`
- **Unsafe:** `N >= Z`
- **Recommended production setting:** `max_concurrent_clients = X`, `gpu_fraction = ...`
- **Primary limiting resource:** host RAM / GPU VRAM / scheduling

---

## Immediate Next Step

Before implementation, prepare:
- one dedicated concurrency-test config template,
- one monitoring/logging procedure,
- one results table template,
- and one exact run order from `N=2` upward.

That keeps the sweep reproducible and makes the final recommendation defendable.
