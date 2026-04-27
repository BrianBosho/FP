# FedProp Experiment Runner Adapter — Implementation Spec
**File:** `/home/bosho/FP/experiments/scheduler/` (create this directory)
**Target agent:** OpenCode or any IDE agent
**Purpose:** Wire the general `experiment_runner` package to the FedProp codebase

---

## 0. What You Are Building

Four Python files that connect the general experiment runner
(`/home/bosho/davout/experiment_runner/`) to the FedProp project at
`/home/bosho/FP/`. The general runner knows nothing about FedProp — these
four files are the only bridge.

```
/home/bosho/FP/experiments/scheduler/
  __init__.py          ← empty, makes it a package
  queue_builder.py     ← reads CHECKLIST.md → writes queue.json
  launcher_adapter.py  ← job dict → shell command + grouping key
  result_parser.py     ← reads FedProp result dirs → ResultPacket dict
  validity_rules.py    ← defines what "valid" means scientifically
```

Do NOT modify anything in `/home/bosho/davout/experiment_runner/`.
Do NOT modify anything in `/home/bosho/FP/src/` or `experiments/run.py`.

---

## 1. Before Writing Any Code — Discover These Facts

Open these files and record what you find. The spec below has `[DISCOVER]`
markers where the answer depends on the actual codebase.

### 1a. Run command
```bash
cd /home/bosho/FP
python experiments/run.py --help
```
Record every CLI flag. Pay attention to:
- How `--config` / `--track` / `--result` are specified
- Whether `--seed` exists as a flag
- Whether propagation method is a flag or comes from the config
- Whether beta is a flag or comes from the config

### 1b. Result directory structure
```bash
ls /home/bosho/FP/experiments/results/R1/
ls /home/bosho/FP/experiments/results/R1/Cora_adjacency_GCN_beta1_clients10/
```
Record:
- The exact directory naming convention
- Every file inside a completed result directory
- Which file contains `val_accuracy` (or equivalent metric)
- Whether it is JSON, CSV, or plain text

### 1c. Config structure
```bash
ls /home/bosho/FP/experiments/configs/R1/
cat /home/bosho/FP/experiments/configs/R1/R1_cora.yaml
```
Record what fields are in a config YAML and what the run command needs
to override at runtime (propagation, beta, hops, seed).

### 1d. Conda environment
```bash
conda env list
cat /home/bosho/FP/environment.yml  # or requirements.txt
```
Record the exact conda environment name used for FedProp.
It will be needed in `launcher_adapter.py`.

### 1e. Existing results (for result_parser calibration)
```bash
find /home/bosho/FP/experiments/results -name "*.json" | head -5
# or
find /home/bosho/FP/experiments/results -name "*.csv" | head -5
```
Open one completed result file. Record the exact keys and value ranges.

---

## 2. queue_builder.py

### Purpose
Reads the experiment checklist or config directory and generates a complete
`queue.json` with all atomic jobs for all tracks.

### Interface
```python
def build_queue(source: Path, output: Path) -> None:
    """
    Args:
        source: Path to /home/bosho/FP/experiments/  (the experiments dir)
        output: Path to write queue.json
    Writes a valid queue.json with all jobs.
    """
```

### Job matrix to generate

**Track R1 — GCN, L=1 and L=2**

For each combination of:
- dataset: `cora`, `citeseer`, `pubmed`, `ogbn_arxiv`
- hops: `1`, `2`
- propagation: `zero_hop`, `adjacency`, `diffusion`
- beta: `1`, `10`, `10000`

→ 4 datasets × 2 hops × 3 propagations × 3 betas = **72 jobs**

Config path pattern:
- L=1: `experiments/configs/R1/R1_{dataset}.yaml`
- L=2: `experiments/configs/R1/R1_{dataset}_2hop.yaml`

Result dir pattern: [DISCOVER from 1b — use the same naming convention as
existing results, e.g. `Cora_adjacency_GCN_beta1_clients10`]

**Track R1b — GAT, L=1 and L=2** (same matrix, model=GAT)
→ 72 jobs. Depends on R1.

**Tracks R4, R5, R6** — generate from `experiments/configs/R4/`, `R5/`, `R6/`
if those directories exist. Each config file = one job group.

**Grand total: ~218 jobs**

### job_id convention
```
{track}_{dataset}_{model}_{propagation}_beta{beta}_hops{hops}_seed{seed}

Examples:
  R1_cora_GCN_adjacency_beta1_hops1_seed42
  R1b_citeseer_GAT_diffusion_beta10000_hops2_seed42
```

### queue.json root structure
```json
{
  "version": "3.0",
  "project": "fedprop",
  "created_at": "<ISO timestamp>",
  "launch_policy": {
    "gpu_free_gb_min": 5,
    "cpu_load_max": 0.75,
    "ram_free_gb_min": 8,
    "framework_actors_max": 12,
    "launch_cooldown_seconds": 90,
    "max_parallel_jobs": 2
  },
  "jobs": [ ... ]
}
```

`framework_actors_max: 12` limits Ray actor oversubscription.
`max_parallel_jobs: 2` is conservative for the L40-48Q GPU.
Both can be tuned by editing queue.json without code changes.

### Per-job object
```json
{
  "job_id": "R1_cora_GCN_adjacency_beta1_hops1_seed42",
  "job_type": "experiment",
  "track": "R1",
  "dataset": "cora",
  "model": "GCN",
  "propagation": "adjacency",
  "hops": 1,
  "beta": 1,
  "n_clients": 10,
  "seed": 42,
  "config_path": "experiments/configs/R1/R1_cora.yaml",
  "result_dir": "experiments/results/R1/Cora_adjacency_GCN_beta1_clients10",
  "expected_metric": "val_accuracy",
  "valid_range": [0.0, 1.0],
  "expected_min": 0.40,
  "execution_status": "pending",
  "result_status": "unassessed",
  "priority": 1,
  "blocking_impact": null,
  "estimated_minutes": 15,
  "dependencies": [],
  "current_attempt": null,
  "retry_count": 0,
  "max_retries": 2,
  "doom_loop_count": 0,
  "run_pid": null,
  "started_at": null,
  "finished_at": null,
  "exit_code": null,
  "failure_reason": null,
  "escalation_ticket": null,
  "linear_issue_id": null
}
```

**Notes on specific fields:**
- `expected_min`: 0.40 is a sanity floor (anything below is clearly wrong).
  Set to 0.0 if unsure — validity_rules.py provides a more nuanced check.
- `estimated_minutes`: 15 for Cora/Citeseer L=1, 20 for Pubmed, 35 for
  OGBN-Arxiv, double these for L=2 configs.
- `result_dir`: [DISCOVER from 1b] — use the exact naming pattern from
  existing completed result directories.
- `seed`: Use 42 as the default single seed unless the run.py supports
  multiple seeds natively. If it does, generate one job per seed.

### Priority assignment
- R1 jobs: priority 1–72 (in track order: Cora L1 first, ogbn L2 last)
- R1b jobs: priority 73–144 (depends on R1 all done)
- R4/R5/R6: priority 145+

R1b jobs should have `dependencies: ["R1_complete"]` if the scheduler
supports track-level dependencies, otherwise leave empty and let the
outer loop handle it.

### CLI for queue_builder
```bash
python experiments/scheduler/queue_builder.py \
  --source experiments/ \
  --output experiments/scheduler/queue.json \
  --tracks R1 R1b R4 R5 R6   # optional filter
  --dry-run                   # print job count without writing
```

---

## 3. launcher_adapter.py

### Purpose
Converts a job dict into the exact shell command to run that job, and
defines the grouping key used for ETA estimation.

### Interface
```python
def build_command(job: dict, attempt_dir: Path) -> list[str]:
    """Return the shell command as a list of strings."""

def grouping_key(job: dict) -> tuple:
    """Return tuple identifying jobs that share runtime characteristics."""
```

### build_command implementation

**Step 1: Discover the exact run command**

From your `--help` output in step 1a, the command will be something like:

```bash
# Pattern A: config + overrides
conda run -n <ENV> python experiments/run.py \
  --config experiments/configs/R1/R1_cora.yaml \
  --propagation adjacency \
  --beta 1 \
  --hops 1 \
  --seed 42 \
  --result-dir experiments/results/R1/Cora_adjacency_GCN_beta1_clients10

# Pattern B: track + result flags
conda run -n <ENV> python experiments/run.py \
  --track fedprop \
  --result R1 \
  --dataset cora \
  --propagation adjacency \
  --beta 1
```

[DISCOVER which pattern matches `run.py --help`]

**Step 2: Template**
```python
def build_command(job: dict, attempt_dir: Path) -> list[str]:
    env_name = "fedprop"  # [DISCOVER: exact conda env name from step 1d]
    fp_root = Path("/home/bosho/FP")

    cmd = [
        "conda", "run", "-n", env_name, "--no-capture-output",
        "python", str(fp_root / "experiments/run.py"),
    ]

    # [DISCOVER: use the flag pattern from run.py --help]
    cmd += ["--config", str(fp_root / job["config_path"])]

    if job.get("propagation"):
        cmd += ["--propagation", job["propagation"]]
    if job.get("beta") is not None:
        cmd += ["--beta", str(job["beta"])]
    if job.get("hops") is not None:
        cmd += ["--hops", str(job["hops"])]  # only if flag exists
    if job.get("seed") is not None:
        cmd += ["--seed", str(job["seed"])]  # only if flag exists
    if job.get("result_dir"):
        cmd += ["--result-dir", str(fp_root / job["result_dir"])]

    return cmd
```

**Important:** `--no-capture-output` on `conda run` is required so that
Ray/Flower output is not swallowed. If `conda run` does not support this
flag on your version, use `subprocess` with `conda activate` instead.

**Alternative if conda run is unreliable:**
```python
# Use bash -c with conda activate
cmd = [
    "bash", "-c",
    f"source $(conda info --base)/etc/profile.d/conda.sh && "
    f"conda activate {env_name} && "
    f"python {fp_root}/experiments/run.py {' '.join(run_args)}"
]
```

### grouping_key implementation
```python
def grouping_key(job: dict) -> tuple:
    """Jobs with same dataset+model+hops share runtime characteristics."""
    return (
        job.get("dataset", "unknown"),
        job.get("model", "GCN"),
        job.get("hops", 1),
    )
```

This means the scheduler will use observed Cora L=1 runtimes to estimate
all other Cora L=1 jobs, etc. The ETA will improve as more jobs complete.

---

## 4. result_parser.py

### Purpose
Reads the output files written by FedProp into a normalized ResultPacket dict.

### Interface
```python
def parse(job: dict, attempt_dir: Path) -> dict:
    """
    Read result files from job["result_dir"].
    Return a ResultPacket dict.
    Raises FileNotFoundError if expected files are absent.
    """
```

### ResultPacket structure (what you must return)
```python
{
    "job_id": job["job_id"],
    "primary_metric": "val_accuracy",
    "primary_value": 0.762,       # float, the actual metric value
    "anomalies_detected": False,   # True if NaN, inf, or suspicious values
    "raw": { ... }                 # optional: full metrics dict for debugging
}
```

### How to implement parse()

**Step 1: Discover result file format** (from step 1b/1e)

FedProp writes results to `job["result_dir"]`. From the conversation history,
completed result dirs look like:
```
experiments/results/R1/Cora_adjacency_GCN_beta1_clients10/
```

[DISCOVER: look inside a completed result dir and record every file name
and its format. Common patterns in federated GNN experiments:]

```
# Pattern A: metrics.json
{
  "val_accuracy": 0.762,
  "train_loss": 0.21,
  "test_accuracy": 0.751
}

# Pattern B: results.csv
round,val_accuracy,train_loss
89,0.762,0.21

# Pattern C: multiple files per client
client_0_metrics.json
client_1_metrics.json
...
aggregated_metrics.json   ← use this one
```

**Step 2: Template (fill in after discovering format)**
```python
import json
import math
from pathlib import Path


class ResultMissing(FileNotFoundError):
    pass


def parse(job: dict, attempt_dir: Path) -> dict:
    result_dir = Path(job["result_dir"])

    # [DISCOVER: replace with the actual file name and format]
    # Common option A: metrics.json
    metrics_file = result_dir / "metrics.json"
    if not metrics_file.exists():
        raise ResultMissing(
            f"metrics.json not found at {metrics_file}. "
            f"Result dir contents: {list(result_dir.iterdir()) if result_dir.exists() else 'dir missing'}"
        )

    metrics = json.loads(metrics_file.read_text())

    # [DISCOVER: use the correct key from the actual metrics file]
    val_acc = metrics.get("val_accuracy") or metrics.get("test_accuracy")

    # Anomaly detection
    anomaly = False
    if val_acc is None:
        anomaly = True
    elif isinstance(val_acc, float) and (math.isnan(val_acc) or math.isinf(val_acc)):
        anomaly = True
    elif val_acc < 0.0 or val_acc > 1.0:
        anomaly = True

    return {
        "job_id": job["job_id"],
        "primary_metric": "val_accuracy",
        "primary_value": val_acc,
        "anomalies_detected": anomaly,
        "raw": metrics,
    }
```

**Note on the result_dir path:** The path in `job["result_dir"]` is relative
to `/home/bosho/FP/`. In `parse()`, either:
- Use `Path("/home/bosho/FP") / job["result_dir"]` if it's a relative path, OR
- Use `Path(job["result_dir"])` if it's an absolute path.

Check how `queue_builder.py` writes it and be consistent.

---

## 5. validity_rules.py

### Purpose
Defines what "valid" means scientifically for a FedProp result.
A job is only marked `result_status: valid` if ALL rules pass.

### Interface
```python
def is_valid(result: dict, job: dict) -> tuple[bool, str]:
    """
    Args:
        result: The ResultPacket dict returned by parse()
        job:    The job dict from queue.json
    Returns:
        (True, "ok") if valid
        (False, "reason string") if invalid
    """
```

### Rules to implement (in order — fail fast)

**Rule 1: No anomalies from parser**
```python
if result.get("anomalies_detected"):
    return False, "anomaly detected by parser (NaN/inf/out-of-range)"
```

**Rule 2: Primary value exists and is a float**
```python
val = result.get("primary_value")
if val is None:
    return False, "primary_value is None — metric not found in result file"
if not isinstance(val, (int, float)):
    return False, f"primary_value is not numeric: {val!r}"
```

**Rule 3: NaN / inf check (redundant but explicit)**
```python
import math
if math.isnan(val) or math.isinf(val):
    return False, f"primary_value is {val} (NaN or inf)"
```

**Rule 4: Within valid range**
```python
lo, hi = job.get("valid_range", [0.0, 1.0])
if not (lo <= val <= hi):
    return False, f"primary_value {val:.4f} outside valid_range [{lo}, {hi}]"
```

**Rule 5: Above sanity floor**
```python
expected_min = job.get("expected_min", 0.0)
if val < expected_min:
    return False, (
        f"primary_value {val:.4f} below expected_min {expected_min} "
        f"for {job.get('dataset')} {job.get('propagation')} beta={job.get('beta')}"
    )
```

**Rule 6: Diffusion NaN flag (FedProp-specific)**

From the session history, diffusion propagation had a known NaN bug.
Add an explicit check:
```python
if job.get("propagation") == "diffusion" and val < 0.1:
    return False, (
        f"diffusion result {val:.4f} suspiciously low — "
        "possible unresolved diffusion NaN bug"
    )
```
This is a soft floor, not a hard block. Adjust the 0.1 threshold based on
what reasonable diffusion results look like once the bug is fixed.

**Rule 7: Result directory actually exists**
```python
result_dir = Path(job.get("result_dir", ""))
if not result_dir.exists():
    return False, f"result_dir does not exist: {result_dir}"
```

**Full implementation:**
```python
import math
from pathlib import Path


def is_valid(result: dict, job: dict) -> tuple[bool, str]:
    # Rule 1
    if result.get("anomalies_detected"):
        return False, "anomaly detected by parser (NaN/inf/out-of-range)"

    # Rule 2
    val = result.get("primary_value")
    if val is None:
        return False, "primary_value is None"
    if not isinstance(val, (int, float)):
        return False, f"primary_value is not numeric: {val!r}"

    # Rule 3
    if math.isnan(float(val)) or math.isinf(float(val)):
        return False, f"primary_value is {val}"

    val = float(val)

    # Rule 4
    lo, hi = job.get("valid_range", [0.0, 1.0])
    if not (lo <= val <= hi):
        return False, f"val_accuracy {val:.4f} outside [{lo}, {hi}]"

    # Rule 5
    expected_min = job.get("expected_min", 0.0)
    if val < expected_min:
        return False, (
            f"val_accuracy {val:.4f} below floor {expected_min} "
            f"({job.get('dataset')} {job.get('propagation')} β={job.get('beta')})"
        )

    # Rule 6 — diffusion soft floor
    if job.get("propagation") == "diffusion" and val < 0.1:
        return False, f"diffusion val_accuracy {val:.4f} suspiciously low"

    return True, "ok"
```

---

## 6. __init__.py

Empty file. Just create it:
```python
# FedProp experiment runner adapter
```

---

## 7. Cron Setup

After the four adapter files are written and verified, set up the cron:

```bash
# Edit crontab
crontab -e

# Add these three lines:
# Monitor: every 1 minute
* * * * * cd /home/bosho/FP && conda run -n <ENV> python -m experiment_runner.monitor --write experiments/scheduler/resource_state.json >> experiments/scheduler/logs/monitor.log 2>&1

# Scheduler inner loop: every 5 minutes
*/5 * * * * cd /home/bosho/FP && conda run -n <ENV> python -m experiment_runner.scheduler --queue experiments/scheduler/queue.json --status experiments/scheduler/status.json --log experiments/scheduler/run_log.jsonl --runs-dir experiments/scheduler/runs --resource-state experiments/scheduler/resource_state.json --adapter experiments/scheduler/launcher_adapter.py >> experiments/scheduler/logs/scheduler.log 2>&1

# Outer loop: every 30 minutes
*/30 * * * * cd /home/bosho/FP && conda run -n <ENV> python -m experiment_runner.outer_loop --queue experiments/scheduler/queue.json --status experiments/scheduler/status.json --log experiments/scheduler/run_log.jsonl --anomaly-output experiments/scheduler/ANOMALY.md >> experiments/scheduler/logs/outer_loop.log 2>&1
```

Create the log directory:
```bash
mkdir -p /home/bosho/FP/experiments/scheduler/logs
```

Make sure `/home/bosho/davout` is on the Python path so `experiment_runner`
is importable:
```bash
# Add to ~/.bashrc or set in cron with PYTHONPATH=
export PYTHONPATH=/home/bosho/davout:$PYTHONPATH
```

---

## 8. Verification Steps

Run these after implementing, in order:

### Step 1: Build the queue (dry run)
```bash
cd /home/bosho/FP
python experiments/scheduler/queue_builder.py \
  --source experiments/ \
  --output /tmp/test_queue.json \
  --dry-run
# Expected: prints job count, should be ~218
```

### Step 2: Validate queue structure
```bash
python -c "
import json
from pathlib import Path
q = json.loads(Path('/tmp/test_queue.json').read_text())
print(f'Jobs: {len(q[\"jobs\"])}')
print(f'Tracks: {set(j[\"track\"] for j in q[\"jobs\"])}')
print(f'Sample job: {json.dumps(q[\"jobs\"][0], indent=2)}')
"
```

### Step 3: Test adapter dry-run
```bash
python -m experiment_runner.scheduler \
  --queue /tmp/test_queue.json \
  --status /tmp/test_status.json \
  --log /tmp/test_log.jsonl \
  --runs-dir /tmp/test_runs \
  --adapter experiments/scheduler/launcher_adapter.py \
  --dry-run
# Expected: JSON showing next_job and gate_reason
```

### Step 4: Test result parser on an existing result
```bash
python -c "
import sys; sys.path.insert(0, '/home/bosho/FP')
from experiments.scheduler.result_parser import parse
from experiments.scheduler.validity_rules import is_valid
import json
from pathlib import Path

# Use an existing completed result directory
job = {
    'job_id': 'test',
    'result_dir': 'experiments/results/R1/Cora_adjacency_GCN_beta1_clients10',
    'expected_metric': 'val_accuracy',
    'valid_range': [0.0, 1.0],
    'expected_min': 0.4,
    'propagation': 'adjacency',
}
result = parse(job, Path('/tmp'))
print('Result:', result)
valid, reason = is_valid(result, job)
print(f'Valid: {valid}, Reason: {reason}')
"
```

### Step 5: Launch one real job manually
```bash
python -m experiment_runner.scheduler \
  --queue experiments/scheduler/queue.json \
  --status experiments/scheduler/status.json \
  --log experiments/scheduler/run_log.jsonl \
  --runs-dir experiments/scheduler/runs \
  --resource-state experiments/scheduler/resource_state.json \
  --adapter experiments/scheduler/launcher_adapter.py
# Expected: {"action": "launched", "pid": <N>, ...}
```

Watch it:
```bash
tail -f experiments/scheduler/runs/<job_id>/attempt_001/stdout.log
```

After it finishes, run scheduler again — it should detect completion:
```bash
# Run again after ~15 min
python -m experiment_runner.scheduler ...
# Expected: {"action": "launched"} for next job, and first job marked done/valid
```

---

## 9. Open Questions for Brian to Answer First

The IDE agent cannot answer these — they require your knowledge of the
codebase:

| # | Question | Where to look | Blocks |
|---|----------|--------------|--------|
| Q1 | Exact conda env name | `conda env list` | launcher_adapter.py |
| Q2 | Does `run.py` take `--seed`? | `python experiments/run.py --help` | queue_builder.py seed handling |
| Q3 | Does `run.py` take `--hops` or is it in the config? | `run.py --help` | launcher_adapter.py |
| Q4 | What files does a completed result dir contain? | `ls results/R1/Cora_*/` | result_parser.py |
| Q5 | Is the metric key `val_accuracy` or something else? | Open a result file | result_parser.py + validity_rules.py |
| Q6 | Is the diffusion NaN bug fixed? | Check git log or run a diffusion config | validity_rules.py rule 6 |

If Q4 and Q5 can't be answered before implementation, implement
`result_parser.py` with a fallback that tries multiple common file/key
names and raises a clear error if none match.

---

## 10. .gitignore additions

Add to `/home/bosho/FP/.gitignore`:
```
experiments/scheduler/status.json
experiments/scheduler/resource_state.json
experiments/scheduler/STATUS.md
experiments/scheduler/runs/*/attempt_*/stdout.log
experiments/scheduler/runs/*/attempt_*/stderr.log
experiments/scheduler/runs/*/attempt_*/heartbeat.json
experiments/scheduler/runs/*/attempt_*/config_snapshot.yaml
experiments/scheduler/logs/
```

Commit these:
```
experiments/scheduler/queue.json
experiments/scheduler/run_log.jsonl
experiments/scheduler/ANOMALY.md       (when written)
experiments/scheduler/ALERTS.md        (when written)
experiments/scheduler/STUCK.md         (when written)
experiments/scheduler/runs/*/attempt_*/result.json     (after valid)
experiments/scheduler/runs/*/attempt_*/repair_patch.diff (if written)
experiments/scheduler/reports/FINAL_REPORT.md
experiments/scheduler/reports/results_*.csv
```
