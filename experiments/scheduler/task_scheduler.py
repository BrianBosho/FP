#!/usr/bin/env python3
"""FedProp Task-Aware Scheduler with Linear Integration.

Reads v4 queue (tasks + jobs), checks Linear priorities, launches jobs,
and updates Linear with job checklists and task status.

Usage:
    python task_scheduler.py --queue queue.json --status task_status.json \
                             --linear-status linear_state.json \
                             --resource-state resource_state.json \
                             --runs-dir runs/ --once

Cron (every 5 min):
    */5 * * * * cd /home/bosho/FP && /home/bosho/.conda/envs/fedgnn/bin/python \
        experiments/scheduler/task_scheduler.py --once \
        --queue experiments/scheduler/queue.json \
        --status experiments/scheduler/task_status.json \
        --linear-status experiments/scheduler/linear_state.json \
        --resource-state experiments/scheduler/resource_state.json \
        --runs-dir experiments/scheduler/runs \
        --log experiments/scheduler/scheduler.log
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

FP_ROOT = Path("/home/bosho/FP")
FEDGNN_PYTHON = "/home/bosho/.conda/envs/fedgnn/bin/python"
LINEAR_API_KEY = None

def _load_api_key() -> str:
    global LINEAR_API_KEY
    if LINEAR_API_KEY:
        return LINEAR_API_KEY
    env_path = Path("/home/bosho/davout/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("LINEAR_API_KEY="):
                LINEAR_API_KEY = line.split("=", 1)[1].strip()
                return LINEAR_API_KEY
    return os.environ.get("LINEAR_API_KEY", "")


def linear_gql(query: str) -> dict:
    api_key = _load_api_key()
    if not api_key:
        return {"error": "LINEAR_API_KEY not configured"}
    resp = requests.post(
        "https://api.linear.app/graphql",
        headers={
            "Content-Type": "application/json",
            "Authorization": api_key,
        },
        json={"query": query},
        timeout=30,
    )
    return resp.json()


def get_team_id(team_key: str = "BOS") -> Optional[str]:
    result = linear_gql(f'{{ teams(filter: {{ key: {{ eq: "{team_key}" }} }}) {{ nodes {{ id }} }} }}')
    nodes = result.get("data", {}).get("teams", {}).get("nodes", [])
    return nodes[0]["id"] if nodes else None


def get_workflow_state_id(team_key: str, state_name: str) -> Optional[str]:
    team_id = get_team_id(team_key)
    if not team_id:
        return None
    result = linear_gql(
        f'{{ workflowStates(filter: {{ team: {{ id: {{ eq: "{team_id}" }} }}, '
        f'name: {{ eq: "{state_name}" }} }}) {{ nodes {{ id name }} }} }}'
    )
    nodes = result.get("data", {}).get("workflowStates", {}).get("nodes", [])
    return nodes[0]["id"] if nodes else None


def get_issue_uuid(identifier: str) -> Optional[str]:
    team_key = identifier.split("-")[0]
    num = identifier.split("-")[1]
    result = linear_gql(
        f'{{ issues(filter: {{ number: {{ eq: {num} }}, '
        f'team: {{ key: {{ eq: "{team_key}" }} }} }}) {{ nodes {{ id title }} }} }}'
    )
    nodes = result.get("data", {}).get("issues", {}).get("nodes", [])
    return nodes[0]["id"] if nodes else None


def update_issue_state(identifier: str, state_name: str) -> bool:
    uuid = get_issue_uuid(identifier)
    if not uuid:
        return False
    team_key = identifier.split("-")[0]
    state_id = get_workflow_state_id(team_key, state_name)
    if not state_id:
        return False
    result = linear_gql(
        f'mutation {{ issueUpdate(id: "{uuid}", input: {{ stateId: "{state_id}" }}) '
        f'{{ success issue {{ identifier state {{ name }} }} }} }}'
    )
    return result.get("data", {}).get("issueUpdate", {}).get("success", False)


def add_comment(identifier: str, body: str) -> bool:
    uuid = get_issue_uuid(identifier)
    if not uuid:
        return False
    body_escaped = body.replace('"', '\\"').replace("\n", "\\n")
    result = linear_gql(
        f'mutation {{ commentCreate(input: {{ issueId: "{uuid}", '
        f'body: "{body_escaped}" }}) {{ success comment {{ id }} }} }}'
    )
    return result.get("data", {}).get("commentCreate", {}).get("success", False)


def update_issue_description(identifier: str, description: str) -> bool:
    uuid = get_issue_uuid(identifier)
    if not uuid:
        return False
    desc_escaped = description.replace('"', '\\"').replace("\n", "\\n")
    result = linear_gql(
        f'mutation {{ issueUpdate(id: "{uuid}", input: {{ '
        f'description: "{desc_escaped}" }}) {{ success issue {{ identifier }} }} }}'
    )
    return result.get("data", {}).get("issueUpdate", {}).get("success", False)


# ── Resource checks ──────────────────────────────────────────────────────────

def check_resources(resource_state_path: Path, launch_policy: dict | None = None) -> Tuple[bool, str]:
    """Check if resources are available for launching a new job.
    Returns (ok, reason)."""
    try:
        data = json.loads(resource_state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False, "resource state unreadable"

    policy = launch_policy or {}
    cpu_load_max = policy.get("cpu_load_max", 1.2)
    gpu_free_min = policy.get("gpu_free_gb_min", 3)
    ram_free_min = policy.get("ram_free_gb_min", 4)

    cpu_load = data.get("cpu_load_ratio_1m", data.get("cpu_load_1m", 0))
    # Handle both raw load and ratio formats
    if cpu_load > 10:  # Raw load value (e.g., 14.65 on 24 cores)
        cpu_ratio = cpu_load / data.get("cpu_cores", 24)
    else:
        cpu_ratio = cpu_load
    gpu_free = data.get("gpu_free_gb", 0)
    ram_free = data.get("ram_free_gb", 0)

    if cpu_ratio > cpu_load_max:
        return False, f"CPU load too high ({cpu_ratio:.2f} > {cpu_load_max})"
    if gpu_free < gpu_free_min:
        return False, f"GPU memory too low ({gpu_free:.1f} GB)"
    if ram_free < ram_free_min:
        return False, f"RAM too low ({ram_free:.1f} GB)"

    return True, "ok"


def count_running_jobs(runs_dir: Path) -> int:
    """Count jobs currently running in runs directory."""
    if not runs_dir.exists():
        return 0
    count = 0
    for task_dir in runs_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for attempt_dir in task_dir.iterdir():
            heartbeat = attempt_dir / "heartbeat.json"
            if heartbeat.exists():
                try:
                    data = json.loads(heartbeat.read_text())
                    if data.get("status") == "running":
                        count += 1
                except (json.JSONDecodeError, KeyError):
                    pass
    return count


# ── GPU memory-aware slot calculation ────────────────────────────────────────

def get_gpu_total_mb() -> float:
    """Query total VRAM in MB from nvidia-smi. Falls back to L40 constant."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return 47700.0  # L40-48Q fallback


def get_per_process_gpu_mb() -> Dict[int, float]:
    """Return {pid: vram_mb} for all GPU compute processes via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        out: Dict[int, float] = {}
        for line in r.stdout.strip().splitlines():
            parts = line.strip().split(", ")
            if len(parts) == 2:
                pid, mb = int(parts[0]), float(parts[1])
                out[pid] = out.get(pid, 0.0) + mb
        return out
    except Exception:
        return {}


def get_running_job_info(runs_dir: Path, queue: dict) -> List[dict]:
    """Return list of {job_id, pid, dataset} for running jobs."""
    result = []
    if not runs_dir.exists():
        return result
    
    # Map job_id to dataset for quick lookup
    job_to_dataset = {j["job_id"]: j["dataset"] for j in queue.get("jobs", [])}

    for task_dir in runs_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for attempt_dir in task_dir.iterdir():
            hb = attempt_dir / "heartbeat.json"
            if not hb.exists():
                continue
            try:
                data = json.loads(hb.read_text())
                if data.get("status") == "running" and data.get("pid"):
                    job_id = attempt_dir.name.replace("attempt_", "")
                    result.append({
                        "job_id": job_id,
                        "pid": int(data["pid"]),
                        "dataset": job_to_dataset.get(job_id, "unknown")
                    })
            except Exception:
                pass
    return result


def compute_launch_slots(
    resource_state: dict,
    runs_dir: Path,
    launch_policy: dict,
    queue: dict,
    pending_jobs: List[dict],
) -> Tuple[int, str]:
    """
    Compute how many new jobs to launch based on per-dataset memory awareness.
    """
    gpu_free_gb = resource_state.get("gpu_free_gb", 0.0)
    gpu_total_gb = get_gpu_total_mb() / 1024.0
    
    profile_path = Path("experiments/scheduler/memory_profile.json")
    try:
        mem_profile = json.loads(profile_path.read_text())
    except Exception:
        mem_profile = {"default": 2.0}

    per_proc_mb = get_per_process_gpu_mb()
    running_jobs = get_running_job_info(runs_dir, queue)
    running_count = len(running_jobs)

    # 1. Update Profile with Live Data
    if running_count > 0:
        dataset_usage = {} # dataset -> list of GB
        for job in running_jobs:
            pid = job["pid"]
            ds = job["dataset"]
            if pid in per_proc_mb:
                usage_gb = per_proc_mb[pid] / 1024.0
                if ds not in dataset_usage: dataset_usage[ds] = []
                dataset_usage[ds].append(usage_gb)
        
        # Smooth update (moving average 0.3)
        updated = False
        for ds, usages in dataset_usage.items():
            avg_usage = sum(usages) / len(usages)
            if avg_usage > 0.1: # ignore tiny idle processes
                current = mem_profile.get(ds, mem_profile.get("default", 2.0))
                mem_profile[ds] = round(current * 0.7 + avg_usage * 0.3, 2)
                updated = True
        
        if updated:
            try:
                profile_path.write_text(json.dumps(mem_profile, indent=2))
            except Exception: pass

    # 2. Estimate cost of NEXT job
    if not pending_jobs:
        return 0, "no pending jobs"
    
    next_job = pending_jobs[0]
    next_ds = next_job.get("dataset", "default")
    # Handle Amazon naming convention mismatch if necessary
    if next_ds == "Computers": next_ds = "Amazon_Computers"
    if next_ds == "Photo": next_ds = "Amazon_Photo"
    
    cost_gb = mem_profile.get(next_ds, mem_profile.get("default", 2.0))

    # 3. Calculate slots
    headroom_gb = max(
        launch_policy.get("gpu_headroom_gb", 3.0),
        gpu_total_gb * 0.06,
    )
    usable_gb = max(0.0, gpu_free_gb - headroom_gb)
    
    # How many of NEXT_JOB can we fit?
    slots = int(usable_gb / cost_gb) if cost_gb > 0 else 0

    max_parallel = launch_policy.get("max_parallel_jobs", 40)
    slots = max(0, min(slots, max_parallel - running_count))

    reason = (
        f"free={gpu_free_gb:.1f}GB headroom={headroom_gb:.1f}GB "
        f"next_job={next_ds} cost={cost_gb:.2f}GB slots={slots}"
    )
    return slots, reason


# ── Queue / Task logic ───────────────────────────────────────────────────────

def load_queue(path: Path) -> dict:
    return json.loads(path.read_text())


def _job_to_condition_key(job: dict) -> str:
    """Convert a scheduler job dict to a RunLedger condition_key."""
    try:
        from src.fedgnn.experiments.ledger import make_condition_key

        def _as_bool(value) -> bool:
            if isinstance(value, str):
                return value.lower() in {"true", "1", "yes", "on"}
            return bool(value)

        seed = job.get("experiment_seed", job.get("seed"))
        if seed is None and job.get("config_path"):
            try:
                import yaml
                config_path = FP_ROOT / job["config_path"]
                config = yaml.safe_load(config_path.read_text()) or {}
                seed = config.get("experiment_seed")
            except Exception:
                seed = None

        return make_condition_key(
            dataset=str(job.get("dataset", "")),
            data_loading=str(job.get("propagation", "")),
            model=str(job.get("model", "")),
            beta=float(job.get("beta", 1.0)),
            clients=int(job.get("n_clients", 0)),
            hop=int(job.get("hops", 1)),
            use_pe=_as_bool(job.get("use_pe", False)),
            seed=seed,
        )
    except Exception:
        return ""


def _ledger_completed_keys(runs_dir: Path) -> set:
    """Load completed condition_keys from run_ledger.jsonl files under *runs_dir*."""
    completed: set = set()
    try:
        from src.fedgnn.experiments.ledger import RunLedger

        ledger_files = set()
        for root in (runs_dir, FP_ROOT / "results_summary"):
            if root.exists():
                ledger_files |= set(root.rglob("run_ledger.jsonl"))

        for ledger_file in ledger_files:
            try:
                ledger = RunLedger(ledger_file.parent)
                completed |= ledger.completed_condition_keys()
            except Exception:
                pass
    except Exception:
        pass
    return completed


def get_pending_jobs(queue: dict, status: dict, runs_dir: Optional[Path] = None) -> List[dict]:
    """Return all pending jobs sorted by priority.

    Checks both the status JSON and (when *runs_dir* is provided) the durable
    run ledger so that jobs completed in a prior sweep are not re-launched even
    if the status file was not updated.
    """
    status_jobs = status.get("jobs", {})
    ledger_done: set = _ledger_completed_keys(runs_dir) if runs_dir else set()

    pending = []
    for job in queue.get("jobs", []):
        job_id = job["job_id"]
        job_status = status_jobs.get(job_id, {}).get("execution_status", "pending")
        if job_status != "pending":
            continue
        # Secondary check: ledger says this condition is already done
        if ledger_done:
            ckey = _job_to_condition_key(job)
            if ckey and ckey in ledger_done:
                continue
        pending.append(job)
    pending.sort(key=lambda j: j["priority"])
    return pending


def get_task_for_job(queue: dict, job_id: str) -> Optional[dict]:
    """Find the task that contains this job."""
    for task in queue.get("tasks", []):
        for child in task.get("children", []):
            if job_id in child.get("job_ids", []):
                return task
    return None


def build_job_checklist(task: dict, status: dict) -> str:
    """Build markdown checklist of jobs grouped by config (not individual seeds)."""
    lines = [f"## Job Checklist (by Config)"]
    status_jobs = status.get("jobs", {})

    for child in task.get("children", []):
        propagation = child["display_name"]
        lines.append(f"\n### {propagation} ({child['job_count']} jobs)")
        
        # Group by config (beta, hops) instead of individual seeds
        from collections import defaultdict
        config_groups = defaultdict(lambda: {"completed": 0, "running": 0, "failed": 0, "pending": 0, "total": 0})
        
        for job_id in child.get("job_ids", []):
            job_status = status_jobs.get(job_id, {}).get("execution_status", "pending")
            # Extract config: R1_cora_GCN_zero_hop_beta1_hops1_seed42 → beta=1, hops=1
            parts = job_id.split("_")
            beta = "?"
            hops = "?"
            for i, p in enumerate(parts):
                if p.startswith("beta"):
                    beta = p[4:]  # Remove 'beta' prefix
                if p.startswith("hops"):
                    hops = p[4:]  # Remove 'hops' prefix
            
            config_key = f"beta={beta}, hops={hops}"
            config_groups[config_key]["total"] += 1
            if job_status in ("completed", "done"):
                config_groups[config_key]["completed"] += 1
            elif job_status == "running":
                config_groups[config_key]["running"] += 1
            elif job_status == "failed":
                config_groups[config_key]["failed"] += 1
            else:
                config_groups[config_key]["pending"] += 1
        
        for config_key, counts in sorted(config_groups.items()):
            if counts["completed"] == counts["total"]:
                symbol = "[x]"
                status_text = "done"
            elif counts["running"] > 0:
                symbol = "[~]"
                status_text = f"running ({counts['running']} jobs)"
            elif counts["failed"] > 0:
                symbol = "[!]"
                status_text = f"failed ({counts['failed']} jobs)"
            else:
                symbol = "[ ]"
                status_text = "pending"
            
            lines.append(f"- {symbol} `{config_key}` — {counts['completed']}/{counts['total']} {status_text}")

    return "\n".join(lines)


def compute_task_progress(task: dict, status: dict) -> Tuple[int, int, int]:
    """Return (completed, running, total) for a task."""
    status_jobs = status.get("jobs", {})
    completed = 0
    running = 0
    total = 0
    for child in task.get("children", []):
        for job_id in child.get("job_ids", []):
            total += 1
            job_status = status_jobs.get(job_id, {}).get("execution_status", "pending")
            if job_status in ("completed", "done"):
                completed += 1
            elif job_status == "running":
                running += 1
    return completed, running, total


# ── Job launching ────────────────────────────────────────────────────────────

def launch_job(job: dict, runs_dir: Path) -> Tuple[bool, Optional[int], str]:
    """Launch a single job. Returns (success, pid, error_msg)."""
    task_id = job.get("task_group_id", "unknown")
    job_id = job["job_id"]
    # Use unique attempt dir per job to avoid conflicts when launching multiple jobs
    attempt_dir = runs_dir / task_id / f"attempt_{job_id}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    config_path = FP_ROOT / job["config_path"] if job.get("config_path") else None
    if not config_path or not config_path.exists():
        # Fallback: find config from track
        config_path = _find_config_file(job)
        if not config_path:
            return False, None, f"No config found for {job_id}"

    # Pass the track dir — run_experiments appends its own subdir name
    results_base = str(FP_ROOT / job["result_dir"]).rsplit("/", 1)[0]
    cmd = [
        FEDGNN_PYTHON,
        "-m", "src.experiments.run_experiments",
        "--config", str(config_path),
        "--datasets", job["dataset"],
        "--data_loading", job["propagation"],
        "--beta", str(job["beta"]),
        "--models", job["model"],
        "--clients", str(job["n_clients"]),
        "--results_dir", results_base,
        "--save_results",
    ]

    if job.get("hops") is not None:
        cmd += ["--hop", str(job["hops"])]

    if job.get("use_pe") is not None:
        cmd += ["--use_pe", str(job["use_pe"]).lower()]

    # Write config snapshot
    snapshot = attempt_dir / "config_snapshot.yaml"
    snapshot.write_text(f"# Auto-generated for {job_id}\n")

    # Launch
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=(attempt_dir / "stdout.log").open("w"),
            stderr=(attempt_dir / "stderr.log").open("w"),
            cwd=str(FP_ROOT),
        )
        # Write heartbeat
        heartbeat = attempt_dir / "heartbeat.json"
        heartbeat.write_text(json.dumps({
            "status": "running",
            "pid": proc.pid,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }))
        return True, proc.pid, ""
    except Exception as e:
        return False, None, str(e)


def _find_config_file(job: dict) -> Optional[Path]:
    """Find config file for a job."""
    track = job["track"]
    dataset = job["dataset"]
    model = job["model"]
    hop = job.get("hops", 1)
    propagation = job["propagation"]

    configs_dir = FP_ROOT / "experiments" / "configs" / track
    if not configs_dir.exists():
        return None

    candidates = []
    for p in configs_dir.glob("*.yaml"):
        if p.name == "base.yaml":
            continue
        try:
            import yaml
            config = yaml.safe_load(p.read_text()) or {}
        except Exception:
            continue
        if dataset not in config.get("datasets", []):
            continue
        if model not in config.get("models", ["GCN"]):
            continue
        if propagation not in config.get("data_loading", []):
            continue
        config_hop = 2 if p.stem.endswith("_2hop") else int(config.get("hop", 1))
        if config_hop != hop:
            continue
        epochs = config.get("epochs", 1)
        candidates.append((p, epochs))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1] != 3, x[0].name))
    return candidates[0][0]


# ── Main scheduler loop ──────────────────────────────────────────────────────

def tick(args: argparse.Namespace) -> None:
    queue_path = Path(args.queue)
    status_path = Path(args.status)
    linear_status_path = Path(args.linear_status)
    resource_state_path = Path(args.resource_state)
    runs_dir = Path(args.runs_dir)

    # Load queue and status
    queue = load_queue(queue_path)
    status = json.loads(status_path.read_text()) if status_path.exists() else {"jobs": {}}
    linear_state = json.loads(linear_status_path.read_text()) if linear_status_path.exists() else {}

    # Load task_id -> "BOS-XX" mapping
    issue_map_path = queue_path.parent / "linear_issue_map.json"
    issue_map = json.loads(issue_map_path.read_text()) if issue_map_path.exists() else {}

    # Check for pre-flight completion before running main queue
    if queue_path.name == "queue.json":
        preflight_queue_path = queue_path.parent / "queue_with_preflight.json"
        preflight_status_path = queue_path.parent / "preflight_status.json"
        if preflight_queue_path.exists():
            # Check if pre-flight is complete
            if preflight_status_path.exists():
                preflight_status = json.loads(preflight_status_path.read_text())
                preflight_jobs = preflight_status.get("jobs", {})
                all_done = all(j.get("execution_status") in ("completed", "done") for j in preflight_jobs.values())
                if not all_done:
                    print("[INFO] Pre-flight check not complete. Skipping main queue.")
                    return
            else:
                print("[INFO] Pre-flight check pending. Skipping main queue.")
                return

    # Hard resource gate — minimum thresholds must pass before we launch anything
    _launch_policy = queue.get("launch_policy", {})
    try:
        resource_state = json.loads(resource_state_path.read_text())
    except Exception:
        resource_state = {}
    launch_ok, gate_reason = check_resources(resource_state_path, _launch_policy)

    # Memory-aware slot calculation: how many jobs can we add right now?
    pending = get_pending_jobs(queue, status, runs_dir=runs_dir)
    slots, slot_reason = compute_launch_slots(resource_state, runs_dir, _launch_policy, queue, pending)
    running_count = len(get_running_job_info(runs_dir, queue))
    max_parallel = _launch_policy.get("max_parallel_jobs", 40)

    log_line = (
        f"[{datetime.now(timezone.utc).isoformat()}] "
        f"running={running_count}/{max_parallel} gate={'ok' if launch_ok else gate_reason} "
        f"{slot_reason}"
    )

    launched_count = 0
    launch_errors = []

    if launch_ok and slots > 0:
        while launched_count < slots:
            # Re-read resource state each iteration — GPU changes as jobs start
            try:
                resource_state = json.loads(resource_state_path.read_text())
            except Exception:
                break
            launch_ok, gate_reason = check_resources(resource_state_path, _launch_policy)
            if not launch_ok:
                log_line += f" stopped={gate_reason}"
                break

            # Re-fetch pending to account for the job we just launched
            pending = get_pending_jobs(queue, status, runs_dir=runs_dir)
            if not pending:
                break

            next_job = pending[0]
            success, pid, error = launch_job(next_job, runs_dir)
            if success:
                launched_count += 1
                job_id = next_job["job_id"]
                status["jobs"][job_id] = {
                    "job_id": job_id,
                    "execution_status": "running",
                    "result_status": "unassessed",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "run_pid": pid,
                    "retry_count": status["jobs"].get(job_id, {}).get("retry_count", 0),
                    "doom_loop_count": 0,
                }
                
                # Stabilization sleep between launches
                if launched_count < slots:
                    time.sleep(10)
            else:
                launch_errors.append(f"{next_job['job_id']}: {error}")
                break

    if launched_count > 0:
        log_line += f" launched={launched_count}"
    if launch_errors:
        log_line += f" errors={'; '.join(launch_errors)}"

    # Surface blocked jobs to Linear (one comment per job, once)
    for job in queue.get("jobs", []):
        job_id = job["job_id"]
        job_status_entry = status.get("jobs", {}).get(job_id, {})
        if job_status_entry.get("execution_status") == "blocked":
            if not job_status_entry.get("linear_notified"):
                task = get_task_for_job(queue, job_id)
                if task:
                    linear_id = issue_map.get(task["task_id"])
                    if linear_id:
                        reason = job_status_entry.get("failure_reason", "unknown")
                        retries = job_status_entry.get("retry_count", 0)
                        add_comment(linear_id,
                            f"BLOCKED: `{job_id}` failed after {retries} retries "
                            f"(reason: {reason}). Needs manual review.")
                status["jobs"][job_id]["linear_notified"] = True

    # Update completed jobs and task statuses
    for task in queue.get("tasks", []):
        completed, running, total = compute_task_progress(task, status)
        task_id = task["task_id"]
        linear_id = issue_map.get(task_id)
        task_state = linear_state.get(task_id, {})

        if not linear_id:
            continue

        if completed == total and total > 0 and task_state.get("linear_status") != "Done":
            if update_issue_state(linear_id, "Done"):
                task_state["linear_status"] = "Done"
                linear_state[task_id] = task_state
                add_comment(linear_id, f"Task complete! All {total} jobs finished.")

        # Update description periodically when in progress
        if task_state.get("linear_status") == "In Progress":
            checklist = build_job_checklist(task, status)
            desc = f"{task['display_name']}\n\nProgress: {completed}/{total} completed, {running} running\n\n{checklist}"
            update_issue_description(linear_id, desc)

    # Save state
    status_path.write_text(json.dumps(status, indent=2))
    linear_status_path.write_text(json.dumps(linear_state, indent=2))

    # Log
    if args.log:
        with open(args.log, "a") as f:
            f.write(log_line + "\n")
    else:
        print(log_line)


def main():
    parser = argparse.ArgumentParser(description="FedProp Task-Aware Scheduler")
    parser.add_argument("--queue", default="experiments/scheduler/queue.json")
    parser.add_argument("--status", default="experiments/scheduler/task_status.json")
    parser.add_argument("--linear-status", default="experiments/scheduler/linear_state.json")
    parser.add_argument("--resource-state", default="experiments/scheduler/resource_state.json")
    parser.add_argument("--runs-dir", default="experiments/scheduler/runs")
    parser.add_argument("--log", default=None)
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval in seconds")
    args = parser.parse_args()

    if args.once:
        tick(args)
    else:
        while True:
            tick(args)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
