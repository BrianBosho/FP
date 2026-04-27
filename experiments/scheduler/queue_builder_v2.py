#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build queue.json v4 with clean task -> job hierarchy.

Rules:
- local_steps (epochs) fixed at 3
- Only one config per dataset-model-track combo (no duplicate hop/epoch files)
- Tasks = track + dataset + model
- Children = dataloading groups
- Jobs = atomic runs with repetitions expanded
- Include 'full' dataloading for tracks that need it
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

FP_ROOT = Path("/home/bosho/FP")
CONFIGS_DIR = FP_ROOT / "experiments" / "configs"


def _dataset_key(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _estimated_minutes(dataset: str, hops: int) -> int:
    base = {
        "cora": 15,
        "citeseer": 15,
        "pubmed": 20,
        "ogbn-arxiv": 35,
        "ogbn_arxiv": 35,
        "texas": 20,
        "wisconsin": 20,
    }.get(dataset.lower().replace("-", "_"), 20)
    return base * hops


# Cache for config lookups
_config_cache: Dict[Tuple[str, str, str, int, str], Path] = {}

def _build_config_cache():
    """Pre-load all configs and build a lookup cache."""
    global _config_cache
    _config_cache = {}
    
    for track_dir in CONFIGS_DIR.iterdir():
        if not track_dir.is_dir():
            continue
        track = track_dir.name
        for p in track_dir.glob("*.yaml"):
            if p.name == "base.yaml":
                continue
            config = _load_config(p)
            for dataset in config.get("datasets", []):
                for model in config.get("models", ["GCN"]):
                    for dataloading in config.get("data_loading", []):
                        hop = 2 if p.stem.endswith("_2hop") else int(config.get("hop", 1))
                        epochs = config.get("epochs", 1)
                        key = (track, dataset, model, hop, dataloading)
                        # Prefer epochs=3 configs
                        if key not in _config_cache:
                            _config_cache[key] = (p, epochs)
                        else:
                            existing_path, existing_epochs = _config_cache[key]
                            if epochs == 3 and existing_epochs != 3:
                                _config_cache[key] = (p, epochs)


def _find_config(track: str, dataset: str, model: str, hop: int, dataloading: str) -> Path:
    """Find the best matching config file for this job using cache."""
    key = (track, dataset, model, hop, dataloading)
    if key in _config_cache:
        return _config_cache[key][0]
    return None


# Define the clean experiment matrix per track
# Each entry: (track, dataset, model, dataloadings, betas, hops_list, n_clients, repetitions, seed, use_pe_list)
# NOTE: 1 queue job = 1 config = all repetitions run internally in one process via run_experiments.py's repetition loop
# use_pe_list: [false, true] for R1/R1b (PE sweep), [false] for others
TRACK_DEFINITIONS = [
    # R1 — Core accuracy table (with PE sweep)  [priority 1]
    ("R1", "Cora",       "GCN",      ["zero_hop", "adjacency", "diffusion", "full"], [1, 10, 10000], [1, 2], 10, 10, 42,  [False, True]),
    ("R1", "Citeseer",   "GCN",      ["zero_hop", "adjacency", "diffusion", "full"], [1, 10, 10000], [1, 2], 10, 10, 142, [False, True]),
    ("R1", "Pubmed",     "GCN",      ["zero_hop", "adjacency", "diffusion", "full"], [1, 10, 10000], [1, 2], 10, 10, 242, [False, True]),
    ("R1", "ogbn-arxiv", "GCN_arxiv",["zero_hop", "adjacency", "diffusion", "full"], [1, 10, 10000], [1, 2], 10, 5,  342, [False, True]),

    # R1b — GAT variants (with PE sweep)  [priority 2]
    ("R1b", "Cora",     "GAT", ["zero_hop", "adjacency", "diffusion", "full"], [1, 10, 10000], [1, 2], 10, 10, 52,  [False, True]),
    ("R1b", "Citeseer", "GAT", ["zero_hop", "adjacency", "diffusion", "full"], [1, 10, 10000], [1, 2], 10, 10, 152, [False, True]),
    ("R1b", "Pubmed",   "GAT", ["zero_hop", "adjacency", "diffusion", "full"], [1, 10, 10000], [1, 2], 10, 10, 252, [False, True]),

    # R5 — Client-count scaling  [priority 3]
    ("R5", "Cora", "GCN", ["zero_hop", "adjacency", "diffusion", "full"], [1, 10, 10000], [1], [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 10, 542, [False]),

    # R4 — Beta sweep  [priority 4]
    ("R4", "Cora", "GCN", ["zero_hop", "adjacency", "diffusion", "full"], [1, 10, 100, 10000], [1], 10, 10, 442, [False]),

    # R7 and R6 excluded from current run — add back when R1/R1b/R5/R4 complete
]


def build_jobs_and_tasks() -> Tuple[List[dict], List[dict]]:
    jobs = []
    tasks = []
    priority = 1

    for track, dataset, model, dataloadings, betas, hops_list, n_clients, repetitions, seed, use_pe_list in TRACK_DEFINITIONS:
        task_id = f"{track}_{_dataset_key(dataset)}_{model}"
        task_jobs = []

        n_clients_list = n_clients if isinstance(n_clients, list) else [n_clients]

        for dataloading in dataloadings:
            prop_jobs = []
            for beta in betas:
                for hop in hops_list:
                    for n_c in n_clients_list:
                        for use_pe in use_pe_list:
                            # 1 queue job = 1 config = all reps run internally in one process
                            job_id = (
                                f"{track}_{_dataset_key(dataset)}_{model}_{dataloading}_"
                                f"beta{beta}_hops{hop}_nk{n_c}_pe{int(use_pe)}"
                            )

                            # Find matching config file
                            config_path = _find_config(track, dataset, model, hop, dataloading)
                            # Fallback: if 'full' not in config, use the config with matching dataset/model/hop
                            if config_path is None and dataloading == "full":
                                config_path = _find_config(track, dataset, model, hop, "zero_hop")
                            if config_path is None and dataloading == "full":
                                config_path = _find_config(track, dataset, model, hop, "diffusion")

                            job = {
                                "job_id": job_id,
                                "job_type": "experiment",
                                "track": track,
                                "dataset": dataset,
                                "model": model,
                                "propagation": dataloading,
                                "hops": hop,
                                "beta": beta,
                                "n_clients": n_c,
                                "seed": seed,
                                "use_pe": use_pe,
                                "repetitions": repetitions,
                                "config_path": str(config_path.relative_to(FP_ROOT)) if config_path else None,
                                "result_dir": f"experiments/results/{track}/{dataset}_{dataloading}_{model}_beta{float(beta)}_clients{n_c}_hop{hop}" + ("_pe" if use_pe else ""),
                                "expected_metric": "average_client_result",
                                "valid_range": [0.0, 1.0],
                                "expected_min": 0.40,
                                "execution_status": "pending",
                                "result_status": "unassessed",
                                "priority": priority,
                                "blocking_impact": None,
                                "estimated_minutes": _estimated_minutes(dataset, hop),
                                "dependencies": [],
                                "current_attempt": None,
                                "retry_count": 0,
                                "max_retries": 2,
                                "doom_loop_count": 0,
                                "n_local_steps": 3,
                                "run_pid": None,
                                "started_at": None,
                                "finished_at": None,
                                "exit_code": None,
                                "failure_reason": None,
                                "escalation_ticket": None,
                                "linear_issue_id": None,
                                "task_group_id": task_id,
                                "subtask_group_id": f"{task_id}_{dataloading}_pe{int(use_pe)}",
                            }
                            prop_jobs.append(job)
                            jobs.append(job)
                            priority += 1

            task_jobs.extend(prop_jobs)

        # Build task with children
        by_prop = defaultdict(list)
        for job in task_jobs:
            by_prop[job["propagation"]].append(job)

        tasks.append({
            "task_id": task_id,
            "task_type": "experiment_group",
            "track": track,
            "dataset": dataset,
            "model": model,
            "display_name": f"{track} — {dataset} {model}",
            "priority": min(job["priority"] for job in task_jobs),
            "execution_status": _derive_status([job["execution_status"] for job in task_jobs]),
            "total_jobs": len(task_jobs),
            "dataloadings": list(by_prop.keys()),
            "children": [
                {
                    "subtask_group_id": f"{task_id}_{prop}",
                    "task_type": "dataloading",
                    "propagation": prop,
                    "display_name": prop,
                    "priority": min(job["priority"] for job in prop_jobs),
                    "execution_status": _derive_status([job["execution_status"] for job in prop_jobs]),
                    "job_count": len(prop_jobs),
                    "job_ids": [job["job_id"] for job in prop_jobs],
                }
                for prop, prop_jobs in sorted(by_prop.items(), key=lambda x: min(job["priority"] for job in x[1]))
            ],
        })

    return jobs, tasks


def _derive_status(statuses: List[str]) -> str:
    if any(s == "running" for s in statuses):
        return "running"
    if statuses and all(s in {"done", "completed"} for s in statuses):
        return "completed"
    if any(s == "failed" for s in statuses):
        return "failed"
    return "pending"


def build_queue(output: Path) -> dict:
    print("Building config cache...")
    _build_config_cache()
    print(f"Cache built: {len(_config_cache)} entries")
    
    print("Building jobs and tasks...")
    jobs, tasks = build_jobs_and_tasks()

    queue = {
        "version": "4.0",
        "project": "fedprop",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "launch_policy": {
            "gpu_free_gb_min": 3,
            "cpu_load_max": 5.0,
            "ram_free_gb_min": 8,
            "framework_actors_max": 25,
            "launch_cooldown_seconds": 60,
            "max_parallel_jobs": 40,
        },
        "grouping": {
            "top_level": ["track", "dataset", "model"],
            "second_level": ["propagation"],
        },
        "tasks": tasks,
        "jobs": jobs,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, indent=2))
    return queue


def main():
    output = FP_ROOT / "experiments" / "scheduler" / "queue.json"
    queue = build_queue(output)
    print(f"Generated {len(queue['jobs'])} jobs in {len(queue['tasks'])} tasks -> {output}")


if __name__ == "__main__":
    main()
