#!/usr/bin/env python3
"""Build queue.json from FedProp experiment configs.

Queue layout supports two levels:
- leaf experiment jobs in ``jobs``
- grouped task containers in ``task_groups`` organized by dataset -> model -> dataloading
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _dataset_key(dataset: str) -> str:
    """Lowercase and replace hyphens for job_id safety."""
    return dataset.lower().replace("-", "_")


def _extract_hops(config_path: Path, config: dict) -> int:
    """Determine hop count from filename suffix or config field."""
    if config_path.stem.endswith("_2hop"):
        return 2
    return int(config.get("hop", 1))


def _estimated_minutes(dataset: str, hops: int) -> int:
    """Runtime estimates based on observed durations."""
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


def _build_job(
    track: str,
    config_path: Path,
    config: dict,
    dataset: str,
    model: str,
    propagation: str,
    beta,
    n_clients: int,
    hops: int,
    seed: int,
    priority: int,
    n_local_steps: int = 1,
    repetition_index: int = 1,
    base_seed: int = None,
) -> dict:
    """Construct a single job dict."""
    # Result directory follows the exact naming convention used by
    # src.fedgnn.experiments.run_experiments.setup_environment_for_experiment
    experiment_name = f"{dataset}_{propagation}_{model}_beta{beta}_clients{n_clients}"
    result_dir = f"experiments/results/{track}/{experiment_name}"

    job_id = (
        f"{track}_{_dataset_key(dataset)}_{model}_{propagation}_"
        f"beta{beta}_hops{hops}_seed{seed}"
    )
    if n_local_steps != 1:
        job_id += f"_loc{n_local_steps}"
    if n_clients != 10:
        job_id += f"_nk{n_clients}"

    # Config path relative to repo root
    config_rel = str(config_path.relative_to(Path("/home/bosho/FP")))

    task_group_id = f"{_dataset_key(dataset)}__{model}"
    subtask_group_id = f"{task_group_id}__{propagation}"

    return {
        "job_id": job_id,
        "job_type": "experiment",
        "track": track,
        "dataset": dataset,
        "model": model,
        "propagation": propagation,
        "hops": hops,
        "beta": beta,
        "n_clients": n_clients,
        "seed": seed,
        "base_seed": base_seed if base_seed is not None else seed,
        "repetition_index": repetition_index,
        "config_path": config_rel,
        "result_dir": result_dir,
        "expected_metric": "average_client_result",
        "valid_range": [0.0, 1.0],
        "expected_min": 0.40,
        "execution_status": "pending",
        "result_status": "unassessed",
        "priority": priority,
        "blocking_impact": None,
        "estimated_minutes": _estimated_minutes(dataset, hops),
        "dependencies": [],
        "current_attempt": None,
        "retry_count": 0,
        "max_retries": 2,
        "doom_loop_count": 0,
        "n_local_steps": n_local_steps,
        "run_pid": None,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "failure_reason": None,
        "escalation_ticket": None,
        "linear_issue_id": None,
        "task_group_id": task_group_id,
        "subtask_group_id": subtask_group_id,
    }


def _generate_track_jobs(track: str, configs_dir: Path, start_priority: int) -> Tuple[List[dict], int]:
    """Generate all jobs for a single track. Returns (jobs, next_priority)."""
    jobs = []
    priority = start_priority

    if not configs_dir.exists():
        return jobs, priority

    for config_path in sorted(configs_dir.glob("*.yaml")):
        if config_path.name == "base.yaml":
            continue

        config = _load_config(config_path)
        hops = _extract_hops(config_path, config)
        seed = int(config.get("experiment_seed", 42))
        n_local_steps = int(config.get("epochs", 1))
        repetitions = int(config.get("repetitions", 1))

        datasets = config.get("datasets", [])
        models = config.get("models", ["GCN"])
        propagations = list(config.get("data_loading", []))
        betas = config.get("beta", [])
        n_clients_list = config.get("num_clients", [10])

        if track in {"R1", "R1b", "R4", "R6"} and "full" not in propagations:
            propagations.append("full")

        for dataset in datasets:
            for model in models:
                for propagation in propagations:
                    for beta in betas:
                        for n_clients in n_clients_list:
                            for repetition_index in range(1, repetitions + 1):
                                run_seed = seed + (repetition_index - 1)
                                job = _build_job(
                                    track=track,
                                    config_path=config_path,
                                    config=config,
                                    dataset=dataset,
                                    model=model,
                                    propagation=propagation,
                                    beta=beta,
                                    n_clients=n_clients,
                                    hops=hops,
                                    seed=run_seed,
                                    priority=priority,
                                    n_local_steps=n_local_steps,
                                    repetition_index=repetition_index,
                                    base_seed=seed,
                                )
                                jobs.append(job)
                                priority += 1

    return jobs, priority


def _derive_group_status(statuses: List[str]) -> str:
    if any(s == "running" for s in statuses):
        return "running"
    if statuses and all(s in {"done", "completed"} for s in statuses):
        return "completed"
    if any(s == "failed" for s in statuses):
        return "failed"
    return "pending"


def _build_task_groups(all_jobs: List[dict]) -> List[dict]:
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for job in all_jobs:
        grouped[(job["dataset"], job["model"])].append(job)

    task_groups = []
    for (dataset, model), jobs in sorted(grouped.items(), key=lambda x: (x[0][0].lower(), x[0][1].lower())):
        jobs = sorted(jobs, key=lambda j: (j["priority"], j["track"], j["propagation"], j["beta"], j["hops"], j["n_clients"]))
        by_prop: Dict[str, List[dict]] = defaultdict(list)
        for job in jobs:
            by_prop[job["propagation"]].append(job)

        task_groups.append({
            "task_group_id": f"{_dataset_key(dataset)}__{model}",
            "task_type": "dataset_model",
            "dataset": dataset,
            "model": model,
            "display_name": f"{dataset} {model}",
            "priority": min(job["priority"] for job in jobs),
            "execution_status": _derive_group_status([job["execution_status"] for job in jobs]),
            "job_count": len(jobs),
            "tracks": sorted({job["track"] for job in jobs}),
            "children": [
                {
                    "subtask_group_id": f"{_dataset_key(dataset)}__{model}__{propagation}",
                    "task_type": "dataloading",
                    "propagation": propagation,
                    "display_name": propagation,
                    "priority": min(job["priority"] for job in prop_jobs),
                    "execution_status": _derive_group_status([job["execution_status"] for job in prop_jobs]),
                    "job_count": len(prop_jobs),
                    "job_ids": [job["job_id"] for job in prop_jobs],
                }
                for propagation, prop_jobs in sorted(by_prop.items(), key=lambda x: min(job["priority"] for job in x[1]))
            ],
        })

    return task_groups


def build_queue(source: Path, output: Path, tracks: Optional[List[str]] = None) -> None:
    """Read configs and write queue.json."""
    if tracks is None:
        tracks = ["R1", "R1b", "R4", "R5", "R6"]

    fp_root = Path("/home/bosho/FP")
    configs_base = fp_root / "experiments" / "configs"

    all_jobs = []
    priority = 1

    for track in tracks:
        track_jobs, priority = _generate_track_jobs(
            track, configs_base / track, priority
        )
        all_jobs.extend(track_jobs)

    queue = {
        "version": "3.1",
        "project": "fedprop",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "launch_policy": {
            "gpu_free_gb_min": 5,
            "cpu_load_max": 0.75,
            "ram_free_gb_min": 8,
            "framework_actors_max": 12,
            "launch_cooldown_seconds": 90,
            "max_parallel_jobs": 2,
        },
        "grouping": {
            "top_level": ["dataset", "model"],
            "second_level": ["propagation"],
        },
        "task_groups": _build_task_groups(all_jobs),
        "jobs": all_jobs,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build FedProp experiment queue")
    parser.add_argument("--source", type=str, default="experiments/",
                        help="Path to experiments directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write queue.json (required unless --dry-run)")
    parser.add_argument("--tracks", nargs="+", default=None,
                        help="Optional track filter (e.g. R1 R1b)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print job count without writing")
    args = parser.parse_args()

    source = Path(args.source).resolve()

    tracks = args.tracks
    if tracks is None:
        tracks = ["R1", "R1b", "R4", "R5", "R6"]

    fp_root = Path("/home/bosho/FP")
    configs_base = fp_root / "experiments" / "configs"

    all_jobs = []
    priority = 1
    for track in tracks:
        track_jobs, priority = _generate_track_jobs(
            track, configs_base / track, priority
        )
        all_jobs.extend(track_jobs)

    if args.dry_run:
        print(f"Would generate {len(all_jobs)} jobs for tracks: {tracks}")
        return

    if not args.output:
        parser.error("--output is required unless --dry-run is specified")

    output = Path(args.output).resolve()

    build_queue(source, output, tracks)
    print(f"Generated {len(all_jobs)} jobs -> {output}")


if __name__ == "__main__":
    main()
