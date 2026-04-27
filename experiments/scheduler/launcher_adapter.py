#!/usr/bin/env python3
"""Launcher adapter: job dict -> shell command + grouping key.

Also re-exports parse() and is_valid() from result_parser and validity_rules
so the scheduler can use a single adapter object.
"""

from importlib import import_module
from pathlib import Path
from typing import List, Tuple

# Re-export parse/is_valid so the scheduler can use --adapter as the single
# entry point for both build_command and result assessment.
result_parser = import_module('experiments.scheduler.result_parser')
validity_rules = import_module('experiments.scheduler.validity_rules')
parse = result_parser.parse
is_valid = validity_rules.is_valid

FP_ROOT = Path("/home/bosho/FP")
FEDGNN_PYTHON = "/home/bosho/.conda/envs/fedgnn/bin/python"


def build_command(job: dict, attempt_dir: Path) -> List[str]:
    """Return the shell command as a list of strings.

    Discovered CLI pattern:
      experiments/run.py does NOT accept atomic overrides.
      The underlying module src.experiments.run_experiments DOES:
        --config, --datasets, --data_loading, --beta, --models,
        --clients, --hop, --results_dir, --save_results
      No --seed flag exists; seeds are driven by experiment_seed + repetitions
      in the YAML config.
    """
    cmd = [
        FEDGNN_PYTHON,
        "-m", "src.experiments.run_experiments",
        "--config", str(FP_ROOT / job["config_path"]),
        "--datasets", job["dataset"],
        "--data_loading", job["propagation"],
        "--beta", str(job["beta"]),
        "--models", job["model"],
        "--clients", str(job["n_clients"]),
        "--results_dir", str(FP_ROOT / f"experiments/results/{job['track']}"),
        "--save_results",
    ]

    if job.get("hops") is not None:
        cmd += ["--hop", str(job["hops"])]

    cmd += ["--use_pe", "true" if job.get("use_pe") else "false"]

    return cmd


def grouping_key(job: dict) -> tuple:
    """Jobs with same dataset+model+hops share runtime characteristics."""
    return (
        job.get("dataset", "unknown"),
        job.get("model", "GCN"),
        job.get("hops", 1),
    )
