#!/usr/bin/env python3
"""Run a focused pilot through the real FedProp experiment runner."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "pilots" / "R1_cora_quick.yaml"
DEFAULT_PYTHON = "/home/bosho/.conda/envs/fedgnn/bin/python"


def latest_summary(results_dir: Path) -> Path | None:
    summary_dir = REPO_ROOT / "results_summary" / results_dir.name
    summaries = sorted(summary_dir.glob("summary_results_*.json"))
    return summaries[-1] if summaries else None


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "src.experiments.run_experiments",
        "--config",
        str(args.config),
        "--results_dir",
        args.results_dir,
    ]

    if args.datasets:
        cmd += ["--datasets", *args.datasets]
    if args.data_loading:
        cmd += ["--data_loading", *args.data_loading]
    if args.models:
        cmd += ["--models", *args.models]
    if args.beta:
        cmd += ["--beta", *[str(x) for x in args.beta]]
    if args.clients:
        cmd += ["--clients", *[str(x) for x in args.clients]]
    if args.rounds is not None:
        cmd += ["--rounds", str(args.rounds)]
    if args.repetitions is not None:
        cmd += ["--repetitions", str(args.repetitions)]
    if args.save_results:
        cmd += ["--save_results"]

    return cmd


def print_summary(summary_path: Path) -> bool:
    data = json.loads(summary_path.read_text())
    rows = data.get("results", [])
    print(f"\nPilot summary: {summary_path}")
    if not rows:
        print("  No result rows found.")
        return False

    ok = True
    for row in rows:
        dataset = row.get("dataset")
        method = row.get("data_loading")
        clients = row.get("clients")
        beta = row.get("beta")
        global_acc = row.get("avg_global")
        client_acc = row.get("avg_client")
        if not (
            isinstance(global_acc, (int, float))
            and isinstance(client_acc, (int, float))
            and math.isfinite(global_acc)
            and math.isfinite(client_acc)
        ):
            ok = False
        print(
            f"  {dataset:8s} {method:10s} beta={beta} clients={clients}: "
            f"global={global_acc:.4f} client={client_acc:.4f}"
        )
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small Cora pilot without modifying the locked Track A matrix."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--results-dir", default="experiments/results/pilot/R1_cora_quick")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--data-loading", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--beta", type=float, nargs="+", default=None)
    parser.add_argument("--clients", type=int, nargs="+", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument("--save-results", dest="save_results", action="store_true")
    save_group.add_argument("--no-save-results", dest="save_results", action="store_false")
    parser.set_defaults(save_results=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.config = args.config.resolve()
    if not args.config.exists():
        print(f"Pilot config not found: {args.config}", file=sys.stderr)
        return 1

    cmd = build_command(args)
    print("Pilot command:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        return result.returncode

    summary = latest_summary((REPO_ROOT / args.results_dir).resolve())
    if summary is not None:
        if not print_summary(summary):
            print("\nPilot completed with invalid metric values.")
            return 2
    else:
        print("\nPilot completed, but no summary JSON was found.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
