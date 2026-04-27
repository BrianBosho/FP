#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Master runner for the split FedProp experiment plan.

Track A ("fedprop") runs experiments implemented in this repo: FedProp-Zero,
FedProp (Adj), FedProp (Diff), centralised variants, and appendix ablations.
Track B ("baselines") is a separate namespace for competitors such as FedGCN
and FedGAT. Baseline execution/import is intentionally not mixed into the
FedProp YAML runner until a baseline plan and provenance files exist.

Usage (run with the fedgnn conda environment activated):
    conda activate fedgnn
    python experiments/run.py --track fedprop --result R1 --dry-run
    python experiments/run.py --track fedprop --result R1,R1b
    python experiments/run.py --track fedprop --all
    python experiments/run.py --track baselines --baseline fedgcn --result R1
    python experiments/run.py --status
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINES_DIR = Path(__file__).resolve().parent / "baselines"

# All valid result IDs
RESULT_IDS = ["R1", "R1b", "R4", "R5", "R6", "R7"]
BASELINE_IDS = ["fedgcn", "fedgat", "fedcog"]

# Only schedule the centralised variant where the locked design needs
# a single-client full-graph baseline row.
CENTRALISED_RESULT_IDS = {"R1", "R1b", "R6"}

# Map from result_id to the subdirectory within results/
RESULT_SUBDIR = {
    "R1": "R1", "R1b": "R1b", "R4": "R4", "R5": "R5",
    "R6": "R6", "R7": "R7",
}


def _conda_python():
    """Path to python in the fedgnn conda environment."""
    return "/home/bosho/.conda/envs/fedgnn/bin/python"


def discover_configs(result_id: str) -> list[Path]:
    """Find all YAML configs for a given result ID."""
    d = CONFIGS_DIR / result_id
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.yaml") if p.name != "base.yaml")


def _list_len(config: dict, key: str, default=None) -> int:
    value = config.get(key, default)
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def atomic_run_count(config_path: Path, variant: str) -> int:
    """Estimate atomic training runs represented by one subprocess launch."""
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return 0

    repetitions = int(config.get("repetitions", 1))
    base = (
        _list_len(config, "datasets")
        * _list_len(config, "beta")
        * _list_len(config, "models")
        * _list_len(config, "use_pe", [False])
        * _list_len(config, "hop", 1)
        * repetitions
    )

    if variant == "standard":
        return (
            base
            * _list_len(config, "num_clients")
            * _list_len(config, "data_loading")
        )
    if variant == "centralised":
        return base
    if variant in {"oracle", "oracle_legacy"}:
        return base * _list_len(config, "num_clients")
    return base


def run_label(config_path: Path, variant: str = "standard") -> str:
    """Human-readable label for a run."""
    stem = config_path.stem
    if variant == "standard":
        return stem
    return f"{stem}__{variant}"


def variants_for_result(result_id: str, include_legacy_oracle: bool) -> list[str]:
    """Return default FedProp-family variants to launch for a result group.

    Individual config files may override this using a top-level
    `schedule_variants: [...]` list.
    """
    variants = ["standard"]
    if result_id in CENTRALISED_RESULT_IDS:
        variants.append("centralised")
    if include_legacy_oracle:
        variants.append("oracle_legacy")
    return variants


def variants_for_config(config_path: Path, result_id: str, include_legacy_oracle: bool) -> list[str]:
    """Determine which variants to launch for one config file."""
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        config = {}

    configured = config.get("schedule_variants", None)
    if configured is None:
        variants = variants_for_result(result_id, include_legacy_oracle)
    else:
        variants = list(configured) if isinstance(configured, (list, tuple)) else [str(configured)]
        if include_legacy_oracle and "oracle_legacy" not in variants:
            variants.append("oracle_legacy")
    return variants


def expected_labels(result_id: str, include_legacy_oracle: bool) -> set[str]:
    """Labels expected under the current scheduling policy."""
    labels = set()
    for config_path in discover_configs(result_id):
        for variant in variants_for_config(config_path, result_id, include_legacy_oracle):
            labels.add(run_label(config_path, variant))
    return labels


def planned_atomic_runs(result_id: str, include_legacy_oracle: bool) -> int:
    return sum(
        atomic_run_count(config_path, variant)
        for config_path in discover_configs(result_id)
        for variant in variants_for_config(config_path, result_id, include_legacy_oracle)
    )


def manifest_path(result_id: str) -> Path:
    return RESULTS_DIR / result_id / "manifest.json"


def load_manifest(result_id: str) -> dict:
    p = manifest_path(result_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, IOError):
            return {"runs": []}
    return {"runs": []}


def save_manifest(result_id: str, manifest: dict):
    p = manifest_path(result_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))


def experiment_output_dir(config_path: Path, variant: str) -> Path:
    """Directory where a run's JSON results are stored."""
    stem = config_path.stem  # e.g. "R1_cora"
    results_base = RESULTS_DIR / config_path.parent.name  # e.g. experiments/results/R1
    return results_base / stem


def is_completed(manifest_entry: dict) -> bool:
    """Check if a run actually produced result files.

    The manifest tracks what was launched. We verify the output directory
    contains at least one results_*.json file before treating the run as done.
    """
    if manifest_entry.get("status") != "completed":
        return False
    config_str = manifest_entry.get("config", "")
    variant = manifest_entry.get("variant", "standard")
    if not config_str:
        return False
    config_path = Path(config_str)
    out_dir = experiment_output_dir(config_path, variant)
    if not out_dir.is_dir():
        return False
    return any(p.name.startswith("results_") and p.suffix == ".json"
               for p in out_dir.iterdir())


def enumerate_atomic_runs(config_path: Path, variant: str) -> list[dict]:
    """Enumerate every atomic training run for a (config, variant) launch.

    Returns a list of atomic run descriptors with enough fields to match
    against output files and mark completion individually.
    """
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return []

    repetitions = int(config.get("repetitions", 1))
    experiment_seed = int(config.get("experiment_seed", 0))

    runs = []
    datasets = config.get("datasets", [])
    betas = config.get("beta", [])
    models = config.get("models", [])
    data_loadings = config.get("data_loading", [])
    use_pes = config.get("use_pe", [False])
    num_clients_list = config.get("num_clients", [10])
    hops = config.get("hop", [1])

    for ds in datasets:
        for beta in betas:
            for model in models:
                for use_pe in use_pes:
                    for hop in (hops if isinstance(hops, (list, tuple)) else [hops]):
                        for seed_idx in range(repetitions):
                            seed = experiment_seed + seed_idx
                            for num_clients in num_clients_list:
                                for dl in data_loadings:
                                    runs.append({
                                        "dataset": ds,
                                        "method": dl,
                                        "model": model,
                                        "beta": beta,
                                        "num_clients": num_clients,
                                        "seed": seed,
                                        "use_pe": use_pe,
                                        "hop": hop,
                                    })
    return runs


def atomic_output_exists(out_dir: Path, run: dict) -> bool:
    """Check whether a results_*.json file matches the atomic run's parameters."""
    if not out_dir.is_dir():
        return False
    for p in out_dir.iterdir():
        if not (p.name.startswith("results_") and p.suffix == ".json"):
            continue
        try:
            data = json.loads(p.read_text())
            ec = data.get("experiment_config", {})
            if (ec.get("dataset") == run["dataset"]
                    and ec.get("data_loading_option") == run["method"]
                    and ec.get("model_type") == run["model"]
                    and ec.get("beta") == run["beta"]
                    and ec.get("num_clients") == run["num_clients"]
                    and ec.get("seed") == run["seed"]
                    and ec.get("hop") == run.get("hop")):
                return True
        except (json.JSONDecodeError, OSError):
            continue
    return False


def are_all_atomics_done(out_dir: Path, atomic_runs: list[dict]) -> bool:
    """Return True only when every atomic run has its output file."""
    return all(atomic_output_exists(out_dir, r) for r in atomic_runs)


def build_command(config_path: Path, variant: str = "standard") -> list[str]:
    """Build the subprocess command for a single run."""
    cmd = [
        _conda_python(), "-m", "src.experiments.run_experiments",
        "--config", str(config_path),
    ]
    if variant == "centralised":
        cmd += ["--clients", "1", "--data_loading", "full"]
    elif variant in {"oracle", "oracle_legacy"}:
        cmd += ["--fulltraining_flag", "--data_loading", "full"]
    return cmd


def _variant_out_dir(config_path: Path, variant: str) -> Path:
    """Output directory for a (config, variant) launch."""
    stem = config_path.stem
    return RESULTS_DIR / config_path.parent.name / stem


def run_single(config_path: Path, variant: str, dry_run: bool) -> (bool, list[dict]):
    """Execute a single experiment run. Returns (success, atomic_runs)."""
    label = run_label(config_path, variant)
    cmd = build_command(config_path, variant)
    atomic_runs = enumerate_atomic_runs(config_path, variant)

    if dry_run:
        atomics = len(atomic_runs)
        print(f"  [DRY RUN] {label} ({atomics} atomic run(s))")
        print(f"            {' '.join(cmd)}")
        return True, atomic_runs

    print(f"  [{datetime.now():%H:%M:%S}] Starting: {label} ({len(atomic_runs)} atomic runs)")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=False,
            text=True,
            timeout=7200,  # 2h per run
        )
        elapsed = time.time() - t0
        ok = result.returncode == 0
        status = "completed" if ok else f"failed (rc={result.returncode})"
        print(f"  [{datetime.now():%H:%M:%S}] {status} in {elapsed:.0f}s: {label}")
        return ok, atomic_runs
    except subprocess.TimeoutExpired:
        print(f"  [{datetime.now():%H:%M:%S}] TIMEOUT after 2h: {label}")
        return False, atomic_runs
    except Exception as e:
        print(f"  [{datetime.now():%H:%M:%S}] ERROR: {e}")
        return False, atomic_runs


def run_fedprop_result(result_id: str, dry_run: bool, force_rerun: bool,
                       include_legacy_oracle: bool):
    """Run all configs for a single result ID."""
    configs = discover_configs(result_id)
    if not configs:
        print(f"No configs found for {result_id}")
        return

    manifest = load_manifest(result_id)

    # Count actually-completed runs (files verified)
    actually_done = sum(1 for r in manifest["runs"] if is_completed(r))
    variants_by_config = {
        str(config_path): variants_for_config(config_path, result_id, include_legacy_oracle)
        for config_path in configs
    }
    total = sum(len(v) for v in variants_by_config.values())
    atomic_total = planned_atomic_runs(result_id, include_legacy_oracle)
    expected = expected_labels(result_id, include_legacy_oracle)
    manifest_done = sum(
        1 for r in manifest["runs"]
        if r.get("label") in expected and r.get("status") == "completed"
    )

    print(f"\n{'='*60}")
    print(f"Track fedprop / Result {result_id}: "
          f"{len(configs)} config(s) = {total} launch(es)")
    print(f"  Atomic training runs represented: {atomic_total}")
    unique_variants = sorted({vv for variants in variants_by_config.values() for vv in variants})
    print(f"  Variants (union): {', '.join(unique_variants)}")
    if not include_legacy_oracle:
        print("  Legacy fulltraining_flag oracle is not scheduled.")
        print("  Use --include-legacy-oracle only for diagnostic label-leakage checks.")
    print(f"  Manifest: {manifest_done}/{total} historically marked, "
          f"{actually_done}/{total} file-verified")
    print(f"{'='*60}")

    for config_path in configs:
        for variant in variants_by_config[str(config_path)]:
            label = run_label(config_path, variant)

            # Check if this exact (config, variant) is in manifest and files exist
            existing = next((r for r in manifest["runs"]
                            if r["label"] == label and r.get("config") == str(config_path)), None)
            out_dir = _variant_out_dir(config_path, variant)
            atomic_runs = enumerate_atomic_runs(config_path, variant)

            if existing and existing.get("status") == "completed" and not force_rerun:
                # Re-verify all atomics are actually on disk
                if are_all_atomics_done(out_dir, atomic_runs):
                    print(f"  SKIP (launch completed, all {len(atomic_runs)} atomics verified): {label}")
                    continue
                # Files missing despite manifest entry — rerun
                print(f"  RERUN (files missing despite manifest entry): {label}")

            ok, _atomic_runs = run_single(config_path, variant, dry_run)

            if dry_run:
                continue

            launch_record = {
                "label": label,
                "config": str(config_path),
                "variant": variant,
                "status": "completed" if ok else "failed",
                "timestamp": datetime.now().isoformat(),
                "atomics_launched": len(atomic_runs),
            }

            if ok:
                # Enumerate atomics and mark which ones produced output files
                atomic_statuses = []
                for ar in atomic_runs:
                    found = atomic_output_exists(out_dir, ar)
                    atomic_statuses.append({
                        **ar,
                        "output_found": found,
                    })
                launch_record["atomic_runs"] = atomic_statuses

            if existing:
                existing.update(launch_record)
            else:
                manifest["runs"].append(launch_record)
            save_manifest(result_id, manifest)


def count_fedprop_status(result_id: str, include_legacy_oracle: bool = False) -> tuple[int, int, int, int, int]:
    """Return (n_configs, total_launches, manifest_done, file_verified, atomics)."""
    configs = discover_configs(result_id)
    total = sum(
        len(variants_for_config(config_path, result_id, include_legacy_oracle))
        for config_path in configs
    )
    atomics = planned_atomic_runs(result_id, include_legacy_oracle)
    manifest = load_manifest(result_id)
    expected = expected_labels(result_id, include_legacy_oracle)
    manifest_done = sum(
        1 for r in manifest["runs"]
        if r.get("label") in expected and r.get("status") == "completed"
    )
    file_verified = sum(
        1 for r in manifest["runs"]
        if r.get("label") in expected and is_completed(r)
    )
    return len(configs), total, manifest_done, file_verified, atomics


def run_baseline_result(baseline: str | None, result_ids: list[str], dry_run: bool):
    """Placeholder for Track B baseline execution/import."""
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    if baseline is None:
        print("Specify --baseline fedgcn or --baseline fedgat for Track B.")
        print(f"Baseline plan: {BASELINES_DIR / 'BASELINE_PLAN.md'}")
        return
    if baseline not in BASELINE_IDS:
        print(f"Unknown baseline: {baseline}")
        print(f"Valid baselines: {', '.join(BASELINE_IDS)}")
        sys.exit(1)

    print(f"\nTrack baselines / {baseline}")
    print("-" * 60)
    print("Baseline execution/import is not implemented in this runner yet.")
    print(f"Requested result IDs: {', '.join(result_ids)}")
    print(f"Workspace: {BASELINES_DIR / baseline}")
    print(f"Plan: {BASELINES_DIR / 'BASELINE_PLAN.md'}")
    if dry_run:
        print("Dry-run only; no files or manifests were changed.")


def show_status(include_legacy_oracle: bool):
    print("\nExperiment Progress")
    print("-" * 60)
    print("Track A: fedprop")
    grand_configs = 0
    grand_total = 0
    grand_verified = 0
    grand_atomics = 0
    for rid in RESULT_IDS:
        n_configs, total, _m_done, f_verified, atomics = count_fedprop_status(
            rid, include_legacy_oracle=include_legacy_oracle)
        grand_configs += n_configs
        grand_total += total
        grand_verified += f_verified
        grand_atomics += atomics
        bar = "#" * f_verified + "." * (total - f_verified) if total > 0 else ""
        print(f"  {rid:4s}  {n_configs:2d} configs  {f_verified:3d}/{total:3d} launches  "
              f"{atomics:4d} atomic  {bar}")
    print("-" * 60)
    print(f"  Total  {grand_configs:2d} configs  {grand_verified:3d}/{grand_total:3d} launches  "
          f"{grand_atomics:4d} atomic")
    print("\nTrack B: baselines")
    if not BASELINES_DIR.exists():
        print("  no baseline workspace yet")
        return
    for baseline in BASELINE_IDS:
        d = BASELINES_DIR / baseline
        n_files = sum(1 for _ in d.rglob("*")) if d.is_dir() else 0
        status = f"{n_files} file(s)" if d.is_dir() else "not created"
        print(f"  {baseline:7s}  {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Split FedProp/competitor baseline experiment runner")
    parser.add_argument("--track", choices=["fedprop", "baselines"], default="fedprop",
                        help="Execution track: repo-native FedProp runs or separate baselines")
    parser.add_argument("--result", type=str, default=None,
                        help="Result ID(s), comma-separated (e.g., R1,R1b)")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Baseline ID for --track baselines (fedgcn, fedgat, fedcog)")
    parser.add_argument("--all", action="store_true",
                        help="Run all result groups")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results already exist on disk")
    parser.add_argument("--status", action="store_true",
                        help="Show progress for all results")
    parser.add_argument("--include-legacy-oracle", action="store_true",
                        help=("Also schedule the old fulltraining_flag oracle. "
                              "This is diagnostic only and may leak labels."))
    args = parser.parse_args()

    if args.status:
        show_status(include_legacy_oracle=args.include_legacy_oracle)
        return

    if args.all:
        result_ids = RESULT_IDS
    elif args.result:
        result_ids = [r.strip() for r in args.result.split(",")]
    else:
        parser.error("Specify --result R1 or --all")

    for rid in result_ids:
        if rid not in RESULT_IDS:
            print(f"Unknown result ID: {rid}")
            print(f"Valid IDs: {', '.join(RESULT_IDS)}")
            sys.exit(1)

    if args.track == "baselines":
        run_baseline_result(args.baseline, result_ids, dry_run=args.dry_run)
        return

    for rid in result_ids:
        run_fedprop_result(
            rid,
            dry_run=args.dry_run,
            force_rerun=args.force,
            include_legacy_oracle=args.include_legacy_oracle,
        )


if __name__ == "__main__":
    main()
