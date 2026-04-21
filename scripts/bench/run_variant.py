#!/usr/bin/env python3
"""Run one FL experiment *variant* over N seeds and capture everything.

A "variant" is just a (base_config, overrides) pair: e.g. "baseline" runs
``conf/cora_minimal.yaml`` as-is; "fedavg_weighted" runs the same config with
``--override aggregation=fedavg_weighted``.  Each seed is executed in a fresh
Python subprocess so Ray/CUDA state can't leak between runs.

Outputs land under::

    bench/runs/<variant>_<timestamp>/
        manifest.json     # git sha, env, seeds, config, per-seed status
        runs.csv          # one row per (seed, experiment-combination)
        config_used.yaml  # the base config after overrides were merged in
        seed_<N>/
            config.yaml   # the exact config file passed to run_experiments
            stdout.log
            stderr.log
            <run_experiments output tree>

Nothing in ``src/`` is touched.  This is purely an orchestration / capture
layer, so it stays compatible with whatever the current training code does.
"""
from __future__ import annotations

import argparse
import copy
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml  # comes with pyyaml that's already a transitive dep

# Make "import scripts.bench.common" work without installing as a package.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import common  # type: ignore  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent.parent  # .../federated-gnn
CONF_DIR = REPO_ROOT / "conf"
BENCH_CONF_DIR = CONF_DIR / "bench"
BENCH_RUNS_DIR = REPO_ROOT / "bench" / "runs"


def parse_seed_list(raw: str) -> List[int]:
    """Accepts '0,1,2' or '0 1 2' or '0-4' (inclusive)."""
    raw = raw.strip()
    if "-" in raw and "," not in raw and " " not in raw:
        lo, hi = raw.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    parts = [p for chunk in raw.split(",") for p in chunk.split()]
    return [int(p) for p in parts if p]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def build_seed_config(
    base_cfg: Dict[str, Any],
    overrides: Dict[str, Any],
    seed: int,
    results_dir: Path,
) -> Dict[str, Any]:
    """Return a merged config dict for a single seed.

    Forced values:
      * ``results_dir`` -> per-seed directory (isolates every run)
      * ``repetitions`` -> 1 (cross-seed variance is driven by separate
        processes, not the inner repetition loop)
      * ``experiment_seed`` -> seed (future-proof: read by fixes that plumb
        real seeding through partitioning / training).

    We preserve any existing wandb/memory/model keys from the base config.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.update(overrides or {})
    cfg["results_dir"] = str(results_dir)
    cfg["repetitions"] = 1
    # Non-breaking: the current training code ignores this; future fix reads it.
    cfg["experiment_seed"] = int(seed)
    return cfg


def run_one_seed(
    seed: int,
    seed_cfg: Dict[str, Any],
    variant_run_dir: Path,
    variant: str,
    dry_run: bool,
) -> Dict[str, Any]:
    seed_dir = variant_run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Write the config file the subprocess will load.  Placing it under
    # conf/bench/... is important: load_config() walks up to find conf/base.yaml
    # for merging.  We keep a copy inside the seed_dir for easy lookup.
    temp_cfg_path = BENCH_CONF_DIR / f"{variant_run_dir.name}_seed_{seed}.yaml"
    dump_yaml(temp_cfg_path, seed_cfg)
    dump_yaml(seed_dir / "config.yaml", seed_cfg)

    cmd = [sys.executable, "-m", "src.experiments.run_experiments",
           "--config", str(temp_cfg_path)]

    record: Dict[str, Any] = {
        "seed": seed,
        "seed_dir": str(seed_dir),
        "config_path": str(temp_cfg_path),
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "status": "skipped" if dry_run else None,
        "returncode": None,
        "duration_s": None,
        "started_at": None,
        "finished_at": None,
    }
    if dry_run:
        return record

    env = os.environ.copy()
    # These are consumed by run_utils.save_results_to_csv / monkey-patch in
    # run_experiments.py.  They will be overwritten per-experiment inside
    # run_experiments anyway; we set them here as a safe default.
    env.setdefault("EXPERIMENT_RESULTS_DIR", str(seed_dir))
    env.setdefault("EXPERIMENT_TIMESTAMP", common.now_stamp())
    # Keep any fix behind its own env flag easy to set via overrides later.
    env["PYTHONHASHSEED"] = str(seed)

    stdout_path = seed_dir / "stdout.log"
    stderr_path = seed_dir / "stderr.log"
    started = time.time()
    record["started_at"] = common.now_stamp()
    try:
        with stdout_path.open("w") as so, stderr_path.open("w") as se:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=so,
                stderr=se,
                check=False,
            )
        record["returncode"] = proc.returncode
        record["status"] = "ok" if proc.returncode == 0 else "failed"
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = repr(exc)
    finally:
        record["duration_s"] = round(time.time() - started, 3)
        record["finished_at"] = common.now_stamp()

    return record


def collect_runs_csv(variant_run_dir: Path, variant: str) -> Path:
    rows = common.collect_variant_rows(variant_run_dir, variant)
    # Drop the big 'rounds' list before writing CSV; the round data is still
    # available from the raw results_*.json paths.
    flat: List[Dict[str, Any]] = []
    for r in rows:
        r2 = {k: v for k, v in r.items() if k != "rounds"}
        flat.append(r2)
    csv_path = variant_run_dir / "runs.csv"
    common.write_csv(csv_path, flat)
    return csv_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", required=True, type=Path,
                    help="Path to the YAML config to extend (e.g. conf/cora_minimal.yaml).")
    ap.add_argument("--variant", required=True,
                    help="Short label for this variant (e.g. 'baseline', 'fedavg_weighted').")
    ap.add_argument("--seeds", default="0",
                    help="Seed list: '0,1,2' or '0-4'.  Default: '0'.")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VAL",
                    help="Extra config overrides.  Repeatable.  Values parsed as YAML.")
    ap.add_argument("--tag", default="",
                    help="Optional tag appended to the run directory name.")
    ap.add_argument("--output-root", type=Path, default=BENCH_RUNS_DIR,
                    help=f"Root output directory (default: {BENCH_RUNS_DIR}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write the manifest + per-seed configs, but skip execution.")
    args = ap.parse_args()

    base_cfg_path: Path = args.base_config.resolve()
    if not base_cfg_path.exists():
        print(f"ERROR: base config not found: {base_cfg_path}", file=sys.stderr)
        return 2
    base_cfg = load_yaml(base_cfg_path)
    overrides = common.parse_overrides(args.override)
    seeds = parse_seed_list(args.seeds)

    stamp = common.now_stamp()
    run_name = f"{args.variant}_{stamp}"
    if args.tag:
        run_name += f"_{args.tag}"
    variant_run_dir: Path = (args.output_root / run_name).resolve()
    variant_run_dir.mkdir(parents=True, exist_ok=True)
    BENCH_CONF_DIR.mkdir(parents=True, exist_ok=True)

    # Freeze the merged "starting point" config for easy inspection.
    dump_yaml(variant_run_dir / "config_used.yaml",
              {**base_cfg, **overrides, "repetitions": 1})

    manifest: Dict[str, Any] = {
        "variant": args.variant,
        "tag": args.tag,
        "run_dir": str(variant_run_dir),
        "base_config": str(base_cfg_path),
        "overrides": overrides,
        "seeds": seeds,
        "started_at": common.now_stamp(),
        "finished_at": None,
        "git": common.git_commit(REPO_ROOT),
        "env": common.env_probe(),
        "cli": " ".join(shlex.quote(a) for a in sys.argv),
        "per_seed": [],
        "dry_run": bool(args.dry_run),
    }

    print(f"[bench] variant={args.variant}  seeds={seeds}  output={variant_run_dir}")
    for seed in seeds:
        seed_dir = variant_run_dir / f"seed_{seed}"
        seed_cfg = build_seed_config(base_cfg, overrides, seed, seed_dir)
        print(f"[bench]   -> seed={seed}  dir={seed_dir}")
        rec = run_one_seed(seed, seed_cfg, variant_run_dir, args.variant, args.dry_run)
        manifest["per_seed"].append(rec)
        status_tag = f"[{rec['status']}]"
        dur = rec.get("duration_s")
        dur_str = f"{dur:.1f}s" if isinstance(dur, (int, float)) else "-"
        print(f"[bench]      {status_tag} returncode={rec.get('returncode')} duration={dur_str}")

    manifest["finished_at"] = common.now_stamp()
    common.dump_json(variant_run_dir / "manifest.json", manifest)

    csv_path = collect_runs_csv(variant_run_dir, args.variant)
    print(f"[bench] flat results table -> {csv_path}")
    print(f"[bench] manifest           -> {variant_run_dir / 'manifest.json'}")

    failed = [r for r in manifest["per_seed"] if r.get("status") not in ("ok", "skipped")]
    if failed:
        print(f"[bench] WARNING: {len(failed)} seed(s) did not finish cleanly.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
