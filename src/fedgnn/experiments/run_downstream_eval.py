"""
Layer 4 downstream accuracy runner.

Runs full FL training for each (operator, dataset, backbone, beta, seed)
combination and saves per-run JSON results. Gap-closed metrics are computed
post-hoc once zero-hop and oracle baseline files exist.

Usage:
    python -m src.fedgnn.experiments.run_downstream_eval \
        --config experiments/propagator_eval/configs/L4_downstream_operators.yaml \
        [--operator diffusion] [--dataset Cora] [--beta 1] [--backbone GCN] [--seed 0]
"""

import argparse
import json
import os
import time
from itertools import product

import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    base_path = os.path.join(os.path.dirname(__file__), "../../../../conf/base.yaml")
    base_path = os.path.normpath(base_path)
    cfg = {}
    if os.path.exists(base_path):
        with open(base_path) as f:
            cfg.update(yaml.safe_load(f) or {})
    with open(config_path) as f:
        cfg.update(yaml.safe_load(f) or {})
    return cfg


def _resolve_training_hp(cfg: dict, dataset: str, backbone: str) -> dict:
    """Pull per-dataset/per-backbone optimizer settings from training_per_dataset block."""
    per_ds = cfg.get("training_per_dataset", {})
    ds_cfg = per_ds.get(dataset, {})
    hp = ds_cfg.get(backbone, {})
    return {
        "lr": hp.get("optimizer", cfg.get("lr", 0.5)),   # fallback to top-level
        "optimizer": hp.get("optimizer", cfg.get("optimizer", "SGD")),
        "weight_decay": hp.get("weight_decay", cfg.get("decay", 0.0005)),
        "lr_val": hp.get("lr", cfg.get("lr", 0.5)),
    }


def _build_run_config(base_cfg: dict, operator: str, dataset: str, backbone: str,
                      beta: int, seed: int) -> dict:
    """Merge base config with per-run parameters."""
    run_cfg = dict(base_cfg)

    # Per-dataset optimizer (if training_per_dataset block exists)
    per_ds = base_cfg.get("training_per_dataset", {})
    ds_hp = per_ds.get(dataset, {}).get(backbone, {})
    if ds_hp:
        run_cfg["optimizer"] = ds_hp.get("optimizer", run_cfg.get("optimizer", "SGD"))
        run_cfg["lr"] = ds_hp.get("lr", run_cfg.get("lr", 0.5))
        run_cfg["decay"] = ds_hp.get("weight_decay", run_cfg.get("decay", 0.0005))

    run_cfg["datasets"] = [dataset]
    run_cfg["models"] = [backbone]
    run_cfg["beta"] = [beta]
    run_cfg["data_loading"] = [operator]
    run_cfg["use_pe"] = [False]
    run_cfg["repetitions"] = 1
    run_cfg["experiment_seed"] = seed
    run_cfg["save_results"] = False  # we handle output ourselves
    run_cfg["use_wandb"] = False

    return run_cfg


# ---------------------------------------------------------------------------
# Single-run logic
# ---------------------------------------------------------------------------

def run_one(operator: str, dataset: str, backbone: str, beta: int, seed: int,
            cfg: dict, results_dir: str, force: bool = False) -> dict | None:
    """Run one FL training run and save result JSON."""
    from src.fedgnn.fl.run import main_experiment

    out_path = os.path.join(
        results_dir, operator, dataset.lower(),
        f"beta{beta}_seed{seed}_{backbone.lower()}.json"
    )

    if not force and os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        test_acc = existing.get("test_accuracy")
        print(f"  skipped (exists) → {out_path}  acc={test_acc:.4f}")
        return existing

    run_cfg = _build_run_config(cfg, operator, dataset, backbone, beta, seed)

    clients_num = run_cfg.get("num_clients", 10)
    hop = run_cfg.get("hop", 1)
    fulltraining_flag = (operator == "full")

    t0 = time.perf_counter()
    try:
        result, _ = main_experiment(
            clients_num,
            beta,
            operator,
            backbone,
            run_cfg,
            dataset_name=dataset,
            hop=hop,
            fulltraining_flag=fulltraining_flag,
            manage_ray_lifecycle=True,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    elapsed = time.perf_counter() - t0

    test_acc = result.get("summary", {}).get("average_global_result", None)
    if test_acc is None:
        test_acc = result.get("test_accuracy", None)

    out = {
        "operator": operator,
        "dataset": dataset.lower(),
        "beta": beta,
        "seed": seed,
        "backbone": backbone.lower(),
        "test_accuracy": test_acc,
        "accuracy_gap_closed": None,   # computed post-hoc
        "zero_hop_accuracy": None,     # filled post-hoc from baseline files
        "oracle_accuracy": None,       # filled post-hoc from baseline files
        "per_client_accuracy": result.get("per_client_accuracy", []),
        "wall_time_sec": elapsed,
    }

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved → {out_path}  acc={test_acc:.4f}  ({elapsed:.0f}s)")

    return out


# ---------------------------------------------------------------------------
# Post-hoc gap computation
# ---------------------------------------------------------------------------

def compute_gap_closed(results_dir: str, operator: str, dataset: str,
                       beta: int, seed: int, backbone: str):
    """Fill accuracy_gap_closed in an existing result file using baseline files."""
    ds = dataset.lower()

    def _read_acc(op):
        p = os.path.join(results_dir, op, ds, f"beta{beta}_seed{seed}_{backbone.lower()}.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f).get("test_accuracy")

    acc_zero = _read_acc("zero_hop")
    acc_oracle = _read_acc("full")
    acc_op = _read_acc(operator)

    if any(v is None for v in [acc_zero, acc_oracle, acc_op]):
        return

    gap_closed = (acc_op - acc_zero) / (acc_oracle - acc_zero + 1e-12)

    path = os.path.join(results_dir, operator, ds, f"beta{beta}_seed{seed}_{backbone.lower()}.json")
    with open(path) as f:
        data = json.load(f)
    data["accuracy_gap_closed"] = gap_closed
    data["zero_hop_accuracy"] = acc_zero
    data["oracle_accuracy"] = acc_oracle
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_grid(cfg: dict, args: argparse.Namespace) -> list[tuple]:
    operators = [args.operator] if args.operator else cfg.get("operators", [])
    datasets = [args.dataset] if args.dataset else cfg.get("datasets", [])
    backbones = [args.backbone] if args.backbone else cfg.get("backbones", ["GCN"])
    betas = [args.beta] if args.beta is not None else cfg.get("beta", [10000])
    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [0])

    # Honour oracle_exclude_datasets
    excluded = set(cfg.get("oracle_exclude_datasets", []))

    grid = []
    for op, ds, bb, beta, seed in product(operators, datasets, backbones, betas, seeds):
        if op == "full" and ds in excluded:
            continue
        grid.append((op, ds, bb, beta, seed))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--operator", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--beta", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--compute-gaps", action="store_true",
                        help="After running, fill gap_closed for all completed operator files.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip grid items whose result file already exists.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result file already exists.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg.get("results_dir", "experiments/propagator_eval/results/downstream")
    grid = build_grid(cfg, args)

    print(f"Downstream eval — {len(grid)} runs")
    compute_gaps_only = args.compute_gaps
    for i, (op, ds, bb, beta, seed) in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] {op} / {ds} / {bb} / beta={beta} / seed={seed}")
        if compute_gaps_only:
            # In gap-only mode, skip the experiment run entirely — just read existing result
            out_path = os.path.join(
                results_dir, op, ds.lower(),
                f"beta{beta}_seed{seed}_{bb.lower()}.json"
            )
            if os.path.exists(out_path):
                with open(out_path) as f:
                    existing = json.load(f)
                test_acc = existing.get("test_accuracy")
                print(f"  skipped (exists) → {out_path}  acc={test_acc:.4f}")
                continue
            # Fall through to run if file missing
        run_one(op, ds, bb, beta, seed, cfg, results_dir, force=args.force)

    if args.compute_gaps:
        print("\nComputing gap-closed metrics...")
        operators = cfg.get("operators", [])
        for op, ds, bb, beta, seed in grid:
            if op not in ("zero_hop", "full"):
                compute_gap_closed(results_dir, op, ds, beta, seed, bb)


if __name__ == "__main__":
    main()
