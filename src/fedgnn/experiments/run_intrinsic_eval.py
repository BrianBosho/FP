"""
Layer 1-3 intrinsic evaluation runner.

Runs propagation-only experiments (no GNN training) and records feature
reconstruction metrics, convergence dynamics, and wall-clock timing.

Usage:
    python -m src.fedgnn.experiments.run_intrinsic_eval \
        --config experiments/propagator_eval/configs/L1_L3_primary.yaml \
        [--operator adjacency] [--dataset Cora] [--beta 10000] [--seed 0]

If --operator/--dataset/--beta/--seed are omitted the full grid in the
config file is run.
"""

import argparse
import json
import os
import time
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import yaml
from torch_geometric.utils import k_hop_subgraph

from src.fedgnn.data.loaders import load_dataset
from src.fedgnn.data.partitioning import label_dirichlet_partition
from src.fedgnn.data.propagation import propagate_features, _compute_intrinsic_metrics
from src.fedgnn.utils.run import resolve_torch_device


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    base_path = os.path.join(os.path.dirname(__file__), "../../../../conf/intrinsic_eval.yaml")
    base_path = os.path.normpath(base_path)
    cfg = {}
    if os.path.exists(base_path):
        with open(base_path) as f:
            cfg.update(yaml.safe_load(f) or {})
    with open(config_path) as f:
        cfg.update(yaml.safe_load(f) or {})
    return cfg


# ---------------------------------------------------------------------------
# Single-run logic
# ---------------------------------------------------------------------------

def run_one(operator: str, dataset: str, beta: int, seed: int, cfg: dict) -> dict:
    """Run intrinsic eval for one (operator, dataset, beta, seed) combination."""
    device_str = cfg.get("feature_prop_device", "cpu")
    device = resolve_torch_device(device_str)
    device_str = str(device)

    hop = cfg.get("hop", 1)
    num_clients = cfg.get("num_clients", 10)

    # Load full graph
    data, _ = load_dataset(dataset, device_str, config=cfg)

    # Partition into client node sets
    labels = data.y.cpu().numpy()
    import numpy as np
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(labels))
    split_data_indexes = label_dirichlet_partition(
        labels, len(labels), n_classes, num_clients, beta, seed=seed
    )

    per_client_results = []

    for client_id, client_node_indices in enumerate(split_data_indexes):
        if not isinstance(client_node_indices, torch.Tensor):
            client_node_indices = torch.tensor(client_node_indices, dtype=torch.long)

        # Build k-hop subgraph with TRUE features (no zeroing)
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            client_node_indices.cpu(),
            hop,
            data.edge_index.cpu(),
            relabel_nodes=True,
        )
        X_true = data.x.cpu()[subset].to(device)
        sub_edge_index = sub_edge_index.to(device)

        # Construct known-feature mask (True = original client node)
        mask = torch.zeros(len(subset), dtype=torch.bool, device=device)
        mask[mapping] = True

        n_known = mask.sum().item()
        n_unknown = (~mask).sum().item()
        n_nodes = len(subset)

        # Boundary coverage (fraction of unknown nodes with >=1 known neighbor)
        row, col = sub_edge_index[0], sub_edge_index[1]
        has_known_nbr = torch.zeros(n_nodes, dtype=torch.float, device=device)
        known_src = mask[row]
        if known_src.any():
            has_known_nbr[col[known_src]] = 1.0
        boundary_coverage = has_known_nbr[~mask].mean().item() if n_unknown > 0 else 1.0

        # Zero unknown features for propagation input
        x_init = X_true.clone()
        x_init[~mask] = 0.0

        # Run propagation with intrinsic eval
        result = propagate_features(
            x_init, sub_edge_index, mask, device,
            num_iterations=cfg.get("num_iterations", 100),
            mode=operator,
            alpha=cfg.get("alpha", 0.5),
            tol=cfg.get("feature_prop_tolerance", 1e-4),
            config=cfg,
            intrinsic_eval=True,
            X_true=X_true,
        )

        client_rec = {
            "client_id": client_id,
            "n_known": n_known,
            "n_unknown": n_unknown,
            "missing_neighbor_frac": n_unknown / max(n_nodes, 1),
            "boundary_coverage": boundary_coverage,
            "n_iters": result["n_iters"],
            "converged": result["converged"],
            "wall_time_sec": result["wall_time"],
            "residuals": result["residuals"],
        }
        if "intrinsic_metrics" in result:
            client_rec.update(result["intrinsic_metrics"])

        per_client_results.append(client_rec)

    # Aggregate across clients
    def _mean(key):
        vals = [c[key] for c in per_client_results if key in c]
        return float(sum(vals) / len(vals)) if vals else None

    def _std(key):
        import statistics
        vals = [c[key] for c in per_client_results if key in c]
        return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0

    aggregate = {
        "mse_mean": _mean("mse"),
        "mse_std": _std("mse"),
        "cosine_sim_mean": _mean("cosine_sim"),
        "cosine_sim_std": _std("cosine_sim"),
        "recovery_ratio_mean": _mean("recovery_ratio"),
        "recovery_ratio_std": _std("recovery_ratio"),
        "boundary_coverage_mean": _mean("boundary_coverage"),
        "n_iters_mean": _mean("n_iters"),
        "n_iters_std": _std("n_iters"),
        "wall_time_total_sec": sum(c["wall_time_sec"] for c in per_client_results),
        "convergence_rate": sum(1 for c in per_client_results if c["converged"]) / num_clients,
    }

    return {
        "operator": operator,
        "dataset": dataset.lower(),
        "beta": beta,
        "seed": seed,
        "n_clients": num_clients,
        "hop": hop,
        "per_client": per_client_results,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_result(result: dict, results_dir: str):
    operator = result["operator"]
    dataset = result["dataset"]
    beta = result["beta"]
    seed = result["seed"]
    out_dir = os.path.join(results_dir, operator, dataset)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"beta{beta}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_grid(cfg: dict, args: argparse.Namespace) -> list[tuple]:
    operators = [args.operator] if args.operator else cfg.get("operators", [])
    datasets = [args.dataset] if args.dataset else cfg.get("datasets", [])
    betas = [args.beta] if args.beta is not None else cfg.get("beta", [10000])
    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [0])
    return list(product(operators, datasets, betas, seeds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--operator", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--beta", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip grid items whose result file already exists.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result file already exists.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg.get("results_dir", "experiments/propagator_eval/results/intrinsic")
    grid = build_grid(cfg, args)

    print(f"Intrinsic eval — {len(grid)} runs")
    for i, (op, ds, beta, seed) in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] {op} / {ds} / beta={beta} / seed={seed}")
        out_path = os.path.join(results_dir, op, ds.lower(), f"beta{beta}_seed{seed}.json")
        if os.path.exists(out_path) and not args.force:
            with open(out_path) as f:
                existing = json.load(f)
            mse = existing.get("aggregate", {}).get("mse_mean", "N/A")
            recovery = existing.get("aggregate", {}).get("recovery_ratio_mean", "N/A")
            print(f"  skipped (exists) → {out_path}  mse={mse}  recovery={recovery}")
            continue
        t0 = time.perf_counter()
        try:
            result = run_one(op, ds, beta, seed, cfg)
            save_result(result, results_dir)
            print(f"  done in {time.perf_counter()-t0:.1f}s  "
                  f"mse={result['aggregate']['mse_mean']:.4f}  "
                  f"recovery={result['aggregate']['recovery_ratio_mean']:.3f}")
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
