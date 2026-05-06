"""Preflight feasibility checker for FP experiments.

Usage:
    python -m src.fedgnn.experiments.preflight --config experiments/configs/R1/cora.yaml
    python -m src.fedgnn.experiments.preflight --config conf/smoke.yaml --json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Static dataset catalogue (nodes, edges, features, classes)
_DATASET_CATALOGUE: dict[str, dict[str, int]] = {
    "cora":           {"nodes": 2708,    "edges": 10556,     "features": 1433, "classes": 7},
    "citeseer":       {"nodes": 3327,    "edges": 9104,      "features": 3703, "classes": 6},
    "pubmed":         {"nodes": 19717,   "edges": 88648,     "features": 500,  "classes": 3},
    "ogbn-arxiv":     {"nodes": 169343,  "edges": 2315598,   "features": 128,  "classes": 40},
    "ogbn-products":  {"nodes": 2449029, "edges": 123718280, "features": 100,  "classes": 47},
    "amazon-computers": {"nodes": 13381, "edges": 491722,    "features": 767,  "classes": 10},
    "amazon-photo":   {"nodes": 7487,    "edges": 238162,    "features": 745,  "classes": 8},
}

_BYTES_F32 = 4
_MB = 1024 * 1024


def _lookup_dataset(name: str) -> dict[str, int] | None:
    key = name.lower().replace("_", "-")
    return _DATASET_CATALOGUE.get(key)


def _first(v: Any) -> Any:
    """Return first element from a list/ListConfig, or v itself."""
    if isinstance(v, (list, tuple)) and v:
        return v[0]
    try:
        from omegaconf import ListConfig
        if isinstance(v, ListConfig) and len(v) > 0:
            return v[0]
    except ImportError:
        pass
    return v


def _model_param_estimate(
    feature_dim: int,
    num_classes: int,
    hidden_dim: int,
    num_layers: int,
    model_type: str,
) -> int:
    """Rough parameter count for common GNN architectures."""
    mt = model_type.upper()
    if mt.startswith("GAT"):
        return (
            feature_dim * hidden_dim
            + max(0, num_layers - 1) * hidden_dim * hidden_dim
            + hidden_dim * num_classes
        )
    # GCN / SAGE default
    params = feature_dim * hidden_dim + hidden_dim
    for _ in range(1, max(1, num_layers - 1)):
        params += hidden_dim * hidden_dim + hidden_dim
    params += hidden_dim * num_classes + num_classes
    return params


def _cache_state(cache_dir: Path | None) -> str:
    if cache_dir is None:
        return "unknown (cache disabled?)"
    if not cache_dir.exists():
        return "miss (not built)"
    state_file = cache_dir / "state"
    if state_file.exists():
        return state_file.read_text().strip()
    if (cache_dir / "manifest.json").exists():
        return "ready (legacy, no state file)"
    return "incomplete"


def run_preflight(config_path: str, *, verbose: bool = True) -> dict[str, Any]:
    """Run all feasibility checks for *config_path* and return a report dict."""
    from src.fedgnn.utils.config import load_config

    cfg = load_config(config_path)

    dataset_name = str(_first(cfg.get("datasets", cfg.get("dataset_name", "Cora"))))
    data_loading = str(_first(cfg.get("data_loading", "zero_hop")))
    model_type   = str(_first(cfg.get("models",      cfg.get("model_type", "GCN"))))

    num_clients   = int(_first(cfg.get("num_clients", 10)))
    num_rounds    = int(cfg.get("num_rounds", 10))
    repetitions   = int(cfg.get("repetitions", 5))
    beta          = float(_first(cfg.get("beta", 1.0)))
    hop           = int(cfg.get("hop", 1))
    max_concurrent = int(cfg.get("max_concurrent_clients", num_clients) or num_clients)

    use_pe_val = _first(cfg.get("use_pe", False))
    use_pe = bool(use_pe_val) if not isinstance(use_pe_val, str) else use_pe_val.lower() == "true"
    pe_r = int(cfg.get("pe_r", 64))

    ds = _lookup_dataset(dataset_name)
    ds_nodes    = ds["nodes"]    if ds else None
    ds_edges    = ds["edges"]    if ds else None
    ds_features = ds["features"] if ds else None
    ds_classes  = ds["classes"]  if ds else None

    base_features = int(ds_features) if ds_features else 128
    num_classes   = int(ds_classes)  if ds_classes  else 10
    feature_dim   = base_features + pe_r if use_pe else base_features

    arch_cfg = cfg.get("model_architecture") or {}
    model_arch = (
        arch_cfg.get(model_type) or arch_cfg.get("default") or {}
    )
    hidden_dim = int(model_arch.get("hidden_dim", 16))
    num_layers = int(model_arch.get("num_layers", 2))

    params     = _model_param_estimate(feature_dim, num_classes, hidden_dim, num_layers, model_type)
    model_mb   = params * _BYTES_F32 / _MB
    # FedAvg: each round = num_clients uploads + num_clients downloads
    total_comm_mb = model_mb * num_clients * num_rounds * 2

    nodes_per_client = ds_nodes // num_clients if ds_nodes else None
    shard_mb = (
        nodes_per_client * feature_dim * _BYTES_F32 / _MB
        if nodes_per_client else None
    )
    partition_ram_mb = shard_mb * num_clients if shard_mb else None
    gpu_mb_per_client = (shard_mb or 0.0) + model_mb + 50.0  # +50 MB activations

    # Shard cache check
    try:
        from src.fedgnn.data.shard_cache import get_cache_dir, build_cache_payload
        _payload = build_cache_payload(
            dataset_name, data_loading, num_clients, beta, hop, False,
            cfg if hasattr(cfg, "get") else dict(cfg),
        )
        _cdir = get_cache_dir(
            dataset_name, data_loading, num_clients, beta, hop, False,
            cfg if hasattr(cfg, "get") else dict(cfg),
        )
    except Exception:
        _cdir = None
    cache_dir_str = str(_cdir) if _cdir else None
    cache_st = _cache_state(_cdir)

    report: dict[str, Any] = {
        "config_path":                config_path,
        "dataset":                    dataset_name,
        "data_loading":               data_loading,
        "model_type":                 model_type,
        "num_clients":                num_clients,
        "num_rounds":                 num_rounds,
        "repetitions":                repetitions,
        "beta":                       beta,
        "hop":                        hop,
        "use_pe":                     use_pe,
        # Dataset
        "dataset_nodes":              ds_nodes,
        "dataset_edges":              ds_edges,
        # Features
        "base_feature_dim":           base_features,
        "feature_dim_after_pe":       feature_dim,
        # Model
        "model_hidden_dim":           hidden_dim,
        "model_num_layers":           num_layers,
        "model_params_approx":        params,
        "model_size_mb":              round(model_mb, 4),
        # Communication
        "total_comm_mb":              round(total_comm_mb, 2),
        # Memory
        "nodes_per_client_approx":    nodes_per_client,
        "shard_feature_mb_approx":    round(shard_mb, 2)      if shard_mb      else None,
        "total_partition_ram_mb":     round(partition_ram_mb, 2) if partition_ram_mb else None,
        "ray_object_store_mb_approx": round(partition_ram_mb, 2) if partition_ram_mb else None,
        "gpu_mb_per_client_approx":   round(gpu_mb_per_client, 2),
        # Concurrency
        "max_concurrent_clients":     max_concurrent,
        # Cache
        "shard_cache_dir":            cache_dir_str,
        "shard_cache_state":          cache_st,
        # Feasibility
        "ray_concurrency_feasible":   max_concurrent <= num_clients,
        "comm_feasible":              total_comm_mb < 10_000,
    }

    if verbose:
        _print_report(report)
    return report


def _ok(flag: bool) -> str:
    return "OK      " if flag else "WARNING "


def _fmt_opt(v: Any, suffix: str = "") -> str:
    return f"{v}{suffix}" if v is not None else "?"


def _print_report(r: dict[str, Any]) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("PREFLIGHT FEASIBILITY REPORT")
    print(sep)
    print(f"  Config :         {r['config_path']}")
    print(f"  Dataset:         {r['dataset']}  "
          f"({_fmt_opt(r['dataset_nodes'])} nodes, {_fmt_opt(r['dataset_edges'])} edges)")
    print(f"  Loading:         {r['data_loading']}  hop={r['hop']}")
    print(f"  Model  :         {r['model_type']}  "
          f"hidden={r['model_hidden_dim']}  layers={r['model_num_layers']}")
    print(f"  Clients:         {r['num_clients']}  "
          f"rounds={r['num_rounds']}  reps={r['repetitions']}")
    print()
    print("  Features:")
    print(f"    Base dim:              {r['base_feature_dim']}")
    pe_note = "(PE enabled)" if r["use_pe"] else "(PE disabled)"
    print(f"    After PE:              {r['feature_dim_after_pe']}  {pe_note}")
    print()
    print("  Model:")
    print(f"    Approx params:         {r['model_params_approx']:,}")
    print(f"    Model size:            {r['model_size_mb']:.4f} MB")
    print(f"    FedAvg comm total:     {r['total_comm_mb']:.2f} MB")
    print()
    print("  Memory estimates (rough):")
    print(f"    Nodes per client:      ~{_fmt_opt(r['nodes_per_client_approx'])}")
    print(f"    Shard features:        ~{_fmt_opt(r['shard_feature_mb_approx'])} MB/client")
    print(f"    Partition RAM total:   ~{_fmt_opt(r['total_partition_ram_mb'])} MB")
    print(f"    Ray object store:      ~{_fmt_opt(r['ray_object_store_mb_approx'])} MB")
    print(f"    GPU per client:        ~{r['gpu_mb_per_client_approx']:.2f} MB (features+model+activations)")
    print()
    print("  Shard cache:")
    print(f"    Dir:   {r['shard_cache_dir']}")
    print(f"    State: {r['shard_cache_state']}")
    print()
    print("  Feasibility:")
    print(f"    {_ok(r['ray_concurrency_feasible'])} Ray concurrency  "
          f"max_concurrent={r['max_concurrent_clients']}  clients={r['num_clients']}")
    print(f"    {_ok(r['comm_feasible'])} Comm budget  "
          f"{r['total_comm_mb']:.0f} MB  (<10 GB threshold)")
    print(sep)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preflight: estimate feasibility of an FP experiment config."
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--json", action="store_true", help="Print report as JSON instead of table")
    ns = parser.parse_args(argv)

    report = run_preflight(ns.config, verbose=not ns.json)
    if ns.json:
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
