#!/usr/bin/env python3
"""
FedProp Experiment Tracker - R1 & R1b Only

Tracks every atomic experiment for R1 (GCN) and R1b (GAT).

Methods: zero_hop, adjacency, diffusion, full
Betas: 1, 10, 10000
Models: GCN (Cora, Citeseer, Pubmed, ogbn-arxiv), GAT (Cora, Citeseer, Pubmed)
        Note: Pubmed+GAT uses PubmedGAT internally. ogbn-arxiv+GCN uses GCN_arxiv.

Usage:
    python /home/bosho/FP/experiments/track_progress.py

Outputs:
    - experiments/EXPERIMENT_TRACKER.csv
    - experiments/EXPERIMENT_TRACKER.md
"""

import csv
import json
import glob
import os
import re
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("/home/bosho/FP/experiments/results")
CONFIGS_DIR = Path("/home/bosho/FP/experiments/configs")
TRACKER_CSV = Path("/home/bosho/FP/experiments/EXPERIMENT_TRACKER.csv")
TRACKER_MD = Path("/home/bosho/FP/experiments/EXPERIMENT_TRACKER.md")


def resolve_actual_model(dataset, model):
    """Map config model to actual instantiated model class."""
    if dataset == "Pubmed" and model == "GAT":
        return "PubmedGAT"
    elif dataset == "ogbn-arxiv" and model == "GCN":
        return "GCN_arxiv"
    return model


def scan_existing_results():
    """Scan results directories for valid JSON result files."""
    results = {}  # key -> count
    
    for json_file in RESULTS_DIR.rglob("*.json"):
        if "propagation_stats" in str(json_file):
            continue
        if "manifest" in json_file.name:
            continue
        
        # Parse filename: results_Cora_zero_hop_GCN_beta10000_clients10_20260423_031010.json
        pattern = r'results_(\w+)_(\w+)_(\w+)_beta([\d.]+)_clients(\d+)'
        match = re.search(pattern, json_file.name)
        if not match:
            continue
        
        dataset, method, model, beta, clients = match.groups()
        beta = float(beta)
        clients = int(clients)
        
        # Skip beta=100 (not in our target set)
        if beta not in [1.0, 10.0, 10000.0]:
            continue
        
        # Check if valid
        try:
            with open(json_file) as f:
                data = json.load(f)
            avg = data.get("summary", {}).get("average_client_result")
            if avg is None or avg != avg:  # NaN check
                continue
            
            cfg = data.get("experiment_config", {})
            seed = cfg.get("seed", -1)
            
            # Determine method: if data_loading_option is 'full', map to 'full'
            dl = cfg.get("data_loading_option", method)
            if dl == "full":
                method = "full"
            
            key = (dataset, model, method, beta, seed)
            results[key] = results.get(key, 0) + 1
        except:
            pass
    
    return results


def get_planned_experiments():
    """Get all planned experiments from R1 and R1b configs."""
    planned = []
    
    for result_id in ["R1", "R1b"]:
        result_configs_dir = CONFIGS_DIR / result_id
        if not result_configs_dir.exists():
            continue
        
        for config_file in sorted(result_configs_dir.glob("*.yaml")):
            if config_file.name == "base.yaml":
                continue
            
            # Load config
            import yaml
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            
            datasets = cfg.get("datasets", [])
            models = cfg.get("models", [])
            data_loadings = cfg.get("data_loading", [])
            betas = cfg.get("beta", [])
            repetitions = int(cfg.get("repetitions", 1))
            experiment_seed = int(cfg.get("experiment_seed", 0))
            hop = cfg.get("hop", 1)
            
            # Filter to target betas
            betas = [b for b in betas if b in [1, 10, 10000]]
            
            # Standard variants: zero_hop, adjacency, diffusion
            for dataset in datasets:
                for model in models:
                    actual_model = resolve_actual_model(dataset, model)
                    for method in data_loadings:
                        if method not in ["zero_hop", "adjacency", "diffusion"]:
                            continue
                        for beta in betas:
                            for seed_idx in range(repetitions):
                                seed = experiment_seed + seed_idx
                                planned.append({
                                    "result_id": result_id,
                                    "config_file": str(config_file.relative_to(Path("/home/bosho/FP"))),
                                    "dataset": dataset,
                                    "model": actual_model,
                                    "method": method,
                                    "beta": beta,
                                    "hop": hop,
                                    "seed": seed,
                                    "variant_type": "standard",
                                })
            
            # Full variant (centralised + fedprop_full)
            # These use data_loading=full
            full_result_ids = {"R1", "R1b", "R6", "A1"}
            if result_id in full_result_ids:
                for dataset in datasets:
                    for model in models:
                        actual_model = resolve_actual_model(dataset, model)
                        for beta in betas:
                            for seed_idx in range(repetitions):
                                seed = experiment_seed + seed_idx
                                # Centralised (1 client)
                                planned.append({
                                    "result_id": result_id,
                                    "config_file": str(config_file.relative_to(Path("/home/bosho/FP"))),
                                    "dataset": dataset,
                                    "model": actual_model,
                                    "method": "full",
                                    "beta": beta,
                                    "hop": hop,
                                    "seed": seed,
                                    "variant_type": "centralised",
                                })
                                # FedProp-Full (10 clients)
                                planned.append({
                                    "result_id": result_id,
                                    "config_file": str(config_file.relative_to(Path("/home/bosho/FP"))),
                                    "dataset": dataset,
                                    "model": actual_model,
                                    "method": "full",
                                    "beta": beta,
                                    "hop": hop,
                                    "seed": seed,
                                    "variant_type": "fedprop_full",
                                })
    
    return planned


def generate_tracker():
    """Generate the tracker CSV and markdown."""
    print("Scanning R1 & R1b configs...")
    planned = get_planned_experiments()
    print(f"Found {len(planned)} planned atomic runs")
    
    print("Scanning existing results...")
    existing = scan_existing_results()
    print(f"Found {len(existing)} valid result files")
    
    # Write CSV
    fieldnames = [
        "result_id", "config_file", "dataset", "model", "method",
        "beta", "hop", "seed", "variant_type", "status"
    ]
    with open(TRACKER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for run in planned:
            key = (run["dataset"], run["model"], run["method"], run["beta"], run["seed"])
            run["status"] = "COMPLETE" if key in existing else "PENDING"
            writer.writerow(run)
    
    # Write Markdown summary
    with open(TRACKER_MD, "w") as f:
        f.write("# FedProp Experiment Tracker (R1 & R1b)\n\n")
        f.write(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Planned runs:** {len(planned)}  \n")
        f.write(f"**Complete runs:** {len(existing)}  \n")
        f.write(f"**Pending runs:** {len(planned) - len(existing)}  \n\n")
        
        # Group by result_id, dataset, model, method, beta
        from collections import defaultdict
        groups = defaultdict(lambda: {"total": 0, "complete": 0})
        
        for run in planned:
            group_key = (run["result_id"], run["dataset"], run["model"], run["method"], run["beta"])
            groups[group_key]["total"] += 1
            key = (run["dataset"], run["model"], run["method"], run["beta"], run["seed"])
            if key in existing:
                groups[group_key]["complete"] += 1
        
        f.write("## Progress by Configuration\n\n")
        f.write("| Result | Dataset | Model | Method | Beta | Complete | Total | Status |\n")
        f.write("|--------|---------|-------|--------|------|----------|-------|--------|\n")
        
        for (result_id, dataset, model, method, beta), counts in sorted(groups.items()):
            complete = counts["complete"]
            total = counts["total"]
            if complete == total:
                status = "✅"
            elif complete > 0:
                status = f"⏳ {complete}/{total}"
            else:
                status = "⬜"
            f.write(f"| {result_id} | {dataset} | {model} | {method} | {beta} | {complete} | {total} | {status} |\n")
    
    print(f"\nTracker written to:")
    print(f"  CSV: {TRACKER_CSV}")
    print(f"  MD:  {TRACKER_MD}")
    
    # Print summary
    print("\n=== Summary ===")
    r1_count = sum(1 for r in planned if r["result_id"] == "R1")
    r1b_count = sum(1 for r in planned if r["result_id"] == "R1b")
    print(f"R1 planned:  {r1_count}")
    print(f"R1b planned: {r1b_count}")
    print(f"Total:       {len(planned)}")


if __name__ == "__main__":
    generate_tracker()
