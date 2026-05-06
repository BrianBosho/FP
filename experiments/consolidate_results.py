#!/usr/bin/env python3
"""Consolidate R1 (GCN) and R1b (GAT) experiment results into canonical CSVs.

Scans all R1_* and R1b_* result directories, deduplicates by config tuple,
picks the most complete 10-rep result per config, and outputs:
  output/R1_consolidated.csv
  output/R1b_consolidated.csv
  output/completeness_R1.txt
  output/completeness_R1b.txt

Usage:
    python experiments/consolidate_results.py
    python experiments/consolidate_results.py --track R1
    python experiments/consolidate_results.py --track R1b
    python experiments/consolidate_results.py --verbose
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Directories to skip (quickval, test runs, log files, known-bad baselines)
# _v2 dirs systematically failed (200 rounds, no convergence, ~40-64% accuracy)
# pubmed_gcn_v3 also failed (59.6%, 200 rounds, no convergence)
SKIP_PATTERNS = [
    "quickval", "parallel_test", "centralized", "minimal",
    "adam_quickval", "sgd_quickval", "pe_quickval",
    "_v2", "pubmed_gcn_v3",
    ".log",
]

# Minimum plausible accuracy — conservative floor to catch catastrophic failures only.
# Known bad runs (_v2, pubmed_gcn_v3) are excluded via SKIP_PATTERNS instead.
MIN_ACC_BY_DATASET = {
    "default": 0.40,
}

# Canonical config dimensions
CONFIG_KEYS = ("dataset", "model", "propagation", "beta", "hop", "use_pe")

# Minimum reps for a MERGED/consolidated result to be reported
MIN_REPS = 5
CANONICAL_REPS = 10
# Per-file minimum — single-rep baseline files are pooled later
MIN_REPS_PER_FILE = 1


def normalize_beta(beta) -> float:
    """Normalize beta to float, treating 10000.0 == 10000."""
    return float(beta)


def parse_result_file(path: Path) -> Optional[dict]:
    """Parse a single result JSON and return a normalized record, or None."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    cfg = data.get("experiment_config", {})
    rounds = data.get("rounds", [])

    # Extract valid repetitions — conservative floor to exclude catastrophic failures
    min_acc = MIN_ACC_BY_DATASET["default"]
    valid_reps = [
        r for r in rounds
        if isinstance(r.get("global_result"), (int, float))
        and r["global_result"] >= min_acc
    ]
    if len(valid_reps) < MIN_REPS_PER_FILE:
        return None

    global_accs = [r["global_result"] for r in valid_reps]
    seeds = [r.get("experiment_seed") for r in valid_reps]

    # Use pre-computed summary if available and consistent
    summary = data.get("summary", {})
    if summary.get("average_global_result") and summary.get("average_global_result") > 0.1:
        mean_acc = summary["average_global_result"]
        std_acc = summary.get("std_global", np.std(global_accs))
    else:
        mean_acc = np.mean(global_accs)
        std_acc = np.std(global_accs)

    return {
        "dataset": cfg.get("dataset", "Unknown"),
        "model": cfg.get("model_type", "GCN"),
        "propagation": cfg.get("data_loading_option", "unknown"),
        "beta": normalize_beta(cfg.get("beta", 0)),
        "hop": int(cfg.get("hop", 1)),
        "use_pe": bool(cfg.get("use_pe", False)),
        "num_clients": int(cfg.get("num_clients", 10)),
        "fulltraining_flag": bool(cfg.get("fulltraining_flag", False)),
        "n_reps": len(valid_reps),
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "per_rep_accs": global_accs,
        "seeds": seeds,
        "_source": str(path),
    }


def should_skip(dir_name: str) -> bool:
    """Return True if this directory should be excluded from consolidation."""
    name_lower = dir_name.lower()
    return any(pat in name_lower for pat in SKIP_PATTERNS)


TRACK_HOP = {"R1": 1, "R1b": 2}
TRACK_MODEL = {"R1": "GCN", "R1b": "GAT"}


def collect_all_records(track: str) -> list[dict]:
    """Scan all relevant result directories for the given track.

    R1:  R1_* dirs  +  baseline_* dirs (single-rep GCN runs, pooled across seeds)
         Only includes hop=1 GCN records.
    R1b: R1b_* dirs only — only includes hop=2 GAT records.
    """
    expected_hop = TRACK_HOP[track]
    expected_model = TRACK_MODEL[track]

    if track == "R1":
        def pattern(name: str) -> bool:
            return (
                (name.startswith("R1_") and not name.startswith("R1b_"))
                or name.startswith("baseline_")
            )
    else:
        def pattern(name: str) -> bool:
            return name.startswith("R1b_")

    all_records = []
    for dir_entry in sorted(RESULTS_DIR.iterdir()):
        if not dir_entry.is_dir():
            continue
        if not pattern(dir_entry.name):
            continue
        if should_skip(dir_entry.name):
            continue

        for json_path in sorted(dir_entry.rglob("results_*.json")):
            rec = parse_result_file(json_path)
            if rec is not None:
                if rec["model"] != expected_model:
                    continue
                if rec["hop"] != expected_hop:
                    continue
                all_records.append(rec)

    return all_records


def config_key(rec: dict) -> tuple:
    """Canonical key for deduplicating records."""
    return (
        rec["dataset"],
        rec["model"],
        rec["propagation"],
        rec["beta"],
        rec["hop"],
        rec["use_pe"],
    )


def merge_records(records: list[dict]) -> dict:
    """Merge multiple records for the same config, pooling all reps.

    Priority: prefer a single file with CANONICAL_REPS; otherwise pool
    individual reps across files, deduplicating near-identical values.
    """
    if len(records) == 1:
        return records[0]

    # If any single file already has canonical reps, use it
    best = max(records, key=lambda r: r["n_reps"])
    if best["n_reps"] >= CANONICAL_REPS:
        return best

    # Pool all per-rep accuracies, round to 4 dp to deduplicate identical reruns
    seen = set()
    all_accs = []
    for rec in sorted(records, key=lambda r: -r["n_reps"]):
        for acc in rec["per_rep_accs"]:
            key = round(acc, 4)
            if key not in seen:
                seen.add(key)
                all_accs.append(acc)

    all_accs = all_accs[:CANONICAL_REPS]
    merged = dict(best)
    merged["per_rep_accs"] = all_accs
    merged["n_reps"] = len(all_accs)
    merged["mean_acc"] = float(np.mean(all_accs))
    merged["std_acc"] = float(np.std(all_accs))
    merged["_source"] = f"pooled:{len(records)} files"
    return merged


def consolidate(track: str, verbose: bool = False) -> pd.DataFrame:
    """Collect, deduplicate, and consolidate records for the given track."""
    print(f"\n{'=' * 60}")
    print(f"Consolidating {track} results...")
    print(f"{'=' * 60}")

    records = collect_all_records(track)
    print(f"Found {len(records)} valid result records across all {track}_* directories")

    # Group by config key
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        groups[config_key(rec)].append(rec)

    if verbose:
        print(f"\nGrouped into {len(groups)} unique configs")

    # Merge each group
    canonical = []
    for key, recs in sorted(groups.items()):
        merged = merge_records(recs)
        if verbose and len(recs) > 1:
            print(f"  Merged {len(recs)} files for {key}: n_reps={merged['n_reps']}")
        canonical.append(merged)

    df = pd.DataFrame(canonical)
    # Drop internal columns for output
    output_cols = [
        "dataset", "model", "propagation", "beta", "hop", "use_pe",
        "num_clients", "n_reps", "mean_acc", "std_acc",
    ]
    df_out = df[output_cols].copy()
    df_out["beta"] = df_out["beta"].astype(int)
    df_out = df_out.sort_values(
        ["dataset", "propagation", "beta", "use_pe", "hop"]
    ).reset_index(drop=True)

    return df_out


def print_completeness(df: pd.DataFrame, track: str) -> str:
    """Print and return a completeness matrix for the given track."""
    datasets = ["Cora", "Citeseer", "Pubmed"]
    propagations = ["zero_hop", "adjacency", "diffusion", "full"]
    betas = [1, 10, 10000]

    if track == "R1":
        # GCN: PE is not used; "pe" in dir name means propagation-enabled
        hops = [1]
        pe_variants = [False]
    else:
        # GAT: PE is a real experimental dimension
        hops = [2]
        pe_variants = [False, True]

    lines = [f"\n{'=' * 70}"]
    lines.append(f"COMPLETENESS MATRIX: {track}")
    lines.append(f"{'=' * 70}")
    lines.append(f"{'Config':<60} {'reps':>6} {'mean_acc':>9}")
    lines.append("-" * 70)

    total = 0
    complete = 0
    incomplete = 0
    missing = 0

    for dataset in datasets:
        for hop in hops:
            for use_pe in pe_variants:
                pe_label = "PE=T" if use_pe else "PE=F"
                for prop in propagations:
                    for beta in betas:
                        total += 1
                        row = df[
                            (df["dataset"] == dataset)
                            & (df["propagation"] == prop)
                            & (df["beta"] == beta)
                            & (df["hop"] == hop)
                            & (df["use_pe"] == use_pe)
                        ]
                        if len(row) == 0:
                            status = "MISSING"
                            missing += 1
                            lines.append(
                                f"  {dataset:10s} hop{hop} {pe_label} {prop:12s} β={beta:<6} "
                                f"{'':>6} {'MISSING':>9}"
                            )
                        elif row.iloc[0]["n_reps"] < CANONICAL_REPS:
                            n = row.iloc[0]["n_reps"]
                            acc = row.iloc[0]["mean_acc"]
                            incomplete += 1
                            lines.append(
                                f"  {dataset:10s} hop{hop} {pe_label} {prop:12s} β={beta:<6} "
                                f"{n:>6} {acc:>9.4f}  ⚠ INCOMPLETE ({n}/10)"
                            )
                        else:
                            acc = row.iloc[0]["mean_acc"]
                            std = row.iloc[0]["std_acc"]
                            complete += 1

    lines.append("-" * 70)
    lines.append(
        f"Summary: {complete}/{total} complete, {incomplete} incomplete, {missing} missing"
    )
    lines.append(f"{'=' * 70}")

    report = "\n".join(lines)
    print(report)
    return report


def print_summary_table(df: pd.DataFrame, track: str, betas: list = None):
    """Print a human-readable summary table of key results."""
    if betas is None:
        betas = [10000, 10]  # Paper betas (IID and non-IID)

    print(f"\n{'=' * 80}")
    print(f"PAPER TABLE PREVIEW: {track} (β=10000 IID and β=10 non-IID)")
    print(f"{'=' * 80}")

    datasets = ["Cora", "Citeseer", "Pubmed"]
    propagations = ["zero_hop", "adjacency", "diffusion", "full"]
    prop_labels = {
        "zero_hop": "FedProp-Zero",
        "adjacency": "FedProp-Adj",
        "diffusion": "FedProp-Diff",
        "full": "FedProp-Full",
    }

    if track == "R1":
        hop = 1
        pe_variants = [False]
    else:
        hop = 2
        pe_variants = [False, True]

    # Header
    header = f"{'Method':<20}"
    for ds in datasets:
        for beta in betas:
            beta_label = "IID" if beta == 10000 else "nIID"
            header += f"  {ds[:3]}-{beta_label}"
    print(header)
    print("-" * len(header))

    for pe in pe_variants:
        pe_label = "  PE" if pe else "noPE"
        for prop in propagations:
            row_str = f"{pe_label} {prop_labels.get(prop, prop):<15}"
            for ds in datasets:
                for beta in betas:
                    match = df[
                        (df["dataset"] == ds)
                        & (df["propagation"] == prop)
                        & (df["beta"] == beta)
                        & (df["hop"] == hop)
                        & (df["use_pe"] == pe)
                    ]
                    if len(match) == 0:
                        row_str += f"  {'  —  ':>7}"
                    else:
                        acc = match.iloc[0]["mean_acc"]
                        std = match.iloc[0]["std_acc"]
                        n = match.iloc[0]["n_reps"]
                        flag = "" if n >= CANONICAL_REPS else f"*{n}"
                        row_str += f"  {acc*100:5.1f}{flag:<2}"
            print(row_str)
        if track == "R1b" and not pe:
            print()


def main():
    parser = argparse.ArgumentParser(description="Consolidate R1/R1b experiment results")
    parser.add_argument("--track", choices=["R1", "R1b", "both"], default="both")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    tracks = ["R1", "R1b"] if args.track == "both" else [args.track]

    for track in tracks:
        df = consolidate(track, verbose=args.verbose)

        # Save consolidated CSV
        out_path = OUTPUT_DIR / f"{track}_consolidated.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path} ({len(df)} rows)")

        # Print completeness
        report = print_completeness(df, track)
        report_path = OUTPUT_DIR / f"completeness_{track}.txt"
        report_path.write_text(report)

        # Print paper table preview
        print_summary_table(df, track)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
