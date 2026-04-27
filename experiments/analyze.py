#!/usr/bin/env python3
"""Analyze FedProp experiment results and produce paper tables + figures.

Reads Track A FedProp JSON results from experiments/results/{R_id}/ and, once
Track B is implemented, normalized competitor baseline JSON from
experiments/baselines/**. Produces formatted tables (CSV + markdown) and
figures (PDF) matching the locked experimental design's reporting format.

Usage:
    python experiments/analyze.py --result R1       # produce Table 1
    python experiments/analyze.py --figure 2         # produce Figure 2
    python experiments/analyze.py --all              # produce everything
    python experiments/analyze.py --status           # show what data exists
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINES_DIR = Path(__file__).resolve().parent / "baselines"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Design doc method names mapped from codebase data_loading values
METHOD_NAMES = {
    "zero_hop": "FedProp-Zero",
    "adjacency": "FedProp (Adj)",
    "diffusion": "FedProp (Diff)",
    "chebyshev-diffusion": "FedProp (Cheb)",
}

BETA_LABELS = {
    10000: "IID (β=10k)",
    100: "β=100",
    10: "non-IID (β=10)",
    1: "β=1",
}

DATASET_ORDER = ["Cora", "Citeseer", "Pubmed", "ogbn-arxiv",
                 "Texas", "Wisconsin",
                 "Computers", "Photo"]


def collect_json_results(result_id: str) -> list[dict]:
    """Read all JSON result files from experiments/results/{result_id}/."""
    d = RESULTS_DIR / result_id
    if not d.is_dir():
        return []

    results = []
    for p in sorted(d.rglob("*.json")):
        if p.name == "manifest.json":
            continue
        try:
            data = json.loads(p.read_text())
            data["_source_file"] = str(p)
            results.append(data)
        except json.JSONDecodeError:
            print(f"Warning: could not parse {p}")
    return results


def collect_baseline_results() -> list[dict]:
    """Read normalized Track B baseline JSON files, when available."""
    if not BASELINES_DIR.is_dir():
        return []
    results = []
    for p in sorted(BASELINES_DIR.rglob("*.json")):
        try:
            data = json.loads(p.read_text())
            data["_source_file"] = str(p)
            results.append(data)
        except json.JSONDecodeError:
            print(f"Warning: could not parse baseline JSON {p}")
    return results


def extract_summary_rows(results: list[dict]) -> pd.DataFrame:
    """Convert raw JSON results into a flat summary DataFrame."""
    rows = []
    for r in results:
        exp_cfg = r.get("experiment_config", {})
        summary = r.get("summary", {})
        duration = r.get("duration", {})

        dataset = exp_cfg.get("dataset", "?")
        data_loading = exp_cfg.get("data_loading_option", "?")
        model = exp_cfg.get("model_type", "?")
        num_clients = exp_cfg.get("num_clients", "?")
        beta = exp_cfg.get("beta", "?")
        use_pe = exp_cfg.get("use_pe", False)
        fulltraining = exp_cfg.get("fulltraining_flag", False)

        method = METHOD_NAMES.get(data_loading, data_loading)
        if data_loading == "full" and num_clients == 1:
            method = "Centralised"
        elif data_loading == "full":
            method = "FedProp-Full"
        if fulltraining and data_loading == "full":
            method = "FedProp-Full"
        if use_pe:
            method = f"{method}+PE"
        hop = exp_cfg.get("hop", 1)
        if hop == 2:
            method = f"{method} (hop=2)"

        rows.append({
            "dataset": dataset,
            "method": method,
            "model": model,
            "num_clients": num_clients,
            "beta": beta,
            "use_pe": use_pe,
            "fulltraining_flag": fulltraining,
            "avg_global": summary.get("average_global_result", float("nan")),
            "avg_client": summary.get("average_client_result", float("nan")),
            "std_global": summary.get("std_global", float("nan")),
            "std_client": summary.get("std_client", float("nan")),
            "duration_s": duration.get("seconds", 0),
            "source": r.get("_source_file", ""),
        })
    return pd.DataFrame(rows)


def format_accuracy(mean: float, std: float) -> str:
    """Format accuracy as percentage: 80.8 ± 1.4"""
    return f"{mean*100:.1f} ± {std*100:.1f}"


def format_accuracy_3dec(mean: float, std: float) -> str:
    """Format accuracy as decimal: 0.808 ± 0.014"""
    return f"{mean:.3f} ± {std:.3f}"


def build_accuracy_table(df: pd.DataFrame, datasets: list[str] | None = None,
                         methods: list[str] | None = None) -> pd.DataFrame:
    """Build a paper-ready accuracy table (Table 1 / Table 2 format).

    Rows = methods, Columns = (dataset, β) pairs.
    Values = mean ± std.
    """
    if datasets:
        df = df[df["dataset"].isin(datasets)]
    if methods:
        df = df[df["method"].isin(methods)]

    if df.empty:
        return pd.DataFrame()

    # Group by (method, dataset, beta) — should be one row per group
    grouped = df.groupby(["method", "dataset", "beta"]).agg(
        mean=("avg_global", "mean"),
        std=("std_global", "mean"),
    ).reset_index()

    # Pivot: columns = (dataset, beta)
    records = []
    for method in methods or df["method"].unique():
        row = {"Method": method}
        for _, grp_row in grouped[grouped["method"] == method].iterrows():
            ds = grp_row["dataset"]
            beta = grp_row["beta"]
            beta_str = BETA_LABELS.get(beta, f"β={beta}")
            col = f"{ds}\n{beta_str}"
            row[col] = format_accuracy_3dec(grp_row["mean"], grp_row["std"])
        records.append(row)

    return pd.DataFrame(records)


# --- Dataset metadata ---

HOMOPHILY_RATIOS = {
    "Texas": 0.11,
    "Wisconsin": 0.20,
}


def add_homophily_metadata(table: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    """Prepend homophily ratio rows above the accuracy table for heterophilic datasets."""
    rows = []
    for ds in datasets:
        h = HOMOPHILY_RATIOS.get(ds)
        if h is not None:
            row = {"": f"Edge homophily ({ds})"}
            for col in table.columns[1:]:  # skip "Method" column
                row[col] = f"h={h:.2f}"
            rows.append(row)
    if rows:
        meta_df = pd.DataFrame(rows)
        return pd.concat([meta_df, table], ignore_index=True)
    return table


def bold_best(df: pd.DataFrame) -> pd.DataFrame:
    """Bold the best value in each column of a formatted table."""
    # Parse numeric values from "mean ± std" strings
    result = df.copy()
    for col in result.columns:
        if col == "Method":
            continue
        vals = []
        for v in result[col]:
            if pd.isna(v) or v == "" or v == "—":
                vals.append(float("-inf"))
            else:
                try:
                    vals.append(float(v.split("±")[0].strip()))
                except (ValueError, IndexError):
                    vals.append(float("-inf"))

        best_idx = int(np.argmax(vals))
        for i, v in enumerate(result[col]):
            if i == best_idx and not pd.isna(v) and v != "":
                result.at[result.index[i], col] = f"**{v}**"
    return result


# --- Figure generators ---

def plot_recovery_bars(df: pd.DataFrame, datasets: list[str]) -> plt.Figure:
    """Figure 2: Recovery bar plot (Zero → Diff → Full → Centralised)."""
    methods_order = ["FedProp-Zero", "FedProp (Diff)", "FedProp-Full", "Centralised"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_df = df[(df["dataset"] == ds) & (df["beta"] == 10000)]
        means = []
        stds = []
        labels = []
        for m in methods_order:
            row = ds_df[ds_df["method"] == m]
            if not row.empty:
                means.append(row["avg_global"].values[0] * 100)
                stds.append(row["std_global"].values[0] * 100)
                labels.append(m)
            else:
                means.append(0)
                stds.append(0)
                labels.append(m)

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"],
               alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(ds)
        ax.set_ylabel("Test Accuracy (%)" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

        # Annotate gap closed
        if len(means) >= 3:
            zero_acc = means[0]
            diff_acc = means[1]
            full_acc = means[2]
            gap = full_acc - zero_acc
            if gap > 0:
                pct = (diff_acc - zero_acc) / gap * 100
                ax.annotate(f"{pct:.0f}% gap closed",
                            xy=(1, diff_acc), xytext=(1.5, diff_acc + 2),
                            fontsize=7, color="#3498db",
                            arrowprops=dict(arrowstyle="->", color="#3498db"))

    fig.tight_layout()
    return fig


def plot_beta_sweep(df: pd.DataFrame) -> plt.Figure:
    """Figure 3: Accuracy vs β (Dirichlet partition severity)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    methods = ["FedProp-Zero", "FedProp (Adj)", "FedProp (Diff)", "FedProp-Full"]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

    for method, color in zip(methods, colors):
        mdf = df[df["method"] == method].sort_values("beta", ascending=False)
        if mdf.empty:
            continue
        betas = mdf["beta"].values
        means = mdf["avg_global"].values * 100
        stds = mdf["std_global"].values * 100

        ax.plot(betas, means, "o-", color=color, label=method, linewidth=2)
        ax.fill_between(betas, means - stds, means + stds, color=color, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet β (higher = more IID)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs Partition Severity (Cora)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_client_scaling(df: pd.DataFrame) -> plt.Figure:
    """Figure 4: Accuracy vs number of clients K."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    methods = ["FedProp-Zero", "FedProp (Diff)"]
    colors = ["#e74c3c", "#3498db"]

    for ax, (beta, beta_label) in zip([ax1, ax2], [(10000, "IID (β=10k)"), (10, "non-IID (β=10)")]):
        for method, color in zip(methods, colors):
            mdf = df[(df["method"] == method) & (df["beta"] == beta)].sort_values("num_clients")
            if mdf.empty:
                continue
            ks = mdf["num_clients"].values
            means = mdf["avg_global"].values * 100
            stds = mdf["std_global"].values * 100

            ax.plot(ks, means, "o-", color=color, label=method, linewidth=2)
            ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.15)

        ax.set_xlabel("Number of Clients (K)")
        ax.set_title(beta_label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Test Accuracy (%)")
    fig.suptitle("Client-Count Scaling (Cora)")
    fig.tight_layout()
    return fig


def plot_t_sweep(df: pd.DataFrame, datasets: list[str]) -> plt.Figure:
    """Figure A3: Accuracy vs propagation iterations T."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#3498db", "#e74c3c"]

    for ds, color in zip(datasets, colors):
        ds_df = df[(df["dataset"] == ds) & (df["beta"] == 10000)]
        if ds_df.empty:
            continue
        # T values come from different configs; extract from source file names or config
        ts = sorted(ds_df["num_iterations"].unique()) if "num_iterations" in ds_df.columns else [10, 25, 50, 100]
        means = ds_df.groupby("num_iterations")["avg_global"].mean().values * 100
        stds = ds_df.groupby("num_iterations")["std_global"].mean().values * 100
        ax.plot(ts, means, "o-", color=color, label=ds, linewidth=2)
        ax.fill_between(ts, means - stds, means + stds, color=color, alpha=0.15)

    ax.set_xlabel("Propagation Iterations (T)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Propagation Depth Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# --- Communication cost estimation ---

MODEL_PARAMS = {
    ("GCN", "Cora"): 23_478,
    ("GCN", "Citeseer"): 15_860,
    ("GCN", "Pubmed"): 24_598,
    ("GCN_arxiv", "ogbn-arxiv"): 334_218,
    ("GAT", "Cora"): 120_614,
    ("GAT", "Citeseer"): 112_806,
    ("GAT", "Pubmed"): 121_582,
}

BYTES_PER_PARAM = 4  # float32


def estimate_comm_cost(model_key: tuple, rounds: int, k: int = 10,
                       pretrain_mb: float = 0) -> float:
    """Estimate total communication in MB per client."""
    params = MODEL_PARAMS.get(model_key, 0)
    per_round = params * BYTES_PER_PARAM * 2 / (1024 ** 2)  # upload + download
    total = pretrain_mb + per_round * rounds
    return total


# --- Main ---

def show_status():
    """Show what result data exists."""
    print("\nExperiment Data Status")
    print("-" * 50)
    print("Track A: fedprop")
    for rid in ["R1", "R1b", "R4", "R5", "R6", "A1", "A2", "A3", "A4"]:
        results = collect_json_results(rid)
        if results:
            df = extract_summary_rows(results)
            methods = df["method"].unique()
            print(f"  {rid:4s}  {len(results):3d} files  methods: {', '.join(methods)}")
        else:
            print(f"  {rid:4s}  no data")
    print("\nTrack B: baselines")
    baseline_results = collect_baseline_results()
    if baseline_results:
        by_baseline = defaultdict(int)
        for r in baseline_results:
            by_baseline[r.get("baseline", "unknown")] += 1
        for name, count in sorted(by_baseline.items()):
            print(f"  {name:8s} {count:3d} normalized file(s)")
    elif BASELINES_DIR.is_dir():
        print("  workspace exists, no normalized baseline JSON yet")
    else:
        print("  no baseline workspace")
    print()


def produce_table(result_id: str, output_dir: Path):
    """Produce table for a given result."""
    results = collect_json_results(result_id)
    if not results:
        print(f"No data for {result_id}")
        return

    df = extract_summary_rows(results)
    output_dir.mkdir(parents=True, exist_ok=True)

    if result_id == "R1":
        table = build_accuracy_table(
            df,
            datasets=["Cora", "Citeseer", "Pubmed", "ogbn-arxiv"],
            methods=["Centralised", "FedProp-Full", "FedProp-Zero",
                     "FedProp (Adj)", "FedProp (Diff)"],
        )
        name = "table1_gcn_accuracy"
    elif result_id == "R1b":
        table = build_accuracy_table(
            df,
            datasets=["Cora", "Citeseer", "Pubmed"],
            methods=["Centralised", "FedProp-Full", "FedProp-Zero",
                     "FedProp (Adj)", "FedProp (Diff)"],
        )
        name = "table2_gat_accuracy"
    elif result_id == "R6":
        table = build_accuracy_table(
            df,
            datasets=["Texas", "Wisconsin"],
            methods=["Centralised", "FedProp-Full", "FedProp-Zero",
                     "FedProp (Adj)", "FedProp (Diff)"],
        )
        table = add_homophily_metadata(table, ["Texas", "Wisconsin"])
        name = "table3_heterophilic"
    elif result_id == "A1":
        table = build_accuracy_table(
            df,
            datasets=["Computers", "Photo"],
            methods=["Centralised", "FedProp-Full", "FedProp-Zero",
                     "FedProp (Adj)", "FedProp (Diff)"],
        )
        name = "tableA1_amazon"
    elif result_id == "A2":
        table = build_accuracy_table(
            df,
            datasets=["Cora", "Citeseer", "Pubmed"],
            methods=["FedProp (Adj)", "FedProp (Adj)+PE",
                     "FedProp (Diff)", "FedProp (Diff)+PE"],
        )
        name = "tableA2_pe_ablation"
    elif result_id == "A4":
        table = build_accuracy_table(
            df,
            datasets=["Cora", "Citeseer"],
            methods=["FedProp-Zero", "FedProp (Diff)",
                     "FedProp-Zero (hop=2)", "FedProp (Diff) (hop=2)"],
        )
        name = "tableA4_hop_radius"
    else:
        # Generic: just output what we have
        table = build_accuracy_table(df)
        name = f"table_{result_id.lower()}"

    if table.empty:
        print(f"No table data for {result_id}")
        return

    table = bold_best(table)

    csv_path = output_dir / f"{name}.csv"
    md_path = output_dir / f"{name}.md"

    table.to_csv(csv_path, index=False)
    table.to_markdown(md_path, index=False)
    print(f"  {result_id}: {csv_path}")
    print(f"  {result_id}: {md_path}")


def produce_figure(figure_num: int, output_dir: Path):
    """Produce a specific figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if figure_num == 2:
        results = collect_json_results("R1")
        if not results:
            print("No R1 data for Figure 2")
            return
        df = extract_summary_rows(results)
        fig = plot_recovery_bars(df, ["Cora", "Citeseer", "Pubmed"])
        path = output_dir / "fig2_recovery_bars.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure 2: {path}")

    elif figure_num == 3:
        results = collect_json_results("R4")
        if not results:
            print("No R4 data for Figure 3")
            return
        df = extract_summary_rows(results)
        fig = plot_beta_sweep(df)
        path = output_dir / "fig3_beta_sweep.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure 3: {path}")

    elif figure_num == 4:
        results = collect_json_results("R5")
        if not results:
            print("No R5 data for Figure 4")
            return
        df = extract_summary_rows(results)
        fig = plot_client_scaling(df)
        path = output_dir / "fig4_client_scaling.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure 4: {path}")

    else:
        print(f"Figure {figure_num} not yet implemented or requires post-hoc data")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze split FedProp/baseline experiment results")
    parser.add_argument("--result", type=str, default=None,
                        help="Result ID(s) to produce table for (e.g., R1,R1b)")
    parser.add_argument("--figure", type=int, nargs="*", default=None,
                        help="Figure number(s) to produce (2, 3, 4)")
    parser.add_argument("--all", action="store_true",
                        help="Produce all tables and figures")
    parser.add_argument("--status", action="store_true",
                        help="Show what data exists")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: experiments/output)")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else OUTPUT_DIR

    if args.status:
        show_status()
        return

    table_ids = []
    if args.all:
        table_ids = ["R1", "R1b", "R6", "A1", "A2", "A3", "A4"]
        figures = [2, 3, 4]
    else:
        if args.result:
            table_ids = [r.strip() for r in args.result.split(",")]
        figures = args.figure or []

    if table_ids:
        print("Producing tables...")
        for rid in table_ids:
            produce_table(rid, output_dir)

    if figures:
        print("Producing figures...")
        for fig_num in figures:
            produce_figure(fig_num, output_dir)

    if not table_ids and not figures:
        parser.error("Specify --result, --figure, --all, or --status")


if __name__ == "__main__":
    main()
