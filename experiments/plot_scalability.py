"""
Plot scalability figures from a canonical scalability_*.csv produced by run_experiments.py.

Usage:
  python experiments/plot_scalability.py results/summary/scalability_20240101_120000.csv
  python experiments/plot_scalability.py results/summary/scalability_20240101_120000.csv --out figures/

Outputs (in --out directory, default same dir as CSV):
  scalability_time.png    — wall-clock time vs. #clients
  scalability_memory.png  — peak GPU / CPU memory vs. #clients
  scalability_comm.png    — theoretical comm cost vs. #clients
  scalability_acc.png     — final + best accuracy vs. #clients
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    num_cols = [
        "Clients", "Beta", "Hop",
        "TotalTime[s]", "LoadTime[s]", "ShardTime[s]", "PartitionTime[s]",
        "PropTime[s]", "ActorInitTime[s]", "TrainTime[s]", "EvalTime[s]",
        "CommTime[s]", "CommCost[MB]", "ModelSize[MB]", "TotalParams",
        "PeakCPU[MB]", "PeakDriverGPU[MB]", "PeakActorGPU[MB]",
        "FinalAcc", "BestAcc",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _group_key(df: pd.DataFrame) -> list[str]:
    candidates = ["Dataset", "Model", "DataLoading", "Beta", "Hop", "UsePE"]
    return [c for c in candidates if c in df.columns and df[c].nunique() > 1]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def plot_scalability(csv_path: str, out_dir: str | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    df = _read_csv(csv_path)
    out = Path(out_dir) if out_dir else Path(csv_path).parent
    base = Path(csv_path).stem

    if "Clients" not in df.columns or df["Clients"].isna().all():
        print("CSV has no usable 'Clients' column — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    # Only rows with status==success or status==partial_failed (have real numbers)
    ok = df[df.get("Status", pd.Series(["success"] * len(df))).isin(
        ["success", "partial_failed"]
    )].copy() if "Status" in df.columns else df.copy()

    group_keys = _group_key(ok)
    groups = ok.groupby(group_keys) if group_keys else [(("all",), ok)]

    def _label(key) -> str:
        if isinstance(key, str):
            return key
        return " | ".join(str(k) for k in key)

    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ── 1) Time breakdown vs clients ──────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for ci, (key, grp) in enumerate(groups):
        grp = grp.sort_values("Clients")
        c = COLORS[ci % len(COLORS)]
        lbl = _label(key)
        time_phases = {
            "Load": "LoadTime[s]",
            "Shard": "ShardTime[s]",
            "Partition": "PartitionTime[s]",
            "Prop": "PropTime[s]",
            "ActorInit": "ActorInitTime[s]",
            "Train": "TrainTime[s]",
            "Eval": "EvalTime[s]",
        }
        plotted_total = False
        if "TotalTime[s]" in grp.columns and grp["TotalTime[s]"].notna().any():
            ax1.plot(grp["Clients"], grp["TotalTime[s]"], marker="o", label=f"{lbl} Total", color=c)
            plotted_total = True
        for phase_name, col in time_phases.items():
            if col in grp.columns and grp[col].notna().any():
                ax1.plot(grp["Clients"], grp[col], marker=".", linestyle="--",
                         label=f"{lbl} {phase_name}", alpha=0.7)
    ax1.set_xlabel("Number of Clients")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Wall-clock Time vs. Clients")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.4)
    _save(fig1, out / f"{base}_time.png")
    plt.close(fig1)

    # ── 2) Memory vs clients ───────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    mem_cols = {
        "PeakActorGPU": "PeakActorGPU[MB]",
        "PeakDriverGPU": "PeakDriverGPU[MB]",
        "PeakCPU": "PeakCPU[MB]",
    }
    for ci, (key, grp) in enumerate(groups):
        grp = grp.sort_values("Clients")
        c = COLORS[ci % len(COLORS)]
        lbl = _label(key)
        for mem_name, col in mem_cols.items():
            if col in grp.columns and grp[col].notna().any():
                ax2.plot(grp["Clients"], grp[col], marker="o",
                         label=f"{lbl} {mem_name}", linestyle="-" if "Actor" in mem_name else "--")
    ax2.set_xlabel("Number of Clients")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_title("Peak Memory vs. Clients")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.4)
    _save(fig2, out / f"{base}_memory.png")
    plt.close(fig2)

    # ── 3) Communication cost vs clients ──────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for ci, (key, grp) in enumerate(groups):
        grp = grp.sort_values("Clients")
        c = COLORS[ci % len(COLORS)]
        lbl = _label(key)
        if "CommCost[MB]" in grp.columns and grp["CommCost[MB]"].notna().any():
            ax3.plot(grp["Clients"], grp["CommCost[MB]"], marker="o", color=c, label=lbl)
    ax3.set_xlabel("Number of Clients")
    ax3.set_ylabel("Theoretical Communication Cost (MB)")
    ax3.set_title("Communication Cost vs. Clients")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.4)
    _save(fig3, out / f"{base}_comm.png")
    plt.close(fig3)

    # ── 4) Accuracy vs clients ─────────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    for ci, (key, grp) in enumerate(groups):
        grp = grp.sort_values("Clients")
        c = COLORS[ci % len(COLORS)]
        lbl = _label(key)
        if "FinalAcc" in grp.columns and grp["FinalAcc"].notna().any():
            ax4.plot(grp["Clients"], grp["FinalAcc"], marker="o", color=c, label=f"{lbl} Final")
        if "BestAcc" in grp.columns and grp["BestAcc"].notna().any():
            ax4.plot(grp["Clients"], grp["BestAcc"], marker="s", linestyle="--",
                     color=c, alpha=0.7, label=f"{lbl} Best")
    ax4.set_xlabel("Number of Clients")
    ax4.set_ylabel("Accuracy")
    ax4.set_title("Accuracy vs. Clients")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.4)
    _save(fig4, out / f"{base}_acc.png")
    plt.close(fig4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scalability figures from a scalability CSV.")
    parser.add_argument("csv", help="Path to scalability_*.csv")
    parser.add_argument("--out", default=None, help="Output directory (default: same as CSV)")
    args = parser.parse_args()
    plot_scalability(args.csv, args.out)


if __name__ == "__main__":
    main()
