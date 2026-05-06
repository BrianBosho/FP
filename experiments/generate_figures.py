#!/usr/bin/env python3
"""Publication-ready figures for the FedProp paper (R1 GCN and R1b GAT).

Method structure (per paper spec):
  - Zero-hop  (lower bound)
  - Full      (upper bound / oracle)
  - Adj       (no PE)
  - Adj+PE    (PE only for adjacency, GAT only)
  - Diff      (no PE)

Figures produced:
  fig_R1_bars.pdf          — GCN: recovery from lower→upper bound
  fig_R1b_bars.pdf         — GAT: same, with Adj+PE
  fig_beta_sweep.pdf       — Accuracy vs heterogeneity (β sweep)
  fig_pe_effect.pdf        — PE gain: Adj+PE minus Adj (R1b only)
  fig_convergence.pdf      — FL round convergence curves (Cora, GAT)

Usage:
    python experiments/generate_figures.py
    python experiments/generate_figures.py --fig bars beta pe conv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FIG_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Design constants ────────────────────────────────────────────────────────

# Okabe-Ito colorblind-safe
C = {
    "zero_hop":   "#E69F00",  # amber
    "adjacency":  "#56B4E9",  # sky blue
    "adj_pe":     "#0072B2",  # deep blue
    "diffusion":  "#009E73",  # teal
    "full":       "#CC79A7",  # mauve
}
LABELS = {
    "zero_hop":  "Zero-hop",
    "adjacency": "Adj",
    "adj_pe":    "Adj+PE",
    "adj":       "Adj",
    "diff":      "Diff",
    "diffusion": "Diff",
    "full":      "Full (oracle)",
}

DATASETS = ["Cora", "Citeseer", "Pubmed"]
BETAS = [10000, 10]
BETA_LABELS = {10000: "IID (β=10k)", 10: "non-IID (β=10)"}
MIN_REPS = 5

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8.5,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "axes.grid.axis":   "y",
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_csv(track: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"{track}_consolidated.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run consolidate_results.py first — missing {path}")
    df = pd.read_csv(path)
    df["beta"] = df["beta"].astype(int)
    return df


def get_val(df: pd.DataFrame, dataset: str, prop: str, beta: int,
            hop: int, use_pe: bool):
    """Return (mean_acc_pct, std_acc_pct) or (None, None) if missing/incomplete."""
    mask = (
        (df["dataset"] == dataset)
        & (df["propagation"] == prop)
        & (df["beta"] == beta)
        & (df["hop"] == hop)
        & (df["use_pe"] == use_pe)
    )
    r = df[mask]
    if len(r) == 0 or r.iloc[0]["n_reps"] < MIN_REPS:
        return None, None
    ri = r.iloc[0]
    return ri["mean_acc"] * 100, ri["std_acc"] * 100


def load_round_history(track: str, dataset: str, prop: str, beta: int,
                       use_pe: bool) -> list[list[float]]:
    """Load per-round avg_client_val_acc from all reps of a config."""
    hop = 1 if track == "R1" else 2
    suffix = "pe" if use_pe else "nope"
    curves = []
    for search_dir in sorted(RESULTS_DIR.glob(f"{track}_{dataset.lower()}_{suffix}*")):
        for jpath in sorted(search_dir.rglob("results_*.json")):
            try:
                data = json.loads(jpath.read_text())
                cfg = data["experiment_config"]
                if (cfg.get("data_loading_option") != prop
                        or float(cfg.get("beta", 0)) != beta
                        or int(cfg.get("hop", 0)) != hop):
                    continue
                for rep in data.get("rounds", []):
                    hist = rep.get("round_history", [])
                    curve = [h["avg_client_val_acc"] for h in hist
                             if "avg_client_val_acc" in h]
                    if len(curve) > 5:
                        curves.append(curve)
            except Exception:
                pass
    return curves[:10]


# ─── Figure: Recovery bar charts ─────────────────────────────────────────────

def _draw_bounds(ax, y_low, y_high, x_min, x_max):
    """Draw dashed horizontal lines for lower/upper bounds."""
    for y, color, label in [
        (y_low,  C["zero_hop"], "Zero-hop"),
        (y_high, C["full"],     "Full oracle"),
    ]:
        if y is not None:
            ax.axhline(y, color=color, linewidth=1.2, linestyle="--", alpha=0.8)


def plot_bars(track: str, save: bool = True):
    """Grouped bar chart: one subplot per dataset, showing all 5 methods.

    Each method group has two bars: IID (solid) and non-IID (hatched/lighter).
    Zero-hop and Full-oracle are shown as dashed reference lines (not bars).
    """
    df = load_csv(track)
    hop = 1 if track == "R1" else 2
    use_gat = track == "R1b"

    # Method groups for bars (zero_hop and full are reference lines)
    if use_gat:
        bar_methods = [
            ("adjacency", False, "adj",    C["adjacency"]),
            ("adjacency", True,  "adj_pe", C["adj_pe"]),
            ("diffusion",  False, "diff",   C["diffusion"]),
        ]
    else:
        bar_methods = [
            ("adjacency", False, "adj",  C["adjacency"]),
            ("diffusion",  False, "diff", C["diffusion"]),
        ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    model_name = "GAT" if use_gat else "GCN"
    fig.suptitle(
        f"{model_name} Test Accuracy — Methods between Lower and Upper Bound",
        fontsize=11, y=1.01
    )

    n_groups = len(bar_methods)
    bar_w = 0.32
    iid_offset = -bar_w / 2
    niid_offset = bar_w / 2

    for ax, dataset in zip(axes, DATASETS):
        ax.set_title(dataset, fontweight="bold")

        x = np.arange(n_groups)

        # --- Reference lines: zero-hop and full ---
        z_iid,  _  = get_val(df, dataset, "zero_hop", 10000, hop, False)
        z_niid, _  = get_val(df, dataset, "zero_hop", 10,    hop, False)
        f_iid,  _  = get_val(df, dataset, "full",     10000, hop, False)
        f_niid, _  = get_val(df, dataset, "full",     10,    hop, False)

        # Draw shaded band between zero and full for IID
        x_span = np.array([-0.5, n_groups - 0.5])
        if z_iid is not None and f_iid is not None:
            ax.fill_between(x_span, z_iid, f_iid,
                            color="#f0f0f0", zorder=0, label="_nolegend_")
            ax.axhline(z_iid, color=C["zero_hop"], linewidth=1.5,
                       linestyle="--", alpha=0.9, zorder=1)
            ax.axhline(f_iid, color=C["full"],     linewidth=1.5,
                       linestyle=":",  alpha=0.9, zorder=1)
        elif z_iid is not None:
            ax.axhline(z_iid, color=C["zero_hop"], linewidth=1.5,
                       linestyle="--", alpha=0.9, zorder=1)
        if f_iid is not None:
            ax.axhline(f_iid, color=C["full"], linewidth=1.5,
                       linestyle=":", alpha=0.9, zorder=1)

        # --- Bars ---
        for g_idx, (prop, pe, key, color) in enumerate(bar_methods):
            # IID bar (solid)
            acc_iid, std_iid = get_val(df, dataset, prop, 10000, hop, pe)
            # non-IID bar (lighter / hatched)
            acc_niid, std_niid = get_val(df, dataset, prop, 10, hop, pe)

            xi_iid  = x[g_idx] + iid_offset
            xi_niid = x[g_idx] + niid_offset

            if acc_iid is not None:
                ax.bar(xi_iid, acc_iid, bar_w, color=color, alpha=1.0,
                       edgecolor="white", linewidth=0.4, zorder=3)
                ax.errorbar(xi_iid, acc_iid, yerr=std_iid,
                            fmt="none", color="black", capsize=2.5,
                            linewidth=0.9, zorder=4)
            else:
                ax.text(xi_iid, ax.get_ylim()[0] + 0.5 if ax.get_ylim()[0] else 50,
                        "—", ha="center", va="bottom", fontsize=7, color="gray")

            if acc_niid is not None:
                ax.bar(xi_niid, acc_niid, bar_w, color=color, alpha=0.55,
                       edgecolor="white", linewidth=0.4,
                       hatch="////", zorder=3)
                ax.errorbar(xi_niid, acc_niid, yerr=std_niid,
                            fmt="none", color="black", capsize=2.5,
                            linewidth=0.9, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[key] for _, _, key, _ in bar_methods],
                           fontsize=8)
        ax.set_ylabel("Test Accuracy (%)")

        # Y limits: tight around data
        all_vals = []
        for prop, pe, key, _ in bar_methods:
            for beta in BETAS:
                v, _ = get_val(df, dataset, prop, beta, hop, pe)
                if v is not None:
                    all_vals.append(v)
        for v in [z_iid, z_niid, f_iid, f_niid]:
            if v is not None:
                all_vals.append(v)
        if all_vals:
            margin = 3
            ax.set_ylim(max(0, min(all_vals) - margin),
                        min(100, max(all_vals) + margin))

    # Shared legend
    legend_handles = []
    # Method color patches
    for _, _, key, color in bar_methods:
        legend_handles.append(
            mpatches.Patch(facecolor=color, alpha=1.0, label=f"{LABELS[key]} — IID")
        )
        legend_handles.append(
            mpatches.Patch(facecolor=color, alpha=0.55, hatch="////",
                           label=f"{LABELS[key]} — non-IID")
        )
    # Reference line proxies
    legend_handles += [
        plt.Line2D([0], [0], color=C["zero_hop"], linewidth=1.5, linestyle="--",
                   label="Zero-hop (lower bound)"),
        plt.Line2D([0], [0], color=C["full"],     linewidth=1.5, linestyle=":",
                   label="Full oracle (upper bound)"),
    ]

    ncol = 3 if use_gat else 3
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=ncol, bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=8)
    plt.tight_layout()

    fname = f"fig_{'R1b' if use_gat else 'R1'}_bars"
    for ext in [".pdf", ".png"]:
        out = FIG_DIR / (fname + ext)
        plt.savefig(out)
        if ext == ".pdf":
            print(f"Saved: {out}")
    plt.close()


# ─── Figure: Beta sweep (accuracy vs heterogeneity) ──────────────────────────

def plot_beta_sweep(save: bool = True):
    """Line plot: accuracy vs β (10000 → 10 → 1) per method and dataset."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("Accuracy vs. Non-IID Heterogeneity (β sweep)", fontsize=11)

    BETAS_SWEEP = [10000, 10, 1]
    x_ticks = ["IID\n(β=10k)", "non-IID\n(β=10)", "very\nnon-IID\n(β=1)"]

    for row_idx, (track, model_name) in enumerate([("R1", "GCN"), ("R1b", "GAT")]):
        df = load_csv(track)
        hop = 1 if track == "R1" else 2

        sweep_methods = [
            ("zero_hop",  False, "zero_hop",  C["zero_hop"]),
            ("full",      False, "full",      C["full"]),
            ("adjacency", False, "adjacency", C["adjacency"]),
            ("diffusion", False, "diffusion", C["diffusion"]),
        ]
        if track == "R1b":
            sweep_methods.insert(3, ("adjacency", True, "adj_pe", C["adj_pe"]))

        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx][col_idx]
            ax.set_title(f"{model_name} — {dataset}", fontsize=9)

            for prop, pe, key, color in sweep_methods:
                accs, stds, xs = [], [], []
                for beta in BETAS_SWEEP:
                    v, s = get_val(df, dataset, prop, beta, hop, pe)
                    if v is not None:
                        accs.append(v)
                        stds.append(s)
                        xs.append(BETAS_SWEEP.index(beta))

                if len(xs) >= 2:
                    ls = "--" if key in ("zero_hop", "full") else "-"
                    ax.plot(xs, accs, marker="o", color=color, linewidth=1.5,
                            markersize=4, linestyle=ls, label=LABELS[key])
                    ax.fill_between(xs,
                                    [a - s for a, s in zip(accs, stds)],
                                    [a + s for a, s in zip(accs, stds)],
                                    color=color, alpha=0.12)
                elif len(xs) == 1:
                    ax.scatter(xs[0], accs[0], color=color, s=30,
                               marker="D", zorder=5)

            ax.set_xticks(range(len(BETAS_SWEEP)))
            ax.set_xticklabels(x_ticks, fontsize=7)
            ax.set_ylabel("Test Accuracy (%)", fontsize=8)

    # Legend from bottom-left subplot
    handles, labels = axes[1][0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)
    plt.tight_layout()

    for ext in [".pdf", ".png"]:
        out = FIG_DIR / ("fig_beta_sweep" + ext)
        plt.savefig(out)
        if ext == ".pdf":
            print(f"Saved: {out}")
    plt.close()


# ─── Figure: PE effect (Adj+PE minus Adj) ────────────────────────────────────

def plot_pe_effect(save: bool = True):
    """Bar chart of accuracy gain from PE: (Adj+PE) − Adj per dataset and beta."""
    df = load_csv("R1b")
    hop = 2

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    fig.suptitle("Accuracy Gain from Positional Encoding (Adj+PE − Adj)",
                 fontsize=11)

    bar_w = 0.35
    x = np.array([0, 1])  # IID, non-IID

    for ax, dataset in zip(axes, DATASETS):
        ax.set_title(dataset, fontweight="bold")

        gains, stds, colors = [], [], []
        for beta in BETAS:
            a_acc, a_std = get_val(df, dataset, "adjacency", beta, hop, False)
            p_acc, p_std = get_val(df, dataset, "adjacency", beta, hop, True)

            if a_acc is not None and p_acc is not None:
                gain = p_acc - a_acc
                # Approximate combined std (conservative: RSS)
                comb_std = np.sqrt(a_std ** 2 + p_std ** 2) if (a_std and p_std) else 0
                gains.append(gain)
                stds.append(comb_std)
                colors.append(C["adj_pe"] if gain >= 0 else "#D55E00")
            else:
                gains.append(0)
                stds.append(0)
                colors.append("lightgray")

        bars = ax.bar(x, gains, bar_w * 1.5, color=colors,
                      edgecolor="white", linewidth=0.4)
        ax.errorbar(x, gains, yerr=stds, fmt="none",
                    color="black", capsize=3, linewidth=0.9)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

        # Annotate bar values
        for xi, g in zip(x, gains):
            va = "bottom" if g >= 0 else "top"
            offset = 0.1 if g >= 0 else -0.1
            ax.text(xi, g + offset, f"{g:+.1f}%", ha="center", va=va, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [BETA_LABELS[b].replace("(", "\n(") for b in BETAS], fontsize=8
        )
        ax.set_ylabel("Accuracy Gain (%)")
        max_abs = max(abs(g) for g in gains) if gains else 3
        ax.set_ylim(-max_abs - 2, max_abs + 2)

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        out = FIG_DIR / ("fig_pe_effect" + ext)
        plt.savefig(out)
        if ext == ".pdf":
            print(f"Saved: {out}")
    plt.close()


# ─── Figure: FL convergence curves ───────────────────────────────────────────

def plot_convergence(save: bool = True):
    """Per-round validation accuracy: all methods on Cora, IID vs non-IID."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("FL Training Convergence — GAT, Cora (noPE)", fontsize=11)

    curve_methods = [
        ("zero_hop",  False, "zero_hop",  C["zero_hop"],  "--"),
        ("adjacency", False, "adjacency", C["adjacency"], "-"),
        ("diffusion",  False, "diffusion",  C["diffusion"],  "-"),
        ("full",      False, "full",      C["full"],      ":"),
    ]

    for beta, ax in zip(BETAS, axes):
        ax.set_title(BETA_LABELS[beta])
        any_data = False

        for prop, pe, key, color, ls in curve_methods:
            curves = load_round_history("R1b", "Cora", prop, beta, pe)
            if not curves:
                continue
            any_data = True
            min_len = min(len(c) for c in curves)
            arr = np.array([c[:min_len] for c in curves]) * 100
            mean_c = arr.mean(axis=0)
            std_c  = arr.std(axis=0)
            x = np.arange(len(mean_c))
            ax.plot(x, mean_c, color=color, linewidth=1.8, linestyle=ls,
                    label=LABELS[key])
            ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.12)

        ax.set_xlabel("FL Round")
        ax.set_ylabel("Avg. Client Val. Accuracy (%)")
        if ax.get_ylim()[0] < 0:
            ax.set_ylim(0)
        if not any_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")

    handles = [
        plt.Line2D([0], [0], color=C[k], linewidth=2, linestyle=ls,
                   label=LABELS[k])
        for k, ls in [("zero_hop", "--"), ("adjacency", "-"),
                      ("diffusion", "-"), ("full", ":")]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    plt.tight_layout()

    for ext in [".pdf", ".png"]:
        out = FIG_DIR / ("fig_convergence" + ext)
        plt.savefig(out)
        if ext == ".pdf":
            print(f"Saved: {out}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

FIG_MAP = {
    "bars":  lambda: [plot_bars("R1"), plot_bars("R1b")],
    "beta":  lambda: plot_beta_sweep(),
    "pe":    lambda: plot_pe_effect(),
    "conv":  lambda: plot_convergence(),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", nargs="+", choices=list(FIG_MAP.keys()),
                        default=list(FIG_MAP.keys()),
                        help="Which figures to generate (default: all)")
    args = parser.parse_args()

    for fig_name in args.fig:
        print(f"\n--- Generating: {fig_name} ---")
        try:
            FIG_MAP[fig_name]()
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback; traceback.print_exc()

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
