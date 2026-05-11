#!/usr/bin/env python3
"""Figure 1 — Intrinsic MSE vs. downstream GCN accuracy on Cora.

Data: Cora, beta=10000 (intrinsic MSE from Phase 1 / Table S3)
      Cora, GCN, beta=1 (downstream accuracy from Table 3.1)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linestyle": "--",
})

# ── Data ──────────────────────────────────────────────────────────────────────
operators = ["APPNP", "Chebyshev", "Diffusion", "Adjacency", "Asym. RW"]

mse_mean = np.array([0.01299, 0.01325, 0.01334, 0.01400, 0.01602])
mse_std  = np.array([0.00002, 0.00002, 0.00002, 0.00002, 0.00004])

acc_mean = np.array([0.7672, 0.7840, 0.7622, 0.7738, 0.7690])
acc_std  = np.array([0.0133, 0.0096, 0.0139, 0.0127, 0.0062])

# ── Styling ───────────────────────────────────────────────────────────────────
COLORS = {
    "APPNP":     "#2A9D8F",
    "Chebyshev": "#E76F51",
    "Diffusion": "#264653",
    "Adjacency": "#E9C46A",
    "Asym. RW":  "#F4A261",
}
MARKERS = {
    "APPNP":     "o",
    "Chebyshev": "s",
    "Diffusion": "^",
    "Adjacency": "D",
    "Asym. RW":  "v",
}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.8, 3.0))

for op, mx, ms, ay, ys in zip(operators, mse_mean, mse_std, acc_mean, acc_std):
    color  = COLORS[op]
    marker = MARKERS[op]
    zorder = 5 if op in ("Chebyshev", "APPNP") else 3

    # error bars first so markers sit on top
    ax.errorbar(mx, ay, xerr=ms, yerr=ys,
                fmt="none", ecolor=color, elinewidth=1.0,
                capsize=3.0, capthick=1.0, alpha=0.55, zorder=zorder - 1)

    ax.scatter(mx, ay, color=color, marker=marker,
               s=70, zorder=zorder,
               edgecolors="white", linewidths=0.8,
               label=op)

# ── Annotation: double-headed arrow linking APPNP and Chebyshev ──────────────
# Arrow placed in the crowded left cluster; text goes into the empty space right
ax.annotate("",
            xy=(0.01299, 0.7672),      # APPNP
            xytext=(0.01325, 0.7840),  # Chebyshev
            arrowprops=dict(arrowstyle="<->", color="#555",
                            lw=0.9, mutation_scale=10,
                            connectionstyle="arc3,rad=-0.30"),
            annotation_clip=False)
# Text lives in the open region between left cluster and Adjacency
ax.text(0.01370, 0.7800,
        "MSE rank $\\neq$\naccuracy rank",
        fontsize=7.5, color="#444",
        ha="left", va="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel(r"Intrinsic MSE on Cora ($\beta$=10 000)  $\downarrow$ better", labelpad=5)
ax.set_ylabel("Downstream GCN accuracy\n"
              r"on Cora ($\beta$=1)  $\uparrow$ better", labelpad=5)

# Limits — give left side breathing room; extend right for the annotation text
ax.set_xlim(mse_mean.min() - 0.0012, mse_mean.max() + 0.0008)
ax.set_ylim(acc_mean.min() - acc_std.max() - 0.003,
            acc_mean.max() + acc_std.max() + 0.004)

# ── Legend — placed below the axes to avoid any overlap ───────────────────────
ax.legend(loc="upper center",
          bbox_to_anchor=(0.5, -0.22),
          ncol=3,
          markerscale=1.1,
          handlelength=1.2,
          handletextpad=0.4,
          columnspacing=0.8,
          fontsize=8)

fig.subplots_adjust(bottom=0.28)   # make room for the below-axes legend

pdf_path = os.path.join(OUT_DIR, "fig_01_intrinsic_vs_downstream.pdf")
png_path = os.path.join(OUT_DIR, "fig_01_intrinsic_vs_downstream.png")
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=300)
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
