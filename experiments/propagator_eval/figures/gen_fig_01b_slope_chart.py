#!/usr/bin/env python3
"""Figure 1b — Slope chart: intrinsic MSE rank vs downstream accuracy rank.

Rank inversions (crossing lines) make the decoupling immediately visible.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "APPNP":     "#2A9D8F",
    "Chebyshev": "#E76F51",
    "Diffusion": "#264653",
    "Adjacency": "#E9C46A",
    "Asym. RW":  "#F4A261",
}
OPS = ["APPNP", "Chebyshev", "Diffusion", "Adjacency", "Asym. RW"]

# ── Data ──────────────────────────────────────────────────────────────────────
# Intrinsic MSE rank (Cora, beta=10000) — 1 = lowest MSE = best
mse_rank = {"APPNP": 1, "Chebyshev": 2, "Diffusion": 3, "Adjacency": 4, "Asym. RW": 5}
mse_val  = {"APPNP": 0.01299, "Chebyshev": 0.01325, "Diffusion": 0.01334,
            "Adjacency": 0.01400, "Asym. RW": 0.01602}

# Downstream GCN accuracy rank (Cora, beta=1) — 1 = highest acc = best
acc_rank = {"Chebyshev": 1, "Adjacency": 2, "Asym. RW": 3, "APPNP": 4, "Diffusion": 5}
acc_val  = {"APPNP": 0.767, "Chebyshev": 0.784, "Diffusion": 0.762,
            "Adjacency": 0.774, "Asym. RW": 0.769}

# rank → y: rank 1 at top
def ry(r): return 6 - r

# ── Layout constants ──────────────────────────────────────────────────────────
XL, XR   = 0.22, 0.78     # x positions of left/right columns
Y_TOP    = 5.55            # header y
Y_BOT    = 0.55            # footer y
DOT_BIG  = 90
DOT_SML  = 55

fig, ax = plt.subplots(figsize=(4.2, 4.0))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 6.5)

# ── Subtle column backgrounds ─────────────────────────────────────────────────
for xc, alpha, color in [(XL, 0.06, "#aaaaaa"), (XR, 0.06, "#aaaaaa")]:
    ax.add_patch(mpatches.FancyBboxPatch(
        (xc - 0.14, Y_BOT - 0.2), 0.28, Y_TOP - Y_BOT + 0.15,
        boxstyle="round,pad=0.02", fc=color, ec="none", alpha=alpha, zorder=0))

# ── Lines first (behind dots) ─────────────────────────────────────────────────
for op in OPS:
    y1 = ry(mse_rank[op])
    y2 = ry(acc_rank[op])
    highlight = op in ("APPNP", "Chebyshev")
    ax.plot([XL, XR], [y1, y2],
            color=COLORS[op],
            lw=2.4 if highlight else 1.0,
            alpha=1.0 if highlight else 0.40,
            solid_capstyle="round",
            zorder=2)

# ── Dots with rank numbers ─────────────────────────────────────────────────────
for op in OPS:
    y1 = ry(mse_rank[op])
    y2 = ry(acc_rank[op])
    highlight = op in ("APPNP", "Chebyshev")
    sz = DOT_BIG if highlight else DOT_SML
    for xc, yc in [(XL, y1), (XR, y2)]:
        ax.scatter(xc, yc, color=COLORS[op], s=sz, zorder=5,
                   edgecolors="white", linewidths=0.9)
    for xc, yc, rk in [(XL, y1, mse_rank[op]), (XR, y2, acc_rank[op])]:
        ax.text(xc, yc, str(rk), ha="center", va="center",
                fontsize=6 if sz == DOT_SML else 6.5,
                color="white", fontweight="bold", zorder=6)

# ── Left labels (sorted by MSE rank) ──────────────────────────────────────────
for op in OPS:
    y = ry(mse_rank[op])
    highlight = op in ("APPNP", "Chebyshev")
    ax.text(XL - 0.05, y + 0.22, op,
            ha="right", va="center", fontsize=8.5, color=COLORS[op],
            fontweight="bold" if highlight else "normal",
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    ax.text(XL - 0.05, y - 0.18, f"MSE {mse_val[op]:.4f}",
            ha="right", va="center", fontsize=6.5, color="#999")

# ── Right labels (sorted by acc rank) ─────────────────────────────────────────
for op in OPS:
    y = ry(acc_rank[op])
    highlight = op in ("APPNP", "Chebyshev")
    ax.text(XR + 0.05, y + 0.22, op,
            ha="left", va="center", fontsize=8.5, color=COLORS[op],
            fontweight="bold" if highlight else "normal",
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    ax.text(XR + 0.05, y - 0.18, f"acc {acc_val[op]:.3f}",
            ha="left", va="center", fontsize=6.5, color="#999")

# ── Rank-change callouts for the two key operators ────────────────────────────
# APPNP: drops from 1 → 4  (midpoint of its line)
ax.annotate("drops\n3 ranks",
            xy=(0.50, (ry(1) + ry(4)) / 2),
            fontsize=7, color=COLORS["APPNP"], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS["APPNP"],
                      lw=0.7, alpha=0.92))

# Chebyshev: rises from 2 → 1 (close together, annotate above)
ax.annotate("rises\n1 rank",
            xy=(0.50, (ry(2) + ry(1)) / 2 + 0.1),
            fontsize=7, color=COLORS["Chebyshev"], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS["Chebyshev"],
                      lw=0.7, alpha=0.92))

# ── Column headers ─────────────────────────────────────────────────────────────
ax.text(XL, Y_TOP + 0.35, "Intrinsic MSE",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#222")
ax.text(XL, Y_TOP + 0.05, "(Cora, β=10k)  ↓ better",
        ha="center", va="bottom", fontsize=7.5, color="#666", style="italic")

ax.text(XR, Y_TOP + 0.35, "Downstream Acc.",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#222")
ax.text(XR, Y_TOP + 0.05, "(Cora GCN, β=1)  ↑ better",
        ha="center", va="bottom", fontsize=7.5, color="#666", style="italic")

# Thin vertical rules under headers
for xc in (XL, XR):
    ax.plot([xc, xc], [Y_BOT - 0.1, Y_TOP - 0.05],
            color="#ddd", lw=0.7, zorder=1)

# ── Footer note ───────────────────────────────────────────────────────────────
ax.text(0.50, Y_BOT - 0.35,
        "Rank 1 = best.  Crossing lines indicate rank inversion.",
        ha="center", va="top", fontsize=7, style="italic", color="#888")

fig.tight_layout(pad=0.3)

for ext in ("pdf", "png"):
    p = os.path.join(OUT_DIR, f"fig_01b_slope_chart.{ext}")
    fig.savefig(p, dpi=300)
    print(f"Saved: {p}")
