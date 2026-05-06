#!/usr/bin/env python3
"""Generate paper tables for FedProp (R1 GCN and R1b GAT).

Method structure (per user spec):
  Row 1  — Zero-hop     (lower bound baseline, no propagation)
  Row 2  — Full         (oracle upper bound, full graph access)
  Row 3  — Adj          (adjacency propagation, no PE)
  Row 4  — Adj+PE       (adjacency + positional encoding) [GAT only]
  Row 5  — Diff         (diffusion propagation, no PE)

PE applies only to adjacency. Diffusion and zero-hop have no PE variant in paper.

Outputs:
  output/tables/table_R1_gcn.tex / .md      — Table 1: GCN
  output/tables/table_R1b_gat.tex / .md     — Table 2: GAT
  output/tables/missing_runs.txt            — Exact list of missing experiments

Usage:
    python experiments/generate_tables.py
    python experiments/generate_tables.py --track R1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["Cora", "Citeseer", "Pubmed"]
BETAS = [10000, 10]
BETA_LABELS = {10000: "IID", 10: "non-IID"}
MIN_REPS = 5

# Method rows: (propagation, use_pe, display_label, apply_to)
# apply_to: "both", "R1_only", "R1b_only"
METHODS_R1 = [
    ("zero_hop",  False, "Zero-hop (baseline)"),
    ("full",      False, "Full-graph (oracle)"),
    ("adjacency", False, "FedProp-Adj"),
    ("diffusion", False, "FedProp-Diff"),
]
METHODS_R1B = [
    ("zero_hop",  False, "Zero-hop (baseline)"),
    ("full",      False, "Full-graph (oracle)"),
    ("adjacency", False, "FedProp-Adj"),
    ("adjacency", True,  "FedProp-Adj + PE"),
    ("diffusion", False, "FedProp-Diff"),
]


def load_csv(track: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"{track}_consolidated.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run consolidate_results.py first")
    df = pd.read_csv(path)
    df["beta"] = df["beta"].astype(int)
    return df


def get_row(df: pd.DataFrame, dataset: str, prop: str, beta: int,
            hop: int, use_pe: bool):
    mask = (
        (df["dataset"] == dataset)
        & (df["propagation"] == prop)
        & (df["beta"] == beta)
        & (df["hop"] == hop)
        & (df["use_pe"] == use_pe)
    )
    r = df[mask]
    if len(r) == 0:
        return None
    ri = r.iloc[0]
    if ri["n_reps"] < MIN_REPS:
        return None
    return ri


def fmt_cell(ri, fmt: str = "console") -> str:
    if ri is None:
        return {"console": "   —  ", "latex": r"\textemdash{}", "md": "—"}[fmt]
    acc = ri["mean_acc"] * 100
    std = ri["std_acc"] * 100
    incomplete = ri["n_reps"] < 10
    if fmt == "console":
        flag = f"*{ri['n_reps']}" if incomplete else "  "
        return f"{acc:5.1f}±{std:3.1f}{flag}"
    elif fmt == "latex":
        val = f"{acc:.1f}\\,{{\\tiny ±{std:.1f}}}"
        if incomplete:
            val = f"\\textit{{{val}}}"
        return val
    else:  # md
        flag = f"\\*" if incomplete else ""
        return f"{acc:.1f}±{std:.1f}{flag}"


def is_best(ri, col_ris) -> bool:
    """True if ri is within 0.15% of the best non-oracle result in this column."""
    if ri is None:
        return False
    valid = [r for r in col_ris if r is not None]
    if not valid:
        return False
    best = max(r["mean_acc"] for r in valid) * 100
    return abs(ri["mean_acc"] * 100 - best) <= 0.15


# ─── Console display ─────────────────────────────────────────────────────────

def print_console(track: str, df: pd.DataFrame, methods: list, hop: int):
    print(f"\n{'═' * 100}")
    print(f"  {'GCN' if track == 'R1' else 'GAT'} (hop={hop})   —   "
          f"{'No PE (GCN does not use PE)' if track == 'R1' else 'PE applies to Adj only'}")
    print(f"{'═' * 100}")

    # Header
    hdr = f"  {'Method':<24}"
    for ds in DATASETS:
        for beta in BETAS:
            hdr += f"  {ds[:3]}-{BETA_LABELS[beta]:>6}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for prop, pe, label in methods:
        row_str = f"  {label:<24}"
        for ds in DATASETS:
            for beta in BETAS:
                ri = get_row(df, ds, prop, beta, hop, pe)
                row_str += f"  {fmt_cell(ri, 'console'):>12}"
        print(row_str)

    print()
    print("  *N = incomplete (<10 reps)    — = missing    ± = std over 10 seeds")


# ─── Missing runs report ──────────────────────────────────────────────────────

def missing_report(track: str, df: pd.DataFrame, methods: list, hop: int) -> str:
    lines = [f"\n{'─' * 70}",
             f"MISSING RUNS: {track} ({'GCN' if track == 'R1' else 'GAT'}, hop={hop})",
             f"{'─' * 70}"]
    total_missing = 0
    total_incomplete = 0
    for ds in DATASETS:
        for beta in BETAS:
            for prop, pe, label in methods:
                ri = get_row(df, ds, prop, beta, hop, pe)
                pe_tag = "+PE" if pe else ""
                config = f"{ds} β={beta} {prop}{pe_tag}"
                if ri is None:
                    lines.append(f"  MISSING      {config:<40} needs 10 runs")
                    total_missing += 1
                elif ri["n_reps"] < 10:
                    need = 10 - int(ri["n_reps"])
                    lines.append(
                        f"  INCOMPLETE   {config:<40} "
                        f"{int(ri['n_reps'])}/10 done, need {need} more"
                    )
                    total_incomplete += 1
    lines.append(f"{'─' * 70}")
    lines.append(f"Total missing: {total_missing}  |  Incomplete: {total_incomplete}")
    return "\n".join(lines)


# ─── LaTeX table ─────────────────────────────────────────────────────────────

def build_latex(track: str, df: pd.DataFrame, methods: list, hop: int) -> str:
    n_data_cols = len(DATASETS) * len(BETAS)
    if track == "R1":
        col_spec = "l" + "r" * n_data_cols
        caption = (
            r"GCN test accuracy (\%) on citation networks (hop=1). "
            r"Rows: lower bound (Zero-hop), oracle upper bound (Full-graph), "
            r"and two propagation methods. Mean\,±\,std over 10 seeds. "
            r"\textemdash{}\,=\,missing; \textit{italics}\,=\,<10 seeds."
        )
        label = "tab:gcn_accuracy"
    else:
        col_spec = "l" + "r" * n_data_cols
        caption = (
            r"GAT test accuracy (\%) on citation networks (hop=2). "
            r"PE (positional encoding) applied to adjacency propagation only. "
            r"Mean\,±\,std over 10 seeds. "
            r"\textemdash{}\,=\,missing; \textit{italics}\,=\,<10 seeds."
        )
        label = "tab:gat_accuracy"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Dataset header row
    ds_row = "Method"
    for ds in DATASETS:
        ds_row += rf" & \multicolumn{{{len(BETAS)}}}{{c}}{{{ds}}}"
    lines.append(ds_row + r" \\")

    # Beta sub-header
    beta_row = ""
    sep_rules = []
    for i, ds in enumerate(DATASETS):
        col_start = 2 + i * len(BETAS)
        col_end = col_start + len(BETAS) - 1
        sep_rules.append(rf"\cmidrule(lr){{{col_start}-{col_end}}}")
        for b in BETAS:
            beta_row += rf" & \small {BETA_LABELS[b]}"
    lines.append(beta_row.lstrip(" &") + r" \\")
    lines.append(" ".join(sep_rules))
    lines.append(r"\midrule")

    # Separator index: draw a \midrule between oracle rows and method rows
    ORACLE_ROWS = {"zero_hop", "full"}

    prev_was_oracle = None
    for row_idx, (prop, pe, label) in enumerate(methods):
        is_oracle = prop in ORACLE_ROWS

        # Add separator between oracle block and methods block
        if prev_was_oracle is True and not is_oracle:
            lines.append(r"\midrule")
        prev_was_oracle = is_oracle

        # Collect per-column row data
        parts = [label]
        col_ris = []
        for ds in DATASETS:
            for beta in BETAS:
                ri = get_row(df, ds, prop, beta, hop, pe)
                col_ris.append(ri)

        for i, (ds, beta) in enumerate(
            [(d, b) for d in DATASETS for b in BETAS]
        ):
            ri = col_ris[i]
            cell = fmt_cell(ri, "latex")
            # Bold if best non-oracle in column
            if not is_oracle and ri is not None:
                col_all = [
                    get_row(df, ds, p, beta, hop, u)
                    for p, u, _ in methods
                    if p not in ORACLE_ROWS
                ]
                if is_best(ri, col_all):
                    cell = rf"\textbf{{{cell}}}"
            parts.append(cell)
        lines.append(" & ".join(parts) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─── Markdown table ──────────────────────────────────────────────────────────

def build_markdown(track: str, df: pd.DataFrame, methods: list, hop: int) -> str:
    header = "| Method |"
    sep = "| --- |"
    for ds in DATASETS:
        for beta in BETAS:
            header += f" {ds} {BETA_LABELS[beta]} |"
            sep += " :---: |"
    lines = [header, sep]

    for prop, pe, label in methods:
        row = f"| {label} |"
        for ds in DATASETS:
            for beta in BETAS:
                ri = get_row(df, ds, prop, beta, hop, pe)
                row += f" {fmt_cell(ri, 'md')} |"
        lines.append(row)

    note = (
        "\n> \\* incomplete (<10 seeds) — = missing   "
        "Bold = best non-oracle per column"
    )
    return "\n".join(lines) + note


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["R1", "R1b", "both"], default="both")
    args = parser.parse_args()
    tracks = ["R1", "R1b"] if args.track == "both" else [args.track]

    all_missing = []
    for track in tracks:
        df = load_csv(track)
        hop = 1 if track == "R1" else 2
        methods = METHODS_R1 if track == "R1" else METHODS_R1B

        # Console
        print_console(track, df, methods, hop)

        # Missing report
        miss = missing_report(track, df, methods, hop)
        print(miss)
        all_missing.append(miss)

        # LaTeX
        tex = build_latex(track, df, methods, hop)
        tex_path = TABLE_DIR / f"table_{track}.tex"
        tex_path.write_text(tex)
        print(f"\nSaved LaTeX: {tex_path}")

        # Markdown
        md = build_markdown(track, df, methods, hop)
        md_path = TABLE_DIR / f"table_{track}.md"
        md_path.write_text(md)
        print(f"Saved MD:    {md_path}")

    # Save combined missing report
    miss_path = TABLE_DIR / "missing_runs.txt"
    miss_path.write_text("\n".join(all_missing))
    print(f"\nMissing runs report: {miss_path}")


if __name__ == "__main__":
    main()
