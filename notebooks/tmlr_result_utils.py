"""Utilities for the TMLR result notebook.

Keeps result-source tracing and table construction out of the notebook body.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path("/home/bosho/FP")
OUTPUT = REPO_ROOT / "experiments/output"
RESULTS = REPO_ROOT / "experiments/results"

R1_CSV = OUTPUT / "R1_consolidated.csv"
R1B_CSV = OUTPUT / "R1b_consolidated.csv"
PHASE6_HETEROPHILY_RAW = (
    REPO_ROOT / "experiments/propagator_eval/results/phase_6_heterophily_stress/raw"
)

SKIP_PATTERNS = (
    "quickval",
    "parallel_test",
    "centralized",
    "minimal",
    "adam_quickval",
    "sgd_quickval",
    "pe_quickval",
    "_v2",
    "pubmed_gcn_v3",
    ".log",
)

DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "Citeseer",
    "pubmed": "Pubmed",
    "ogbn-arxiv": "ogbn-arxiv",
    "amazon-photo": "Photo",
    "amazon-computers": "Computers",
    "chameleon": "Chameleon",
    "squirrel": "Squirrel",
    "texas": "Texas",
    "wisconsin": "Wisconsin",
}
PROP_LABELS = {
    "zero_hop": "Zero-hop",
    "full": "Full-graph",
    "adjacency": "FedProp-Adj",
    "diffusion": "FedProp-Diff",
    "appnp": "APPNP",
}

TABLE_KEYS = [
    "cora_gcn_1-hop",
    "cora_gcn_2-hop",
    "cora_gat_2-hop",
    "citeseer_gcn_1-hop",
    "citeseer_gcn_2-hop",
    "citeseer_gat_2-hop",
    "pubmed_gcn_1-hop",
    "pubmed_gcn_2-hop",
    "pubmed_gat_2-hop",
]

STATUS_KEYS = [
    "cora_gcn_1-hop",
    "cora_gcn_2-hop",
    "cora_gat_1-hop",
    "cora_gat_2-hop",
    "citeseer_gcn_1-hop",
    "citeseer_gcn_2-hop",
    "citeseer_gat_1-hop",
    "citeseer_gat_2-hop",
    "pubmed_gcn_1-hop",
    "pubmed_gcn_2-hop",
    "pubmed_gat_1-hop",
    "pubmed_gat_2-hop",
]

LARGE_SCALE_KEYS = [
    "ogbn-arxiv_gcn_1-hop",
    "ogbn-arxiv_gcn_2-hop",
    "ogbn-arxiv_gat_1-hop",
    "ogbn-arxiv_gat_2-hop",
    "amazon-photo_gcn_1-hop",
    "amazon-photo_gcn_2-hop",
    "amazon-photo_gat_1-hop",
    "amazon-photo_gat_2-hop",
    "amazon-computers_gcn_1-hop",
    "amazon-computers_gcn_2-hop",
    "amazon-computers_gat_1-hop",
    "amazon-computers_gat_2-hop",
]

HETEROPHILY_KEYS = [
    "chameleon_gcn_1-hop",
    "chameleon_gcn_2-hop",
    "chameleon_gat_1-hop",
    "chameleon_gat_2-hop",
    "squirrel_gcn_1-hop",
    "squirrel_gcn_2-hop",
    "squirrel_gat_1-hop",
    "squirrel_gat_2-hop",
    "texas_gcn_1-hop",
    "texas_gcn_2-hop",
    "texas_gat_1-hop",
    "texas_gat_2-hop",
    "wisconsin_gcn_1-hop",
    "wisconsin_gcn_2-hop",
    "wisconsin_gat_1-hop",
    "wisconsin_gat_2-hop",
]

FOLDER_RE = re.compile(
    r"^(?P<dataset>[a-zA-Z0-9\-]+)"
    r"_(?P<propagation>zero_hop|adjacency|diffusion|full)"
    r"_(?P<model>GCN|GAT)"
    r"_beta(?P<beta>[\d.]+)_clients(?P<num_clients>\d+)_hop(?P<hop>\d+)"
    r"_iter(?P<iter>\d+)_t(?P<t>[\d.]+)_alpha(?P<alpha>[\d.]+)(?P<pe>_pe)?$"
)
GLOBAL_TXT_RE = re.compile(r"The global test results:\s*(\[[^\n]*\])")


def r1_baseline_dirs(dataset: str) -> list[Path]:
    """Return valid baseline dirs used by experiments/consolidate_results.py."""
    dataset = dataset.lower()
    return sorted(
        p
        for p in RESULTS.glob(f"baseline_{dataset}*")
        if p.is_dir() and not any(pat in p.name.lower() for pat in SKIP_PATTERNS)
    )


def result_paths() -> dict[str, list[Path]]:
    """Source paths by dataset/model/hop key."""
    return {
        "cora_gcn_1-hop": [
            R1_CSV,
            REPO_ROOT / "experiments/results/R1_cora_nope",
            REPO_ROOT / "experiments/results/R1_cora_pe",
            REPO_ROOT / "experiments/results/R1_cora_rerun",
            REPO_ROOT / "experiments/results/R1_cora_topup",
            *r1_baseline_dirs("cora"),
            REPO_ROOT / "experiments/cora_results_test",
        ],
        "cora_gcn_2-hop": [REPO_ROOT / "experiments/cora_results_test"],
        "cora_gat_1-hop": [],
        "cora_gat_2-hop": [
            R1B_CSV,
            REPO_ROOT / "experiments/results/R1b_cora_nope",
            REPO_ROOT / "experiments/results/R1b_cora_pe",
            REPO_ROOT / "experiments/cora_results_test",
        ],
        "citeseer_gcn_1-hop": [
            R1_CSV,
            REPO_ROOT / "experiments/results/R1_citeseer_nope",
            REPO_ROOT / "experiments/results/R1_citeseer_pe",
            REPO_ROOT / "experiments/results/R1_citeseer_rerun",
            REPO_ROOT / "experiments/results/R1_citeseer_topup",
            *r1_baseline_dirs("citeseer"),
        ],
        "citeseer_gcn_2-hop": [REPO_ROOT / "experiments/citeseer_results_prelim"],
        "citeseer_gat_1-hop": [],
        "citeseer_gat_2-hop": [
            R1B_CSV,
            REPO_ROOT / "experiments/results/R1b_citeseer_nope",
            REPO_ROOT / "experiments/results/R1b_citeseer_pe",
            REPO_ROOT / "experiments/results/R1b_citeseer_rerun",
            REPO_ROOT / "experiments/citeseer_results_prelim",
        ],
        "pubmed_gcn_1-hop": [
            R1_CSV,
            REPO_ROOT / "experiments/results/R1_pubmed_nope",
            REPO_ROOT / "experiments/results/R1_pubmed_pe",
            REPO_ROOT / "experiments/results/R1_pubmed_rerun",
            *r1_baseline_dirs("pubmed"),
        ],
        "pubmed_gcn_2-hop": [REPO_ROOT / "experiments/pubmed_results_prelim"],
        "pubmed_gat_1-hop": [],
        "pubmed_gat_2-hop": [
            R1B_CSV,
            REPO_ROOT / "experiments/results/R1b_pubmed_nope",
            REPO_ROOT / "experiments/results/R1b_pubmed_pe",
            REPO_ROOT / "experiments/results/R1b_pubmed_pe_adj",
            REPO_ROOT / "experiments/results/R1b_pubmed_pe_diff",
            REPO_ROOT / "experiments/results/R1b_pubmed_pe_full",
            REPO_ROOT / "experiments/pubmed_results_prelim",
        ],
        "ogbn-arxiv_gcn_1-hop": [
            REPO_ROOT / "experiments/results/scalability/ogbn_arxiv_smoke",
            REPO_ROOT / "results/ogbn-arxiv",
        ],
        "ogbn-arxiv_gcn_2-hop": [
            REPO_ROOT / "experiments/results/ogbn-arxiv/zero_full2",
            REPO_ROOT / "experiments/results/ogbn-arxiv/adjacency_diffusion2",
            REPO_ROOT / "experiments/results/ogbn-arxiv/adjacency_diffusion_pe",
            REPO_ROOT / "results/ogbn-arxiv",
        ],
        "ogbn-arxiv_gat_1-hop": [
            REPO_ROOT / "results/ogbn-arxiv",
        ],
        "ogbn-arxiv_gat_2-hop": [],
        "amazon-photo_gcn_1-hop": [
            REPO_ROOT / "experiments/configs/R7/R7_photo.yaml",
            REPO_ROOT / "experiments/results/R7",
            REPO_ROOT / "results/photos",
        ],
        "amazon-photo_gcn_2-hop": [
            REPO_ROOT / "experiments/configs/R7/R7_photo_2hop.yaml",
            REPO_ROOT / "experiments/results/R7",
        ],
        "amazon-photo_gat_1-hop": [],
        "amazon-photo_gat_2-hop": [],
        "amazon-computers_gcn_1-hop": [
            REPO_ROOT / "experiments/configs/R7/R7_computers.yaml",
            REPO_ROOT / "experiments/results/R7",
        ],
        "amazon-computers_gcn_2-hop": [
            REPO_ROOT / "experiments/configs/R7/R7_computers_2hop.yaml",
            REPO_ROOT / "experiments/results/R7",
            REPO_ROOT / "experiments/results/smoke/scalability/amazon_computers",
            REPO_ROOT / "experiments/results/smoke/scalability_v2/amazon_computers",
        ],
        "amazon-computers_gat_1-hop": [],
        "amazon-computers_gat_2-hop": [],
        "chameleon_gcn_1-hop": [],
        "chameleon_gcn_2-hop": [],
        "chameleon_gat_1-hop": [],
        "chameleon_gat_2-hop": [],
        "squirrel_gcn_1-hop": [],
        "squirrel_gcn_2-hop": [],
        "squirrel_gat_1-hop": [],
        "squirrel_gat_2-hop": [],
        "texas_gcn_1-hop": [
            REPO_ROOT / "experiments/configs/R6/R6_texas.yaml",
            REPO_ROOT / "experiments/results/R6",
            PHASE6_HETEROPHILY_RAW,
        ],
        "texas_gcn_2-hop": [],
        "texas_gat_1-hop": [],
        "texas_gat_2-hop": [],
        "wisconsin_gcn_1-hop": [
            REPO_ROOT / "experiments/configs/R6/R6_wisconsin.yaml",
            REPO_ROOT / "experiments/results/R6",
            PHASE6_HETEROPHILY_RAW,
        ],
        "wisconsin_gcn_2-hop": [],
        "wisconsin_gat_1-hop": [],
        "wisconsin_gat_2-hop": [],
    }


RESULT_PATHS = result_paths()


def split_result_key(result_key: str) -> tuple[str, str, int]:
    dataset_name, model_name, hop_name = result_key.split("_")
    return DATASET_LABELS[dataset_name], model_name.upper(), int(hop_name.split("-")[0])


def trace_training_csvs(result_key: str) -> list[Path]:
    """Return concrete training CSVs under the folders for one result key."""
    dataset, model, hop = split_result_key(result_key)
    csvs: list[Path] = []
    for path in RESULT_PATHS[result_key]:
        if path.is_dir():
            csvs.extend(
                p
                for p in path.rglob("training_*.csv")
                if f"{dataset}_" in p.name
                and f"_{model}_" in p.name
                and f"_hop{hop}_" in p.name
            )
    return sorted(csvs)


def path_summary() -> pd.DataFrame:
    """Summarize source paths and concrete training CSV counts."""
    return pd.DataFrame(
        [
            {
                "result_key": key,
                "source_paths": [str(p.relative_to(REPO_ROOT)) for p in paths],
                "n_training_csvs": len(trace_training_csvs(key)),
            }
            for key, paths in RESULT_PATHS.items()
        ]
    )


def _fmt_accs(values: list[float]) -> str:
    if not values:
        return "-"
    return ", ".join(f"{100 * v:.1f}" for v in values)


def _fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{100 * np.mean(values):.1f}+/-{100 * np.std(values):.1f}"


def _parse_result_dir(run_dir: Path) -> dict | None:
    m = FOLDER_RE.match(run_dir.name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "dataset": d["dataset"],
        "model": d["model"],
        "propagation": d["propagation"],
        "beta": int(float(d["beta"])),
        "hop": int(d["hop"]),
        "use_pe": bool(d["pe"]),
        "run_dir": run_dir,
    }


def extract_global_results(run_dir: Path) -> tuple[list[float], str]:
    """Return per-repetition global test accuracies from one result directory."""
    per_rep = run_dir / "per_repetition.jsonl"
    if per_rep.exists():
        vals = []
        for line in per_rep.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj.get("global_result"), (int, float)):
                vals.append(float(obj["global_result"]))
        if vals:
            return vals, "per_repetition.jsonl"

    best_vals: list[float] = []
    best_source = "missing"

    for path in sorted(run_dir.glob("results_*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        summary_vals = [
            float(v)
            for v in data.get("summary", {}).get("global_results", [])
            if isinstance(v, (int, float)) and np.isfinite(float(v))
        ]
        if len(summary_vals) > len(best_vals):
            best_vals = summary_vals
            best_source = path.name
            continue
        vals = [
            float(r["global_result"])
            for r in data.get("rounds", [])
            if isinstance(r.get("global_result"), (int, float)) and np.isfinite(float(r["global_result"]))
        ]
        if len(vals) > len(best_vals):
            best_vals = vals
            best_source = path.name

    for path in sorted(run_dir.glob("results_*.txt")):
        m = GLOBAL_TXT_RE.search(path.read_text(errors="ignore"))
        if not m:
            continue
        try:
            vals = [float(v) for v in ast.literal_eval(m.group(1)) if np.isfinite(float(v))]
        except Exception:
            vals = []
        if len(vals) > len(best_vals):
            best_vals = vals
            best_source = path.name

    return best_vals, best_source


def load_canonical_with_reps(track: str) -> pd.DataFrame:
    """Use the consolidation code that created R1/R1b, keeping per-rep accuracies."""
    experiments_dir = str(REPO_ROOT / "experiments")
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
    cr = importlib.import_module("consolidate_results")

    records = cr.collect_all_records(track)
    groups = {}
    for rec in records:
        groups.setdefault(cr.config_key(rec), []).append(rec)

    rows = []
    for recs in groups.values():
        merged = cr.merge_records(recs)
        vals = list(merged["per_rep_accs"])
        rows.append(
            {
                "dataset": merged["dataset"],
                "model": merged["model"],
                "hop": int(merged["hop"]),
                "propagation": merged["propagation"],
                "method": PROP_LABELS.get(merged["propagation"], merged["propagation"]),
                "beta": int(merged["beta"]),
                "PE": "yes" if merged["use_pe"] else "no",
                "n_reps": len(vals),
                "acc_values": vals,
                "mean+/-std (%)": _fmt_mean_std(vals),
                "test_accs (%)": _fmt_accs(vals),
                "source": "canonical R1" if track == "R1" else "canonical R1b",
            }
        )
    return pd.DataFrame(rows)


def load_prelim_from_paths(result_key: str) -> pd.DataFrame:
    """Load non-canonical/prelim results from RESULT_PATHS folders."""
    dataset, model, hop = split_result_key(result_key)
    rows = []
    seen_dirs = set()
    for base in RESULT_PATHS[result_key]:
        if not isinstance(base, Path) or not base.is_dir():
            continue
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir():
                continue
            real = run_dir.resolve()
            if real in seen_dirs:
                continue
            seen_dirs.add(real)
            meta = _parse_result_dir(run_dir)
            if not meta:
                continue
            if meta["dataset"] != dataset or meta["model"] != model or meta["hop"] != hop:
                continue
            vals, source = extract_global_results(run_dir)
            if not vals:
                continue
            rows.append(
                {
                    "dataset": meta["dataset"],
                    "model": meta["model"],
                    "hop": meta["hop"],
                    "propagation": meta["propagation"],
                    "method": PROP_LABELS.get(meta["propagation"], meta["propagation"]),
                    "beta": meta["beta"],
                    "PE": "yes" if meta["use_pe"] else "no",
                    "n_reps": len(vals),
                    "acc_values": vals,
                    "mean+/-std (%)": _fmt_mean_std(vals),
                    "test_accs (%)": _fmt_accs(vals),
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def load_phase6_heterophily_records(result_key: str, include_appnp: bool = False) -> pd.DataFrame:
    """Load the narrower propagator-eval Phase 6 heterophily JSON results.

    Phase 6 overlaps R6 for Texas/Wisconsin GCN hop-1, beta=10000, but it is not
    the full R6 matrix: it has 5 seeds and no beta 10/1 or full-graph runs.
    """
    dataset, model, hop = split_result_key(result_key)
    if dataset not in {"Texas", "Wisconsin"} or model != "GCN" or hop != 1:
        return pd.DataFrame()
    if not PHASE6_HETEROPHILY_RAW.exists():
        return pd.DataFrame()

    operators = ["zero_hop", "adjacency", "diffusion"]
    if include_appnp:
        operators.append("appnp")

    rows = []
    for operator in operators:
        ds_dir = PHASE6_HETEROPHILY_RAW / operator / dataset.lower()
        if not ds_dir.exists():
            continue
        groups: dict[int, list[tuple[int, float]]] = {}
        for path in sorted(ds_dir.glob("beta*_seed*_gcn.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if str(data.get("backbone", "")).upper() != model:
                continue
            beta = int(data.get("beta", 0))
            seed = int(data.get("seed", len(groups.get(beta, []))))
            acc = data.get("test_accuracy")
            if not isinstance(acc, (int, float)):
                continue
            groups.setdefault(beta, []).append((seed, float(acc)))

        for beta, seed_accs in groups.items():
            vals = [acc for _, acc in sorted(seed_accs)]
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "hop": hop,
                    "propagation": operator,
                    "method": PROP_LABELS.get(operator, operator),
                    "beta": beta,
                    "PE": "no",
                    "n_reps": len(vals),
                    "acc_values": vals,
                    "mean+/-std (%)": _fmt_mean_std(vals),
                    "test_accs (%)": _fmt_accs(vals),
                    "source": "propagator_eval Phase 6",
                }
            )
    return pd.DataFrame(rows)


def _sort_result_table(df: pd.DataFrame) -> pd.DataFrame:
    prop_rank = {p: i for i, p in enumerate(["zero_hop", "full", "adjacency", "diffusion", "appnp"])}
    pe_rank = {"no": 0, "yes": 1}
    df = df.copy()
    df["_prop_rank"] = df["propagation"].map(prop_rank).fillna(99)
    df["_pe_rank"] = df["PE"].map(pe_rank).fillna(99)
    df["_beta_rank"] = df["beta"].map({10000: 0, 10: 1, 1: 2}).fillna(99)
    return df.sort_values(["_prop_rank", "_pe_rank", "_beta_rank"]).drop(
        columns=["_prop_rank", "_pe_rank", "_beta_rank"]
    )


def _dedupe_result_records(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the richest row when multiple folders contain the same config."""
    if df.empty:
        return df
    deduped = (
        df.sort_values("n_reps", ascending=False)
        .drop_duplicates(["dataset", "model", "hop", "method", "PE", "beta"], keep="first")
    )
    return _sort_result_table(deduped)


def result_table(
    result_key: str,
    r1_canonical: pd.DataFrame | None = None,
    r1b_canonical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one display-ready result table for a dataset/model/hop key."""
    dataset, model, hop = split_result_key(result_key)
    is_planetoid = dataset in {"Cora", "Citeseer", "Pubmed"}

    if is_planetoid and model == "GCN" and hop == 1:
        df = r1_canonical.copy() if r1_canonical is not None else load_canonical_with_reps("R1")
    elif is_planetoid and model == "GAT" and hop == 2:
        df = r1b_canonical.copy() if r1b_canonical is not None else load_canonical_with_reps("R1b")
    else:
        df = load_prelim_from_paths(result_key)

    if df.empty:
        phase6 = load_phase6_heterophily_records(result_key)
        if phase6.empty:
            return df
        df = phase6

    df = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["hop"] == hop)].copy()
    if df.empty:
        return df

    df = _dedupe_result_records(_sort_result_table(df))
    return df[
        ["method", "PE", "beta", "n_reps", "mean+/-std (%)", "test_accs (%)", "source"]
    ].reset_index(drop=True)


def result_records(
    result_key: str,
    r1_canonical: pd.DataFrame | None = None,
    r1b_canonical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return full records for a result key, including raw acc_values lists."""
    dataset, model, hop = split_result_key(result_key)
    is_planetoid = dataset in {"Cora", "Citeseer", "Pubmed"}

    if is_planetoid and model == "GCN" and hop == 1:
        df = r1_canonical.copy() if r1_canonical is not None else load_canonical_with_reps("R1")
    elif is_planetoid and model == "GAT" and hop == 2:
        df = r1b_canonical.copy() if r1b_canonical is not None else load_canonical_with_reps("R1b")
    else:
        df = load_prelim_from_paths(result_key)

    if df.empty:
        phase6 = load_phase6_heterophily_records(result_key)
        if phase6.empty:
            return df
        df = phase6

    df = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["hop"] == hop)].copy()
    if df.empty:
        return df
    return _dedupe_result_records(_sort_result_table(df)).reset_index(drop=True)


def result_points(
    result_key: str,
    r1_canonical: pd.DataFrame | None = None,
    r1b_canonical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one row per recovered repetition datapoint for plotting."""
    records = result_records(result_key, r1_canonical=r1_canonical, r1b_canonical=r1b_canonical)
    rows = []
    for _, row in records.iterrows():
        values = row.get("acc_values", [])
        for rep_idx, acc in enumerate(values, start=1):
            rows.append(
                {
                    "result_key": result_key,
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "hop": row["hop"],
                    "method": row["method"],
                    "propagation": row["propagation"],
                    "PE": row["PE"],
                    "beta": int(row["beta"]),
                    "rep": rep_idx,
                    "acc": float(acc),
                    "acc_pct": 100 * float(acc),
                    "source": row["source"],
                }
            )
    return pd.DataFrame(rows)


def table_summary(table: pd.DataFrame) -> dict[str, int]:
    """Compact repetition-count summary for one result table."""
    if table.empty:
        return {"n_configs": 0, "total_reps": 0, "min_reps": 0, "max_reps": 0}
    return {
        "n_configs": int(len(table)),
        "total_reps": int(table["n_reps"].sum()),
        "min_reps": int(table["n_reps"].min()),
        "max_reps": int(table["n_reps"].max()),
    }


def _expected_configs(result_key: str, observed: pd.DataFrame | None = None) -> list[dict]:
    """Expected config rows for coverage accounting."""
    dataset, model, hop = split_result_key(result_key)
    is_planetoid = dataset in {"Cora", "Citeseer", "Pubmed"}

    if is_planetoid and model == "GCN" and hop == 1:
        return [
            {"method": PROP_LABELS[prop], "propagation": prop, "PE": "no", "beta": beta}
            for prop in ["zero_hop", "full", "adjacency", "diffusion"]
            for beta in [10000, 10, 1]
        ]

    if is_planetoid and model == "GAT" and hop == 2:
        rows = [
            {"method": PROP_LABELS[prop], "propagation": prop, "PE": "no", "beta": beta}
            for prop in ["zero_hop", "full"]
            for beta in [10000, 10, 1]
        ]
        rows.extend(
            {"method": PROP_LABELS[prop], "propagation": prop, "PE": pe, "beta": beta}
            for prop in ["adjacency", "diffusion"]
            for pe in ["no", "yes"]
            for beta in [10000, 10, 1]
        )
        return rows

    if dataset in {"Texas", "Wisconsin"} and model == "GCN" and hop == 1:
        return [
            {"method": PROP_LABELS[prop], "propagation": prop, "PE": "no", "beta": beta}
            for prop in ["zero_hop", "full", "adjacency", "diffusion"]
            for beta in [10000, 10, 1]
        ]

    # Prelim / non-Planetoid groups use the configs currently present on disk.
    if observed is not None and not observed.empty:
        return [
            {
                "method": row["method"],
                "propagation": row.get("propagation", ""),
                "PE": row["PE"],
                "beta": int(row["beta"]),
            }
            for _, row in observed.iterrows()
        ]

    return []


def _config_key(row: pd.Series | dict) -> tuple[str, str, int]:
    return (row["method"], row["PE"], int(row["beta"]))


def _missing_config_text(missing_rows: list[dict], max_items: int = 4) -> str:
    if not missing_rows:
        return "-"
    labels = [f"{row['method']} PE={row['PE']} beta={row['beta']}" for row in missing_rows]
    shown = labels[:max_items]
    if len(labels) > max_items:
        shown.append(f"+{len(labels) - max_items} more")
    return "; ".join(shown)


def status_detail_table(result_key: str, target_reps: int = 10) -> pd.DataFrame:
    """Per-config repetition coverage for one result key."""
    observed = result_records(result_key)
    expected = _expected_configs(result_key, observed=observed)
    dataset, model, hop = split_result_key(result_key)

    if not expected and observed.empty:
        return pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "model": model,
                    "hop": hop,
                    "method": "-",
                    "PE": "-",
                    "beta": "-",
                    "reps": f"0/{target_reps}",
                    "missing_reps": target_reps,
                    "status": "missing source",
                }
            ]
        )

    observed_by_key = {_config_key(row): row for _, row in observed.iterrows()}
    rows = []
    for exp in expected:
        obs = observed_by_key.get(_config_key(exp))
        n_reps = int(obs["n_reps"]) if obs is not None else 0
        missing = max(target_reps - n_reps, 0)
        status = "complete" if missing == 0 else "partial" if n_reps else "missing"
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "hop": hop,
                "method": exp["method"],
                "PE": exp["PE"],
                "beta": exp["beta"],
                "reps": f"{n_reps}/{target_reps}",
                "missing_reps": missing,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def status_report_table(keys: list[str] | None = None, target_reps: int = 10) -> pd.DataFrame:
    """Group-level status table for dataset/model/hop groups."""
    keys = keys or STATUS_KEYS
    rows = []
    for key in keys:
        dataset, model, hop = split_result_key(key)
        observed = result_records(key)

        if observed.empty:
            has_declared_paths = bool(RESULT_PATHS.get(key))
            note = "No source folder found"
            if has_declared_paths:
                note = "Config/path found, but no usable result datapoints"
            rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Hop": hop,
                    "Result key": key,
                    "Configs": "0",
                    "Repetitions": f"0/{target_reps}",
                    "Status": "missing source",
                    "Missing / notes": note,
                }
            )
            continue

        expected = _expected_configs(key, observed=observed)
        expected_keys = {_config_key(row): row for row in expected}
        observed_keys = {_config_key(row): row for _, row in observed.iterrows()}

        missing_configs = [row for cfg_key, row in expected_keys.items() if cfg_key not in observed_keys]
        expected_config_count = len(expected_keys)
        found_config_count = len([cfg_key for cfg_key in observed_keys if cfg_key in expected_keys])
        expected_reps = expected_config_count * target_reps
        found_reps = int(
            sum(
                min(int(row["n_reps"]), target_reps)
                for cfg_key, row in observed_keys.items()
                if cfg_key in expected_keys
            )
        )
        missing_reps = expected_reps - found_reps
        status = "complete" if missing_reps == 0 and not missing_configs else "partial"

        notes = _missing_config_text(missing_configs)
        if missing_reps and notes == "-":
            partial = observed[observed["n_reps"] < target_reps]
            notes = "; ".join(
                f"{row['method']} PE={row['PE']} beta={int(row['beta'])}: {int(row['n_reps'])}/{target_reps}"
                for _, row in partial.head(4).iterrows()
            )
            if len(partial) > 4:
                notes += f"; +{len(partial) - 4} more partial configs"

        rows.append(
            {
                "Dataset": dataset,
                "Model": model,
                "Hop": hop,
                "Result key": key,
                "Configs": f"{found_config_count}/{expected_config_count}",
                "Repetitions": f"{found_reps}/{expected_reps}",
                "Status": status,
                "Missing / notes": notes,
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def status_report_markdown(keys: list[str] | None = None, target_reps: int = 10) -> str:
    """Clean Markdown coverage report for notebook display or export."""
    table = status_report_table(keys=keys, target_reps=target_reps)
    return "\n".join(
        [
            "## Result Coverage Status",
            "",
            f"Target is **{target_reps} repetitions per config** unless noted. "
            "Some sections include prelim or overlapping evidence when full canonical results are absent.",
            "",
            _markdown_table(table),
        ]
    )


def display_status_report(keys: list[str] | None = None, target_reps: int = 10) -> None:
    """Display coverage status report as Markdown in a notebook."""
    from IPython.display import Markdown, display

    display(Markdown(status_report_markdown(keys=keys, target_reps=target_reps)))


def phase6_heterophily_table(include_appnp: bool = True) -> pd.DataFrame:
    """Return the available Phase 6 Texas/Wisconsin heterophily summary."""
    frames = [
        load_phase6_heterophily_records(key, include_appnp=include_appnp)
        for key in ["texas_gcn_1-hop", "wisconsin_gcn_1-hop"]
    ]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "hop",
                "method",
                "PE",
                "beta",
                "n_reps",
                "mean+/-std (%)",
                "test_accs (%)",
                "source",
            ]
        )
    df = _sort_result_table(pd.concat(frames, ignore_index=True))
    return df[
        [
            "dataset",
            "model",
            "hop",
            "method",
            "PE",
            "beta",
            "n_reps",
            "mean+/-std (%)",
            "test_accs (%)",
            "source",
        ]
    ].reset_index(drop=True)


def heterophily_status_markdown(target_reps: int = 10) -> str:
    """Markdown status report for the requested heterophilic result groups."""
    lines = [
        "## Heterophily Result Status",
        "",
        "R6 configs exist only for **Texas GCN 1-hop** and **Wisconsin GCN 1-hop**. "
        "Those configs target beta {10000, 10, 1}, methods {Zero-hop, Full-graph, "
        "FedProp-Adj, FedProp-Diff}, and 10 repetitions.",
        "",
        "The only recovered datapoints are from `experiments/propagator_eval/results/"
        "phase_6_heterophily_stress/raw`: Texas/Wisconsin, GCN, beta=10000, "
        "5 seeds, for Zero-hop/FedProp-Adj/FedProp-Diff plus an extra APPNP operator. "
        "No top-level `experiments/results/R6` artifacts were found.",
        "",
        _markdown_table(status_report_table(keys=HETEROPHILY_KEYS, target_reps=target_reps)),
    ]
    return "\n".join(lines)


def display_heterophily_status(target_reps: int = 10) -> None:
    """Display the heterophily status report in a notebook."""
    from IPython.display import Markdown, display

    display(Markdown(heterophily_status_markdown(target_reps=target_reps)))


def export_heterophily_outputs(
    csv_path: Path | str | None = None,
    md_path: Path | str | None = None,
    include_appnp: bool = True,
) -> tuple[Path, Path]:
    """Write the available heterophily data and status note to disk."""
    csv_path = Path(csv_path) if csv_path is not None else OUTPUT / "tables/heterophily_results_summary.csv"
    md_path = Path(md_path) if md_path is not None else OUTPUT / "tables/heterophily_results_status.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    phase6_frames = [
        load_phase6_heterophily_records(key, include_appnp=include_appnp)
        for key in ["texas_gcn_1-hop", "wisconsin_gcn_1-hop"]
    ]
    phase6_frames = [df for df in phase6_frames if not df.empty]
    phase6 = pd.concat(phase6_frames, ignore_index=True) if phase6_frames else pd.DataFrame()
    for _, row in phase6.iterrows():
        values = list(row["acc_values"])
        records.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "hop": int(row["hop"]),
                "method": row["method"],
                "PE": row["PE"],
                "beta": int(row["beta"]),
                "n_reps": int(row["n_reps"]),
                "mean_acc": float(np.mean(values)),
                "std_acc": float(np.std(values)),
                "mean_acc_pct": 100 * float(np.mean(values)),
                "std_acc_pct": 100 * float(np.std(values)),
                "note": "extra Phase 6 operator" if row["method"] == "APPNP" else "overlaps R6 beta=10000",
            }
        )
    pd.DataFrame(records).to_csv(csv_path, index=False)
    md_path.write_text(heterophily_status_markdown())
    return csv_path, md_path


def plot_result_table(
    result_key: str,
    *,
    ax=None,
    show_points: bool = True,
    show_pe: str | None = None,
    title: str | None = None,
):
    """Plot mean test accuracy by method and beta for one result key.

    Parameters
    ----------
    result_key:
        Example: "cora_gcn_1-hop".
    ax:
        Optional Matplotlib axis.
    show_points:
        Overlay individual repetition datapoints when True.
    show_pe:
        Optional PE filter: "yes" or "no".
    title:
        Optional title override.
    """
    import matplotlib.pyplot as plt

    records = result_records(result_key)
    points = result_points(result_key)

    if show_pe is not None:
        records = records[records["PE"] == show_pe].copy()
        points = points[points["PE"] == show_pe].copy()

    if records.empty:
        raise ValueError(f"No result records found for {result_key!r}")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.8))

    has_multiple_pe = records["PE"].nunique() > 1
    records = records.copy()
    points = points.copy()
    records["plot_group"] = records["method"] + np.where(
        has_multiple_pe, "\nPE=" + records["PE"], ""
    )
    if not points.empty:
        points["plot_group"] = points["method"] + np.where(
            has_multiple_pe, "\nPE=" + points["PE"], ""
        )

    groups = list(records["plot_group"].drop_duplicates())
    betas = [b for b in [10000, 10, 1] if b in set(records["beta"])]
    pe_values = list(records["PE"].drop_duplicates())
    pe_suffix = "" if len(pe_values) == 1 else " + PE"

    width = 0.22
    x = np.arange(len(groups))
    offsets = np.linspace(-width, width, max(len(betas), 1))
    colors = {10000: "#4c78a8", 10: "#f58518", 1: "#54a24b"}

    for offset, beta in zip(offsets, betas):
        sub = records[records["beta"] == beta]
        means = []
        stds = []
        ns = []
        for group in groups:
            match = sub[sub["plot_group"] == group]
            if match.empty:
                means.append(np.nan)
                stds.append(0)
                ns.append(0)
                continue
            vals = []
            for value_list in match["acc_values"]:
                vals.extend(value_list)
            means.append(100 * np.mean(vals))
            stds.append(100 * np.std(vals))
            ns.append(len(vals))

        label = f"beta={beta}"
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=label,
            color=colors.get(beta),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, n in zip(bars, ns):
            if n:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.35,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

        if show_points and not points.empty:
            point_sub = points[points["beta"] == beta]
            for group_idx, group in enumerate(groups):
                vals = point_sub[point_sub["plot_group"] == group]["acc_pct"].to_numpy()
                if len(vals) == 0:
                    continue
                jitter = np.linspace(-0.035, 0.035, len(vals)) if len(vals) > 1 else np.array([0.0])
                ax.scatter(
                    np.full(len(vals), x[group_idx] + offset) + jitter,
                    vals,
                    s=18,
                    color="black",
                    alpha=0.45,
                    linewidths=0,
                    zorder=3,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(len(betas), 3))

    if title is None:
        title = result_key.replace("_", " ").replace("-", " ")
        if pe_suffix:
            title = f"{title}{pe_suffix}"
    ax.set_title(title)
    return ax


def plot_result_points(result_key: str, *, ax=None, show_pe: str | None = None, title: str | None = None):
    """Plot every repetition datapoint as a compact strip plot."""
    import matplotlib.pyplot as plt

    points = result_points(result_key)
    if show_pe is not None:
        points = points[points["PE"] == show_pe].copy()
    if points.empty:
        raise ValueError(f"No repetition datapoints found for {result_key!r}")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))

    has_multiple_pe = points["PE"].nunique() > 1
    points = points.copy()
    points["plot_group"] = points["method"] + np.where(
        has_multiple_pe, "\nPE=" + points["PE"], ""
    )

    labels = []
    grouped = []
    for group in points["plot_group"].drop_duplicates():
        for beta in [10000, 10, 1]:
            vals = points[(points["plot_group"] == group) & (points["beta"] == beta)]["acc_pct"].to_numpy()
            if len(vals):
                labels.append(f"{group}\nbeta={beta}")
                grouped.append(vals)

    for idx, vals in enumerate(grouped):
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), idx) + jitter, vals, s=24, alpha=0.7)
        ax.hlines(np.mean(vals), idx - 0.25, idx + 0.25, color="black", linewidth=2)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(title or f"{result_key.replace('_', ' ').replace('-', ' ')} repetition datapoints")
    return ax


def display_result_tables(keys: list[str] | None = None) -> None:
    """Display Markdown summaries plus result tables in a notebook."""
    from IPython.display import Markdown, display

    keys = keys or TABLE_KEYS
    r1 = load_canonical_with_reps("R1")
    r1b = load_canonical_with_reps("R1b")

    for key in keys:
        table = result_table(key, r1_canonical=r1, r1b_canonical=r1b)
        title = key.replace("_", " ").replace("-", " ")
        if table.empty:
            display(Markdown(f"### {title}\n\nNo source data found."))
            continue
        stats = table_summary(table)
        display(
            Markdown(
                f"### {title}\n\n"
                f"Configs: **{stats['n_configs']}** · "
                f"total recovered repetition datapoints: **{stats['total_reps']}** · "
                f"reps/config: **{stats['min_reps']}–{stats['max_reps']}**."
            )
        )
        display(table)
