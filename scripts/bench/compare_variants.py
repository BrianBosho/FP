#!/usr/bin/env python3
"""Compare two (or more) variant runs produced by ``run_variant.py``.

Usage (typical A/B)::

    python scripts/bench/compare_variants.py \
        --baseline  bench/runs/baseline_20260420_153000 \
        --candidate bench/runs/fedavg_weighted_20260420_154500 \
        --output    bench/compare/baseline_vs_fedavg_weighted

Emits under ``--output``:

* ``comparison.md``   - human-friendly table per experiment key
* ``comparison.csv``  - machine-readable form of the same table
* ``per_round.csv``   - per-round global/client accuracy for both variants
                        so convergence curves can be plotted separately

Group-by dimensions (called the "experiment key") are:
dataset, data_loading_option, model_type, num_clients, beta, hop, use_pe,
fulltraining_flag.  Two runs are compared only when their keys match
exactly — we do not silently average over heterogeneous configurations.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import common  # type: ignore  # noqa: E402


def load_variant(run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Returns (manifest, rows).  `rows` includes the 'rounds' list so
    per-round curves can be reconstructed."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {run_dir}")
    manifest = json.loads(manifest_path.read_text())
    variant = manifest.get("variant", run_dir.name)
    rows = common.collect_variant_rows(run_dir, variant)
    return manifest, rows


def group_by_key(rows: List[Dict[str, Any]]) -> Dict[Tuple, List[Dict[str, Any]]]:
    grouped: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[common.experiment_key(r)].append(r)
    return grouped


def format_compare_table(
    baseline_name: str,
    candidate_name: str,
    comparisons: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Comparison: `{candidate_name}` vs `{baseline_name}`")
    lines.append("")
    lines.append(f"Rows: {len(comparisons)} experiment key(s) in common.")
    lines.append("")
    lines.append("| Experiment key | Metric | Baseline (mean ± std, n) | Candidate (mean ± std, n) | Δ (cand − base) | Welch t | p-value | Cohen's d |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in comparisons:
        key_str = c["experiment_key_str"]
        for metric in ("avg_global_acc", "avg_client_acc"):
            b = c[metric]["baseline"]
            k = c[metric]["candidate"]
            delta = c[metric]["delta"]
            t, p = c[metric]["welch_t"], c[metric]["welch_p"]
            d = c[metric]["cohens_d"]
            lines.append(
                f"| {key_str} | `{metric}` "
                f"| {b['mean']:.4f} ± {b['std']:.4f} (n={b['n']}) "
                f"| {k['mean']:.4f} ± {k['std']:.4f} (n={k['n']}) "
                f"| {delta:+.4f} "
                f"| {t:.3f} | {p:.3g} | {d:.3f} |"
            )
    lines.append("")
    lines.append("## Per-seed wall time (seconds)")
    lines.append("")
    lines.append("| Experiment key | Baseline seeds (mean) | Candidate seeds (mean) |")
    lines.append("|---|---|---|")
    for c in comparisons:
        b = c["duration_s"]["baseline"]
        k = c["duration_s"]["candidate"]
        lines.append(f"| {c['experiment_key_str']} | {b['mean']:.2f} (n={b['n']}) | {k['mean']:.2f} (n={k['n']}) |")
    return "\n".join(lines) + "\n"


def build_comparisons(
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    b_groups = group_by_key(baseline_rows)
    c_groups = group_by_key(candidate_rows)
    common_keys = sorted(set(b_groups.keys()) & set(c_groups.keys()),
                         key=lambda k: tuple(str(v) for v in k))

    comparisons: List[Dict[str, Any]] = []
    for key in common_keys:
        b = b_groups[key]
        c = c_groups[key]
        entry: Dict[str, Any] = {
            "experiment_key": dict(zip(common.EXPERIMENT_KEY_FIELDS, key)),
            "experiment_key_str": common.experiment_key_str(b[0]),
        }
        for metric in ("avg_global_acc", "avg_client_acc", "duration_s"):
            bv = [r.get(metric) for r in b]
            cv = [r.get(metric) for r in c]
            bm, bs = common.mean_std(bv)
            cm, cs = common.mean_std(cv)
            t, p = common.welch_t_test(bv, cv) if metric != "duration_s" else (float("nan"), float("nan"))
            d = common.cohens_d(bv, cv) if metric != "duration_s" else float("nan")
            entry[metric] = {
                "baseline": {"mean": bm, "std": bs, "n": len([x for x in bv if x is not None])},
                "candidate": {"mean": cm, "std": cs, "n": len([x for x in cv if x is not None])},
                "delta": cm - bm if (cm == cm and bm == bm) else float("nan"),  # nan-safe
                "welch_t": t,
                "welch_p": p,
                "cohens_d": d,
            }
        comparisons.append(entry)
    return comparisons


def write_comparison_csv(path: Path, comparisons: List[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for c in comparisons:
        base_row = {**c["experiment_key"]}
        for metric in ("avg_global_acc", "avg_client_acc", "duration_s"):
            m = c[metric]
            base_row[f"{metric}_baseline_mean"] = m["baseline"]["mean"]
            base_row[f"{metric}_baseline_std"] = m["baseline"]["std"]
            base_row[f"{metric}_baseline_n"] = m["baseline"]["n"]
            base_row[f"{metric}_candidate_mean"] = m["candidate"]["mean"]
            base_row[f"{metric}_candidate_std"] = m["candidate"]["std"]
            base_row[f"{metric}_candidate_n"] = m["candidate"]["n"]
            base_row[f"{metric}_delta"] = m["delta"]
            base_row[f"{metric}_welch_t"] = m["welch_t"]
            base_row[f"{metric}_welch_p"] = m["welch_p"]
            base_row[f"{metric}_cohens_d"] = m["cohens_d"]
        rows.append(base_row)
    common.write_csv(path, rows)


def write_per_round_csv(
    path: Path,
    baseline_name: str,
    baseline_rows: List[Dict[str, Any]],
    candidate_name: str,
    candidate_rows: List[Dict[str, Any]],
) -> None:
    """Emits long-form per-repetition (final-round) accuracy for both variants.

    Kept for backward compatibility -- this reflects the legacy JSON layout
    where ``rounds`` was indexed by repetition, not by FL round.  For
    per-FL-round convergence curves, see ``write_convergence_csv``.
    """
    out: List[Dict[str, Any]] = []
    for variant_name, rows in ((baseline_name, baseline_rows), (candidate_name, candidate_rows)):
        for r in rows:
            for rd in r.get("rounds", []) or []:
                out.append({
                    "variant": variant_name,
                    "seed": r["seed"],
                    **{k: r.get(k) for k in common.EXPERIMENT_KEY_FIELDS},
                    "round": rd.get("round"),
                    "global_result": rd.get("global_result"),
                    "client_result": rd.get("client_result"),
                })
    common.write_csv(path, out)


def write_convergence_csv(
    path: Path,
    baseline_name: str,
    baseline_rows: List[Dict[str, Any]],
    candidate_name: str,
    candidate_rows: List[Dict[str, Any]],
) -> int:
    """Emits long-form per-FL-round convergence curves.

    Requires `log_per_round: true` in the config used by ``run_variant.py``
    (the default in ``conf/base.yaml``).  Each row is:

        (variant, seed, <experiment_key...>, repetition, fl_round,
         avg_client_val_acc, avg_client_val_loss, best_eval_acc_so_far,
         patience, round_time_s, global_test_acc [optional])

    Returns the number of rows written; 0 means no ``round_history`` was
    present in any result file (older runs, or `log_per_round: false`).
    """
    out: List[Dict[str, Any]] = []
    for variant_name, rows in ((baseline_name, baseline_rows), (candidate_name, candidate_rows)):
        for r in rows:
            for rd in r.get("rounds", []) or []:
                history = rd.get("round_history")
                if not history:
                    continue
                for hist in history:
                    out.append({
                        "variant": variant_name,
                        "seed": r["seed"],
                        **{k: r.get(k) for k in common.EXPERIMENT_KEY_FIELDS},
                        "repetition": rd.get("round"),
                        "fl_round": hist.get("fl_round"),
                        "avg_client_val_acc": hist.get("avg_client_val_acc"),
                        "avg_client_val_loss": hist.get("avg_client_val_loss"),
                        "best_eval_acc_so_far": hist.get("best_eval_acc_so_far"),
                        "best_eval_loss_so_far": hist.get("best_eval_loss_so_far"),
                        "patience": hist.get("patience"),
                        "round_time_s": hist.get("round_time_s"),
                        "global_test_acc": hist.get("global_test_acc"),
                    })
    if out:
        common.write_csv(path, out)
    return len(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, type=Path,
                    help="Path to baseline variant run directory.")
    ap.add_argument("--candidate", required=True, type=Path,
                    help="Path to candidate variant run directory.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output directory (default: bench/compare/<base>_vs_<cand>_<ts>).")
    args = ap.parse_args()

    base_manifest, base_rows = load_variant(args.baseline)
    cand_manifest, cand_rows = load_variant(args.candidate)

    base_name = base_manifest.get("variant", args.baseline.name)
    cand_name = cand_manifest.get("variant", args.candidate.name)

    if args.output is None:
        out_dir = Path("bench/compare") / f"{base_name}_vs_{cand_name}_{common.now_stamp()}"
    else:
        out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base_rows:
        print(f"ERROR: baseline run has no parseable results: {args.baseline}", file=sys.stderr)
        return 2
    if not cand_rows:
        print(f"ERROR: candidate run has no parseable results: {args.candidate}", file=sys.stderr)
        return 2

    comparisons = build_comparisons(base_rows, cand_rows)
    if not comparisons:
        print("ERROR: no common experiment keys between baseline and candidate runs.", file=sys.stderr)
        print("       Baseline keys:", sorted(set(common.experiment_key(r) for r in base_rows)), file=sys.stderr)
        print("       Candidate keys:", sorted(set(common.experiment_key(r) for r in cand_rows)), file=sys.stderr)
        return 3

    md = format_compare_table(base_name, cand_name, comparisons)
    (out_dir / "comparison.md").write_text(md)
    write_comparison_csv(out_dir / "comparison.csv", comparisons)
    write_per_round_csv(out_dir / "per_round.csv", base_name, base_rows, cand_name, cand_rows)
    n_conv = write_convergence_csv(
        out_dir / "convergence.csv", base_name, base_rows, cand_name, cand_rows,
    )

    pointer = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_variant": base_name,
        "candidate_variant": cand_name,
        "baseline_git": base_manifest.get("git"),
        "candidate_git": cand_manifest.get("git"),
        "num_experiment_keys": len(comparisons),
        "generated_at": common.now_stamp(),
    }
    common.dump_json(out_dir / "provenance.json", pointer)

    print(md)
    print(f"[bench-compare] wrote: {out_dir}/comparison.md")
    print(f"[bench-compare] wrote: {out_dir}/comparison.csv")
    print(f"[bench-compare] wrote: {out_dir}/per_round.csv")
    if n_conv > 0:
        print(f"[bench-compare] wrote: {out_dir}/convergence.csv "
              f"({n_conv} per-FL-round rows)")
    else:
        print("[bench-compare] no round_history found in results; "
              "convergence.csv not written. Re-run with `log_per_round: true`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
