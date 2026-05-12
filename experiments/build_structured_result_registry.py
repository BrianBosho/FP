#!/usr/bin/env python3
"""Build a structured, non-destructive mirror of experiment result artifacts.

The raw experiment tree contains historical batch names such as R1, R1b, R5,
and rerun folders. This script copies the condition-level artifacts into a
stable layout that mirrors experiments/configs:

    experiments/output/result_registry/structured_sources/
      <family>/<dataset>/<model>/<hop>/<purpose>/<propagation>/<beta>/

Original files are never moved or deleted.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
DEFAULT_DEST = EXPERIMENTS_ROOT / "output" / "result_registry" / "structured_sources"

CONDITION_RE = re.compile(
    r"^(?P<dataset>[A-Za-z0-9-]+)"
    r"_(?P<propagation>zero_hop|adjacency|diffusion|full|appnp|asymmetric_random_walk|chebyshev_diffusion)"
    r"_(?P<model>GCN|GAT)"
    r"_beta(?P<beta>[\d.]+)_clients(?P<num_clients>\d+)_hop(?P<hop>\d+)"
    r"_iter(?P<iterations>\d+)_t(?P<t>[\d.]+)_alpha(?P<alpha>[\d.]+)(?P<pe>_pe)?$"
)
TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")

PLANETOID = {"cora", "citeseer", "pubmed"}
LARGE = {
    "ogbn-arxiv",
    "ogbn_arxiv",
    "amazon-computers",
    "amazon_computers",
    "computers",
    "amazon-photos",
    "amazon_photos",
    "photo",
    "photos",
}
HETEROPHILIC = {
    "texas",
    "wisconsin",
    "actor",
    "roman-empire",
    "roman_empire",
    "amazon-ratings",
    "amazon_ratings",
    "minesweeper",
}


@dataclass(frozen=True)
class Condition:
    dataset: str
    dataset_slug: str
    family: str
    propagation: str
    model: str
    beta: str
    num_clients: int
    hop: str
    iterations: int
    t: str
    alpha: str
    use_pe: bool
    purpose: str
    source_dir: str
    destination_dir: str


def slugify(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def canonical_dataset_slug(dataset: str) -> str:
    slug = slugify(dataset)
    aliases = {
        "photo": "amazon-photos",
        "photos": "amazon-photos",
        "computers": "amazon-computers",
        "ogbn-arxiv": "ogbn-arxiv",
        "ogbn-arxiv": "ogbn-arxiv",
    }
    return aliases.get(slug, slug)


def family_for(dataset_slug: str) -> str:
    if dataset_slug in PLANETOID:
        return "planetoid"
    if dataset_slug in LARGE:
        return "large"
    if dataset_slug in HETEROPHILIC:
        return "heterophilic"
    return "other"


def beta_label(beta: str) -> str:
    value = float(beta)
    if value.is_integer():
        return f"beta{int(value)}"
    return f"beta{str(beta).replace('.', 'p')}"


def infer_purpose(source_dir: Path, use_pe: bool) -> str:
    parts = {p.lower() for p in source_dir.parts}
    name = source_dir.name.lower()
    parent = source_dir.parent.name.lower()

    if "topup" in parent or "topup" in name:
        return "topups"
    if "rerun" in parent or "rerun" in name:
        return "reruns"
    if "quickval" in parent or "quickval" in name:
        return "quickval"
    if use_pe:
        return "pe"
    if "pe" in parent and not use_pe:
        return "main"
    if "r5" in parts or parent == "r5":
        return "client_sweep"
    return "main"


def iter_source_roots(exp_root: Path) -> list[Path]:
    candidates = [
        exp_root / "results",
        exp_root / "cora_results_test",
        exp_root / "citeseer_results_prelim",
        exp_root / "pubmed_results_prelim",
    ]
    return [p for p in candidates if p.exists()]


def find_condition_dirs(source_roots: Iterable[Path]) -> Iterable[tuple[Path, re.Match[str]]]:
    for root in source_roots:
        for directory in sorted(p for p in root.rglob("*") if p.is_dir()):
            match = CONDITION_RE.match(directory.name)
            if match:
                yield directory, match


def timestamp_for(path: Path) -> str:
    match = TIMESTAMP_RE.search(path.name)
    return match.group(1) if match else "undated"


def final_accuracy_from_training_csv(path: Path) -> float | None:
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            finals: list[float] = []
            for row in reader:
                values = ast.literal_eval(row.get("training_accuracies", "[]"))
                if values:
                    finals.append(float(values[-1]))
            if finals:
                return mean(finals)
    except Exception:
        return None
    return None


def copy_file(src: Path, dest: Path, *, overwrite: bool) -> int:
    if dest.exists() and not overwrite:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return src.stat().st_size


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_summary_csv(path: Path, condition: Condition, accuracies: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_acc = mean(accuracies) if accuracies else math.nan
    std_acc = pstdev(accuracies) if len(accuracies) > 1 else 0.0 if accuracies else math.nan
    fieldnames = [
        "dataset",
        "model",
        "propagation",
        "beta",
        "hop",
        "use_pe",
        "num_clients",
        "n_reps",
        "metric_source",
        "mean_final_train_acc",
        "std_final_train_acc",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": condition.dataset,
                "model": condition.model,
                "propagation": condition.propagation,
                "beta": condition.beta,
                "hop": condition.hop.removeprefix("hop"),
                "use_pe": condition.use_pe,
                "num_clients": condition.num_clients,
                "n_reps": len(accuracies),
                "metric_source": "training_csv_final_client_accuracy",
                "mean_final_train_acc": mean_acc,
                "std_final_train_acc": std_acc,
            }
        )


def condition_from_match(source_dir: Path, match: re.Match[str], dest_root: Path) -> Condition:
    groups = match.groupdict()
    dataset_slug = canonical_dataset_slug(groups["dataset"])
    family = family_for(dataset_slug)
    use_pe = groups["pe"] is not None
    purpose = infer_purpose(source_dir, use_pe)
    hop = f"hop{groups['hop']}"
    beta = beta_label(groups["beta"])
    model = groups["model"].lower()

    dest_dir = (
        dest_root
        / family
        / dataset_slug
        / model
        / hop
        / purpose
        / groups["propagation"]
        / beta
    )

    condition = Condition(
        dataset=groups["dataset"],
        dataset_slug=dataset_slug,
        family=family,
        propagation=groups["propagation"],
        model=groups["model"],
        beta=str(int(float(groups["beta"])) if float(groups["beta"]).is_integer() else groups["beta"]),
        num_clients=int(groups["num_clients"]),
        hop=hop,
        iterations=int(groups["iterations"]),
        t=groups["t"],
        alpha=groups["alpha"],
        use_pe=use_pe,
        purpose=purpose,
        source_dir=str(source_dir.relative_to(REPO_ROOT)),
        destination_dir=str(dest_dir.relative_to(REPO_ROOT)),
    )

    return condition


def mirror_condition_group(
    condition: Condition,
    source_dirs: list[Path],
    *,
    overwrite: bool,
) -> tuple[Condition, list[dict[str, object]], int]:
    dest_dir = REPO_ROOT / condition.destination_dir

    copied_rows: list[dict[str, object]] = []
    copied_bytes = 0
    accuracies: list[float] = []

    for source_dir in sorted(source_dirs):
        files = [p for p in sorted(source_dir.iterdir()) if p.is_file()]
        by_timestamp: dict[str, list[Path]] = defaultdict(list)
        loose_files: list[Path] = []
        for file_path in files:
            ts = timestamp_for(file_path)
            if ts == "undated":
                loose_files.append(file_path)
            else:
                by_timestamp[ts].append(file_path)

        for ts, timestamp_files in sorted(by_timestamp.items()):
            rep_dir = dest_dir / "reps" / f"run_{ts}"
            for file_path in timestamp_files:
                if file_path.name.startswith("training_"):
                    dest_name = "training.csv"
                    acc = final_accuracy_from_training_csv(file_path)
                    if acc is not None:
                        accuracies.append(acc)
                elif file_path.name.startswith("results_") and file_path.suffix == ".json":
                    dest_name = "results.json"
                elif file_path.name.startswith("results_") and file_path.suffix == ".txt":
                    dest_name = "results.txt"
                elif file_path.name.startswith("provenance_") and file_path.suffix == ".json":
                    dest_name = "provenance.json"
                else:
                    dest_name = file_path.name
                dest_path = rep_dir / dest_name
                copied_bytes += copy_file(file_path, dest_path, overwrite=overwrite)
                copied_rows.append(
                    {
                        "source_path": str(file_path.relative_to(REPO_ROOT)),
                        "registry_path": str(dest_path.relative_to(REPO_ROOT)),
                        "condition_path": str(dest_dir.relative_to(REPO_ROOT)),
                        "timestamp": ts,
                        "bytes": file_path.stat().st_size,
                    }
                )

        for file_path in loose_files:
            dest_path = dest_dir / "artifacts" / file_path.name
            copied_bytes += copy_file(file_path, dest_path, overwrite=overwrite)
            copied_rows.append(
                {
                    "source_path": str(file_path.relative_to(REPO_ROOT)),
                    "registry_path": str(dest_path.relative_to(REPO_ROOT)),
                    "condition_path": str(dest_dir.relative_to(REPO_ROOT)),
                    "timestamp": ts,
                    "bytes": file_path.stat().st_size,
                }
            )

    write_summary_csv(dest_dir / "summary.csv", condition, accuracies)
    write_json(
        dest_dir / "provenance.json",
        {
            **asdict(condition),
            "source_dirs": [str(path.relative_to(REPO_ROOT)) for path in sorted(source_dirs)],
            "source_files": [row["source_path"] for row in copied_rows],
            "n_source_dirs": len(source_dirs),
            "n_source_files": len(copied_rows),
            "n_training_reps": len(accuracies),
        },
    )
    (dest_dir / "source_paths.txt").write_text(
        "\n".join(str(row["source_path"]) for row in copied_rows) + ("\n" if copied_rows else "")
    )

    condition_row = {
        **asdict(condition),
        "source_dir": ";".join(str(path.relative_to(REPO_ROOT)) for path in sorted(source_dirs)),
    }
    return Condition(**condition_row), copied_rows, copied_bytes


def write_manifest_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Remove destination before rebuilding.")
    args = parser.parse_args()

    dest_root = args.dest.resolve()
    if args.clean and dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, tuple[Condition, list[Path]]] = {}
    for source_dir, match in find_condition_dirs(iter_source_roots(EXPERIMENTS_ROOT)):
        condition = condition_from_match(source_dir, match, dest_root)
        if condition.destination_dir not in grouped:
            grouped[condition.destination_dir] = (condition, [])
        grouped[condition.destination_dir][1].append(source_dir)

    conditions: list[Condition] = []
    file_rows: list[dict[str, object]] = []
    copied_bytes = 0

    for _destination_dir, (condition, source_dirs) in sorted(grouped.items()):
        condition, rows, n_bytes = mirror_condition_group(
            condition,
            source_dirs,
            overwrite=args.overwrite,
        )
        conditions.append(condition)
        file_rows.extend(rows)
        copied_bytes += n_bytes

    condition_rows = [asdict(condition) for condition in conditions]
    write_manifest_csv(dest_root / "manifest.csv", condition_rows)
    write_manifest_csv(dest_root / "source_file_manifest.csv", file_rows)
    (dest_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in condition_rows)
    )

    readme = f"""# Structured Result Sources

Generated by `experiments/build_structured_result_registry.py`.

This is a non-destructive copy of raw result artifacts into a stable layout:

```text
structured_sources/<family>/<dataset>/<model>/<hop>/<purpose>/<propagation>/<beta>/
```

Each condition directory contains:

- `summary.csv`: one tidy aggregate row computed from copied `training_*.csv` files. Metrics are final client training accuracy, not paper test accuracy.
- `provenance.json`: parsed condition metadata and original source paths.
- `source_paths.txt`: original files copied into this condition.
- `reps/run_<timestamp>/`: copied `training.csv`, `results.json`, and `results.txt` artifacts when present.
- `artifacts/`: condition-level files without timestamps.

Counts:

- conditions mirrored: {len(conditions)}
- source files copied/listed: {len(file_rows)}
- bytes copied this run: {copied_bytes}
"""
    (dest_root / "README.md").write_text(readme)

    print(f"Structured registry written to: {dest_root}")
    print(f"Conditions mirrored: {len(conditions)}")
    print(f"Source files listed: {len(file_rows)}")
    print(f"Bytes copied this run: {copied_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
