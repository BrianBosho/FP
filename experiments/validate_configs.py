#!/usr/bin/env python3
"""Validate Track A experiment YAMLs before launching runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fedgnn.utils.config import load_config  # noqa: E402


CODE_SUPPORTED_DATASETS = {
    "Cora",
    "Citeseer",
    "Pubmed",
    "ogbn-arxiv",
    "ogbn-products",
    "Computers",
    "Photo",
    "FacebookPagePage",
    "Texas",
    "Wisconsin",
}


def as_list(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def validate_config(path: Path) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []
    path = path.resolve()

    try:
        raw = OmegaConf.load(str(path))
    except Exception as exc:
        return [f"raw YAML load failed: {exc}"], warnings, {}

    try:
        cfg = load_config(str(path))
    except Exception as exc:
        return [f"merged config load failed: {exc}"], warnings, {}

    required = [
        "datasets",
        "num_clients",
        "num_rounds",
        "beta",
        "data_loading",
        "models",
        "use_pe",
        "repetitions",
        "hop",
        "lr",
        "optimizer",
        "decay",
        "epochs",
        "early_stopping_patience",
        "num_iterations",
        "diffusion_t",
        "aggregation",
    ]
    for key in required:
        if key not in cfg or cfg.get(key) is None:
            errors.append(f"missing resolved key: {key}")

    training = raw.get("training")
    if training is not None:
        comparisons = [
            ("lr", "lr"),
            ("optimizer", "optimizer"),
            ("weight_decay", "decay"),
            ("epochs", "epochs"),
            ("patience", "early_stopping_patience"),
        ]
        for src_key, dst_key in comparisons:
            if src_key in training and cfg.get(dst_key) != training.get(src_key):
                errors.append(
                    f"training.{src_key}={training.get(src_key)!r} "
                    f"did not resolve to {dst_key}={cfg.get(dst_key)!r}"
                )

    datasets = [str(x) for x in as_list(cfg.get("datasets", []))]
    unsupported = sorted(set(datasets) - CODE_SUPPORTED_DATASETS)
    if unsupported:
        warnings.append(
            "dataset loader support not present yet: " + ", ".join(unsupported)
        )

    summary = {
        "path": str(path.relative_to(REPO_ROOT)),
        "datasets": ",".join(datasets),
        "models": ",".join(str(x) for x in as_list(cfg.get("models", []))),
        "loaders": ",".join(str(x) for x in as_list(cfg.get("data_loading", []))),
        "beta": ",".join(str(x) for x in as_list(cfg.get("beta", []))),
        "clients": ",".join(str(x) for x in as_list(cfg.get("num_clients", []))),
        "repetitions": str(cfg.get("repetitions")),
        "optimizer": str(cfg.get("optimizer")),
        "lr": str(cfg.get("lr")),
        "decay": str(cfg.get("decay")),
        "epochs": str(cfg.get("epochs")),
        "patience": str(cfg.get("early_stopping_patience")),
    }
    return errors, warnings, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Track A experiment configs")
    parser.add_argument("--config", type=str, default=None,
                        help="Validate one YAML file instead of all configs")
    parser.add_argument("--strict-datasets", action="store_true",
                        help="Treat unsupported dataset warnings as errors")
    args = parser.parse_args()

    paths = [Path(args.config)] if args.config else sorted(CONFIGS_DIR.rglob("*.yaml"))
    if not paths:
        print("No configs found.")
        return 1

    total_errors = 0
    total_warnings = 0
    print("Track A Config Validation")
    print("-" * 100)
    for path in paths:
        errors, warnings, summary = validate_config(path)
        if args.strict_datasets and warnings:
            errors.extend(warnings)
            warnings = []
        total_errors += len(errors)
        total_warnings += len(warnings)

        status = "OK" if not errors else "FAIL"
        if summary:
            print(
                f"{status:4s} {summary['path']} | "
                f"ds={summary['datasets']} model={summary['models']} "
                f"loader={summary['loaders']} beta={summary['beta']} K={summary['clients']} "
                f"reps={summary['repetitions']} opt={summary['optimizer']} "
                f"lr={summary['lr']} decay={summary['decay']} "
                f"epochs={summary['epochs']} patience={summary['patience']}"
            )
        else:
            print(f"{status:4s} {path}")

        for warning in warnings:
            print(f"     WARN: {warning}")
        for error in errors:
            print(f"     ERROR: {error}")

    print("-" * 100)
    print(f"Validated {len(paths)} config(s): {total_errors} error(s), {total_warnings} warning(s)")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
