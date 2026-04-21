#!/usr/bin/env python3
"""Legacy wrapper — real implementation is in `src.fedgnn.experiments`."""

from src.fedgnn.experiments.run_simple import (
    run_simple_experiment,
    parse_arguments,
    load_yaml_config,
    setup_environment_for_experiment,
    copy_training_csv_to_experiment_dir,
    format_time,
    create_example_config,
)

__all__ = [
    "run_simple_experiment",
    "parse_arguments",
    "load_yaml_config",
    "setup_environment_for_experiment",
    "copy_training_csv_to_experiment_dir",
    "format_time",
    "create_example_config",
]

if __name__ == "__main__":
    args = parse_arguments()
    if args.config == "generate_example":
        create_example_config()
        exit(0)
    run_simple_experiment(args)
