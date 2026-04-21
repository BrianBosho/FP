#!/usr/bin/env python3
"""Legacy wrapper — real implementation is in `src.fedgnn.experiments`."""

from src.fedgnn.experiments.run_experiments import (
    run_experiments,
    parse_arguments,
    load_yaml_config,
    setup_environment_for_experiment,
    copy_training_csv_to_experiment_dir,
    format_time,
    save_summary_results,
    print_summary,
    create_example_config,
)

__all__ = [
    "run_experiments",
    "parse_arguments",
    "load_yaml_config",
    "setup_environment_for_experiment",
    "copy_training_csv_to_experiment_dir",
    "format_time",
    "save_summary_results",
    "print_summary",
    "create_example_config",
]

if __name__ == "__main__":
    args = parse_arguments()
    if args.config == "generate_example":
        create_example_config()
        exit(0)
    summary_rows, all_results = run_experiments(args)
    print_summary(summary_rows)
