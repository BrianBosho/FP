"""Experiment entrypoints and runners (Phase C migration target)."""

from .run_experiments import (
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
from .run_simple import run_simple_experiment, parse_arguments as parse_args_simple
from .test_config import test_config_passing

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
    "run_simple_experiment",
    "parse_args_simple",
    "test_config_passing",
]
