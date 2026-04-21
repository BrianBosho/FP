"""Legacy compatibility wrapper — real implementations live in `src.fedgnn.experiments`."""

from src.fedgnn.experiments import (
    run_experiments,
    parse_arguments,
    load_yaml_config,
    setup_environment_for_experiment,
    copy_training_csv_to_experiment_dir,
    format_time,
    save_summary_results,
    print_summary,
    create_example_config,
    run_simple_experiment,
    parse_args_simple,
    test_config_passing,
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
    "run_simple_experiment",
    "parse_args_simple",
    "test_config_passing",
]
