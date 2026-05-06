"""Experiment entrypoints and runners (Phase C migration target)."""

# NOTE: we intentionally do NOT re-export `run_experiments` (the function) here
# because it would shadow the `run_experiments` *submodule*, breaking direct
# module imports such as `import src.fedgnn.experiments.run_experiments`.
from .run_experiments import (
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
from .preflight import run_preflight
from .ledger import RunLedger, RunPacket, make_condition_key
from .staged_policy import (
    smoke_overrides,
    pilot_overrides,
    should_promote_to_full,
    ci_95,
    format_ci,
    enrich_summary_with_ci,
)

__all__ = [
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
    "run_preflight",
    "RunLedger",
    "RunPacket",
    "make_condition_key",
    "smoke_overrides",
    "pilot_overrides",
    "should_promote_to_full",
    "ci_95",
    "format_ci",
    "enrich_summary_with_ci",
]
