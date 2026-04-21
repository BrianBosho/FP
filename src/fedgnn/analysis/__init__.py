"""Analysis utilities — results parsing, plotting, table formatting."""

from .results import (
    process_results_folder_json_v2,
    process_results_folder_json,
)

from .training_logs import (
    parse_client_csv,
    process_fp_logs,
)

from .plots import (
    plot_round_loss,
    plot_round_accuracy,
    plot_training_curves,
    plot_federated_comparison,
    plot_energy_dynamics,
    plot_missing_rate,
)

from .tables import (
    format_results_table,
    summary_table,
    export_table,
)

__all__ = [
    # results
    "process_results_folder_json_v2",
    "process_results_folder_json",
    # training_logs
    "parse_client_csv",
    "process_fp_logs",
    # plots
    "plot_round_loss",
    "plot_round_accuracy",
    "plot_training_curves",
    "plot_federated_comparison",
    "plot_energy_dynamics",
    "plot_missing_rate",
    # tables
    "format_results_table",
    "summary_table",
    "export_table",
]