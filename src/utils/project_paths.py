"""Legacy compatibility wrapper — real implementation is in `src.fedgnn.utils.project_paths`."""

from src.fedgnn.utils.project_paths import (  # noqa: F401
    ResolvedResultsPaths,
    find_repo_root,
    resolve_results_and_summary_dirs,
)

__all__ = [
    "ResolvedResultsPaths",
    "find_repo_root",
    "resolve_results_and_summary_dirs",
]
