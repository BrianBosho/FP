"""
Compatibility re-exports for the Phase C migration.

In Phase C4 we introduce `fedgnn.*` import paths while keeping the existing
`src.*` modules as the implementation source. Later phases can move the
implementation into `fedgnn/` and keep `src/` as wrappers.
"""

from src.utils.project_paths import (  # noqa: F401
    ResolvedResultsPaths,
    find_repo_root,
    resolve_results_and_summary_dirs,
)

__all__ = [
    "ResolvedResultsPaths",
    "find_repo_root",
    "resolve_results_and_summary_dirs",
]

