from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class ResolvedResultsPaths:
    repo_root: Path
    results_dir: Path
    summary_dir: Path
    is_results_dir_inside_repo: bool


def find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Locate the federated-gnn repository root by walking parents until we find `conf/base.yaml`.
    Falls back to the directory containing this file's grandparent if not found.
    """
    if start is None:
        start = Path(__file__).resolve()

    for p in [start, *start.parents]:
        if (p / "conf" / "base.yaml").exists():
            return p

    # Fallback: src/utils/project_paths.py -> src/utils -> src -> repo_root
    return Path(__file__).resolve().parents[2]


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def resolve_results_and_summary_dirs(
    results_dir: Optional[str],
    *,
    default_results_subdir: str = "runs/experiments",
    repo_root: Optional[Path] = None,
) -> ResolvedResultsPaths:
    """
    Resolve results_dir and summary_dir in a way that:
    - keeps defaults repo-local (no ../ paths)
    - respects user overrides (CLI/YAML)
    - preserves legacy behavior when results_dir points outside the repo
    """
    if repo_root is None:
        repo_root = find_repo_root()

    if results_dir is None or str(results_dir).strip() == "":
        results_dir_path = repo_root / default_results_subdir
    else:
        rd = Path(str(results_dir))
        results_dir_path = rd if rd.is_absolute() else (repo_root / rd)

    results_dir_path = results_dir_path.resolve()
    inside_repo = _is_within(results_dir_path, repo_root)

    # Summary directory:
    # - If results are inside the repo, keep summaries under <repo>/results_summary/<results_dir_name>
    # - If results are outside the repo, keep summaries adjacent to results_dir (legacy-friendly)
    if inside_repo:
        summary_dir = (repo_root / "results_summary" / results_dir_path.name).resolve()
    else:
        summary_dir = (results_dir_path.parent / "results_summary" / results_dir_path.name).resolve()

    return ResolvedResultsPaths(
        repo_root=repo_root.resolve(),
        results_dir=results_dir_path,
        summary_dir=summary_dir,
        is_results_dir_inside_repo=inside_repo,
    )

