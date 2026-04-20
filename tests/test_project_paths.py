from pathlib import Path

from src.utils.project_paths import find_repo_root, resolve_results_and_summary_dirs


def test_find_repo_root_contains_conf_base():
    repo_root = find_repo_root()
    assert (repo_root / "conf" / "base.yaml").exists()


def test_default_results_dir_is_repo_local():
    repo_root = find_repo_root()
    resolved = resolve_results_and_summary_dirs(None, repo_root=repo_root)
    assert resolved.is_results_dir_inside_repo is True
    assert resolved.results_dir == (repo_root / "runs" / "experiments").resolve()
    assert resolved.summary_dir == (repo_root / "results_summary" / resolved.results_dir.name).resolve()


def test_relative_override_is_repo_local():
    repo_root = find_repo_root()
    resolved = resolve_results_and_summary_dirs("runs/smoke", repo_root=repo_root)
    assert resolved.is_results_dir_inside_repo is True
    assert resolved.results_dir == (repo_root / "runs" / "smoke").resolve()


def test_absolute_override_outside_repo_puts_summary_next_to_results(tmp_path: Path):
    repo_root = find_repo_root()
    out = tmp_path / "external_results"
    resolved = resolve_results_and_summary_dirs(str(out), repo_root=repo_root)
    assert resolved.is_results_dir_inside_repo is False
    assert resolved.results_dir == out.resolve()
    assert resolved.summary_dir == (out.parent / "results_summary" / out.name).resolve()

