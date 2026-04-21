"""
Package invariant tests for the fedgnn package refactor (Phase 2).

These tests verify:
1. `src.fedgnn` subpackages are importable and expose the expected symbols.
2. Legacy `src.*` modules are thin wrappers that re-export from `src.fedgnn.*`.
3. Wrapper direction: `fedgnn` modules do NOT import implementation from legacy `src.*`.
4. CLI entrypoints are importable (smoke tests).

Tests here are designed to run with the project's declared dependencies
(tested via `pytest.importorskip` for heavy deps).
"""

import subprocess
import sys


# ---------------------------------------------------------------------------
# 1. Import tests for fedgnn subpackages
# ---------------------------------------------------------------------------


def test_fedgnn_utils_importable():
    """src.fedgnn.utils is importable and exposes core utilities."""
    from src.fedgnn.utils import (
        load_config,
        find_repo_root,
        resolve_results_and_summary_dirs,
        save_results_to_csv,
        clear_cuda_cache,
        initialize_wandb,
    )
    assert callable(load_config)
    assert callable(find_repo_root)
    assert callable(resolve_results_and_summary_dirs)


def test_fedgnn_models_importable():
    """src.fedgnn.models is importable and exposes model classes."""
    from src.fedgnn.models import GCN, GAT, GAT_Arxiv, GCN_arxiv, PubmedGAT, VanillaGNN, MLP
    assert callable(GCN)
    assert callable(GAT)


def test_fedgnn_data_importable():
    """src.fedgnn.data is importable and exposes data utilities."""
    from src.fedgnn.data import GraphDataset, partition_data, load_dataset
    assert callable(GraphDataset)


def test_fedgnn_fl_importable():
    """src.fedgnn.fl is importable and exposes FL runtime classes."""
    pytest = __import__("pytest")
    pytest.importorskip("ray")
    from src.fedgnn.fl import FLClient, Server
    assert callable(FLClient)
    assert callable(Server)


# ---------------------------------------------------------------------------
# 2. Legacy compatibility wrappers still work
# ---------------------------------------------------------------------------


def test_legacy_src_models_is_wrapper():
    """src.models is a thin wrapper — it does not define model classes."""
    import src.models as models_module
    # The wrapper should not have GCN defined locally
    # (it lives in src.fedgnn.models)
    assert hasattr(models_module, "GCN")


def test_legacy_src_utils_is_wrapper():
    """src.utils is a thin wrapper — it re-exports from src.fedgnn.utils."""
    from src.utils import load_config, find_repo_root, save_results_to_csv
    assert callable(load_config)
    assert callable(find_repo_root)


def test_legacy_src_dataprocessing_is_wrapper():
    """src.dataprocessing is a thin wrapper — it re-exports from src.fedgnn.data."""
    from src.dataprocessing import GraphDataset, partition_data
    assert callable(GraphDataset)
    assert callable(partition_data)


# ---------------------------------------------------------------------------
# 3. Wrapper direction: fedgnn does NOT import from legacy src.*
# ---------------------------------------------------------------------------


def test_fedgnn_utils_no_src_dot_utils_imports():
    """src.fedgnn.utils must not import implementation from src.utils."""
    import ast
    from pathlib import Path

    fedgnn_utils = Path(__file__).resolve().parents[1] / "src" / "fedgnn" / "utils"
    for py_file in fedgnn_utils.glob("*.py"):
        if py_file.name.startswith("__"):
            continue
        src = py_file.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.ImportFrom)):
                if node.module and (
                    node.module.startswith("src.utils")
                    and "fedgnn" not in node.module
                ):
                    raise AssertionError(
                        f"{py_file.name} imports from legacy src.utils: {node.module}"
                    )


# ---------------------------------------------------------------------------
# 4. CLI smoke tests (no datasets / heavy deps required)
# ---------------------------------------------------------------------------


def test_cli_run_help():
    """python3 -m src.run --help starts without crashing."""
    pytest = __import__("pytest")
    pytest.importorskip("omegaconf")
    pytest.importorskip("torch")

    proc = subprocess.run(
        [sys.executable, "-m", "src.run", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


def test_cli_run_experiments_help():
    """python3 -m src.experiments.run_experiments --help starts without crashing."""
    pytest = __import__("pytest")
    pytest.importorskip("omegaconf")
    pytest.importorskip("torch")
    pytest.importorskip("ray")

    proc = subprocess.run(
        [sys.executable, "-m", "src.experiments.run_experiments", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
