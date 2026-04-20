import subprocess
import sys

import pytest


def test_run_experiments_generate_example_cli_smoke():
    """
    Smoke test: verify the CLI entrypoint starts and can generate an example config.

    This avoids any training and primarily checks that imports + argparse wiring are intact.
    Skips if heavy runtime deps aren't installed in the current environment.
    """
    pytest.importorskip("omegaconf")
    pytest.importorskip("torch")
    pytest.importorskip("ray")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.experiments.run_experiments",
            "--config",
            "generate_example",
        ],
        cwd=".",
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

