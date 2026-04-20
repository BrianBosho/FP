import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf


def test_base_and_dataset_config_merge():
    """
    Lightweight config merge test that avoids importing training code.
    """
    base = OmegaConf.load("conf/base.yaml")
    dataset = OmegaConf.load("conf/cora_config.yaml")
    merged = OmegaConf.merge(base, dataset)

    # Dataset config should override base
    assert merged["results_dir"] == "results/IEEE/cora"

    # Base config should still provide defaults
    assert "paths" in merged
    assert "datasets_dir" in merged["paths"]

