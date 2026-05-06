from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_resolve_data_loading_device_prefers_cpu_for_memory_bound_cuda_runs():
    pytest.importorskip("torch")
    from src.fedgnn.fl.run import _resolve_data_loading_device

    resolved = _resolve_data_loading_device(
        "cuda",
        {
            "keep_data_on_gpu": False,
            "feature_prop_device": "cpu",
        },
    )

    assert str(resolved) == "cpu"


def test_run_experiments_reuses_one_ray_runtime_across_conditions(tmp_path, monkeypatch):
    pytest.importorskip("ray")
    pytest.importorskip("torch")
    import src.fedgnn.experiments.run_experiments as run_experiments_module

    config = {
        "num_clients": [2],
        "beta": [0.5, 1.0],
        "datasets": ["Cora"],
        "data_loading": ["adjacency"],
        "models": ["GCN"],
        "num_rounds": 1,
        "epochs": 1,
        "lr": 0.1,
        "optimizer": "SGD",
        "decay": 0.0,
        "results_dir": str(tmp_path / "results"),
        "save_results": False,
        "hop": 1,
        "fulltraining_flag": False,
        "use_pe": [False],
        "device": "cpu",
    }
    args = SimpleNamespace(
        config="dummy.yaml",
        clients=None,
        rounds=None,
        epochs=None,
        beta=None,
        lr=None,
        optimizer=None,
        decay=None,
        datasets=None,
        data_loading=None,
        models=None,
        hop=None,
        save_results=False,
        results_dir=None,
        fulltraining_flag=False,
        repetitions=None,
        ray_port=None,
        use_pe=None,
    )

    init_calls = []
    main_calls = []
    shutdown_calls = []
    ray_state = {"initialized": True}

    monkeypatch.setattr(run_experiments_module, "load_yaml_config", lambda _: config)
    monkeypatch.setattr(
        run_experiments_module,
        "resolve_results_and_summary_dirs",
        lambda _: SimpleNamespace(
            results_dir=Path(tmp_path / "results"),
            summary_dir=Path(tmp_path / "summary"),
        ),
    )
    monkeypatch.setattr(
        run_experiments_module,
        "setup_environment_for_experiment",
        lambda *args, **kwargs: (str(tmp_path / "exp"), "exp"),
    )
    monkeypatch.setattr(
        run_experiments_module,
        "save_summary_results",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        run_experiments_module,
        "copy_training_csv_to_experiment_dir",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(run_experiments_module.wandb, "run", None)

    def fake_ensure_ray_initialized(cfg, using_cuda_device):
        init_calls.append((cfg["device"], using_cuda_device))
        ray_state["initialized"] = True
        return True

    def fake_main_experiment(*args, **kwargs):
        main_calls.append(kwargs["manage_ray_lifecycle"])
        return (
            {
                "summary": {
                    "average_global_result": 0.8,
                    "average_client_result": 0.7,
                    "std_global": 0.0,
                    "std_client": 0.0,
                }
            },
            "ok",
        )

    def fake_ray_shutdown():
        shutdown_calls.append("shutdown")
        ray_state["initialized"] = False

    monkeypatch.setattr(run_experiments_module, "ensure_ray_initialized", fake_ensure_ray_initialized)
    monkeypatch.setattr(run_experiments_module, "main_experiment", fake_main_experiment)
    monkeypatch.setattr(run_experiments_module.ray, "shutdown", fake_ray_shutdown)
    monkeypatch.setattr(run_experiments_module.ray, "is_initialized", lambda: ray_state["initialized"])

    summary_rows, all_results = run_experiments_module.run_experiments(args)

    assert len(init_calls) == 1
    assert main_calls == [False, False]
    assert len(shutdown_calls) == 2
    assert len(summary_rows) == 2
    assert len(all_results) == 2
