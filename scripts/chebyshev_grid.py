import argparse
import copy
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(cfg: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_experiment(temp_cfg_path: Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "src.experiments.run_experiments",
        "--config",
        str(temp_cfg_path),
    ]
    print("Running:", " ".join(cmd))
    start = time.time()
    try:
        result = subprocess.run(cmd, check=False)
        code = result.returncode
    except Exception as e:
        print("ERROR running experiment:", e)
        code = 1
    dur = time.time() - start
    print(f"Finished in {dur:.1f}s with code={code}\n")
    return code


def main():
    parser = argparse.ArgumentParser(description="Chebyshev grid search helper")
    parser.add_argument(
        "--base_config",
        type=str,
        default=str(Path("conf/ablation/ogbn-arxiv_config.yaml")),
        help="Path to base YAML config to clone and modify",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="*",
        default=["chebyshev_diffusion", "chebyshev_diffusion_operator"],
        help="Which chebyshev modes to evaluate",
    )
    parser.add_argument(
        "--t_vals",
        type=float,
        nargs="*",
        default=[0.01, 0.05, 0.1, 0.2],
        help="Diffusion time values to sweep",
    )
    parser.add_argument(
        "--k_vals",
        type=int,
        nargs="*",
        default=[3, 5, 8, 10],
        help="Chebyshev order values to sweep",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override num_rounds (optional)",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=None,
        help="Override number of clients (optional)",
    )
    args = parser.parse_args()

    base_cfg_path = Path(args.base_config)
    base_cfg = load_yaml(base_cfg_path)

    # Ensure keys exist
    base_cfg.setdefault("data_loading", ["chebyshev_diffusion"]) 
    base_cfg.setdefault("datasets", ["Cora"])  

    # Prepare output directory for temp configs
    tmp_dir = Path(tempfile.gettempdir()) / "chebyshev_sweep_cfgs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    total = 0

    for mode in args.modes:
        for t in args.t_vals:
            for k in args.k_vals:
                total += 1
                cfg = copy.deepcopy(base_cfg)
                cfg["data_loading"] = [mode]
                cfg["chebyshev_t"] = float(t)
                cfg["chebyshev_k"] = int(k)
                if args.rounds is not None:
                    cfg["num_rounds"] = int(args.rounds)
                if args.clients is not None:
                    cfg["num_clients"] = [int(args.clients)]

                tag = f"{mode}_t{t}_k{k}".replace(".", "p")
                temp_cfg_path = tmp_dir / f"{base_cfg_path.stem}_{tag}.yaml"
                dump_yaml(cfg, temp_cfg_path)

                print("=== RUN:", tag)
                print("- config:", temp_cfg_path)
                code = run_experiment(temp_cfg_path)
                if code != 0:
                    failures += 1

    print(f"Grid done. total={total}, failures={failures}")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()


