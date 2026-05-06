#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments" / "propagator_eval"
STATUS_PATH = EXP_DIR / "SEQUENTIAL_RUNNER_STATUS.md"
TMP_CFG_DIR = EXP_DIR / ".tmp_runner_configs"
LOG_DIR = EXP_DIR / "results" / "sequential_runner_logs"
PYTHON_BIN = Path("/home/bosho/.conda/envs/fedgnn/bin/python")
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path(sys.executable)

PHASE3_OPERATORS = ["zero_hop", "full", "adjacency", "asymmetric_random_walk", "diffusion", "chebyshev_diffusion", "appnp"]
PHASE4_OPERATORS = ["zero_hop", "full", "adjacency", "asymmetric_random_walk", "diffusion", "chebyshev_diffusion", "appnp"]
PHASE6_OPERATORS = ["zero_hop", "adjacency", "diffusion", "appnp"]


@dataclass
class Step:
    label: str
    phase: str
    policy: str
    kind: str  # intrinsic | downstream | compute_gaps
    config_path: Path
    cli_args: dict[str, Any] = field(default_factory=dict)
    patches: dict[str, Any] = field(default_factory=dict)
    output_path: Path | None = None
    log_name: str | None = None

    def done(self) -> bool:
        if self.kind == "compute_gaps":
            return False
        return self.output_path is not None and self.output_path.exists()


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def step_log_path(step: Step) -> Path:
    name = step.log_name or slugify(step.label)
    return LOG_DIR / f"{name}.log"


def build_steps() -> list[Step]:
    steps: list[Step] = []

    # Phase 1 companion: exact heat-kernel reference on Cora
    cfg = EXP_DIR / "configs" / "phase_1_cora_intrinsic_heat_kernel.yaml"
    out = EXP_DIR / "results" / "phase_1_cora_intrinsic" / "raw" / "heat_kernel_exact" / "cora" / "beta10000_seed0.json"
    steps.append(Step(
        label="phase1_heat_kernel_cora_beta10000_seed0",
        phase="Phase 1",
        policy="heat_kernel_cpu",
        kind="intrinsic",
        config_path=cfg,
        cli_args={"operator": "heat_kernel_exact", "dataset": "Cora", "beta": 10000, "seed": 0},
        output_path=out,
    ))

    # Phase 2: APPNP alpha sweep
    cfg = EXP_DIR / "configs" / "phase_2_cora_ablation_appnp_alpha.yaml"
    for alpha in [0.05, 0.1, 0.2]:
        alpha_tag = str(alpha).replace(".", "")
        result_root = EXP_DIR / "results" / "phase_2_cora_ablation" / "raw" / "appnp_alpha" / f"alpha_{alpha_tag}"
        for seed in [0, 1, 2]:
            out = result_root / "appnp" / "cora" / f"beta10000_seed{seed}.json"
            steps.append(Step(
                label=f"phase2_appnp_alpha_{alpha}_seed{seed}",
                phase="Phase 2",
                policy="intrinsic_small",
                kind="intrinsic",
                config_path=cfg,
                cli_args={"operator": "appnp", "dataset": "Cora", "beta": 10000, "seed": seed},
                patches={"appnp_alpha": alpha, "results_dir": str(result_root)},
                output_path=out,
            ))

    # Phase 2: epsilon sweep on Cora
    cfg = EXP_DIR / "configs" / "phase_2_cora_ablation_epsilon_cora.yaml"
    for tol in [1.0e-2, 1.0e-3, 1.0e-4]:
        tol_tag = f"tol_{tol:.0e}".replace("+", "")
        result_root = EXP_DIR / "results" / "phase_2_cora_ablation" / "raw" / "epsilon_cora" / tol_tag
        for seed in [0, 1, 2]:
            out = result_root / "diffusion" / "cora" / f"beta10000_seed{seed}.json"
            steps.append(Step(
                label=f"phase2_epsilon_{tol}_seed{seed}",
                phase="Phase 2",
                policy="intrinsic_small",
                kind="intrinsic",
                config_path=cfg,
                cli_args={"operator": "diffusion", "dataset": "Cora", "beta": 10000, "seed": seed},
                patches={"feature_prop_tolerance": tol, "results_dir": str(result_root)},
                output_path=out,
            ))

    # Phase 2: hop depth comparison
    cfg = EXP_DIR / "configs" / "phase_2_cora_ablation_hop_depth.yaml"
    for hop in [1, 2]:
        result_root = EXP_DIR / "results" / "phase_2_cora_ablation" / "raw" / "hop_depth" / f"hop{hop}"
        for op in ["adjacency", "diffusion"]:
            for seed in [0, 1, 2]:
                out = result_root / op / "cora" / f"beta10000_seed{seed}.json"
                steps.append(Step(
                    label=f"phase2_hop{hop}_{op}_seed{seed}",
                    phase="Phase 2",
                    policy="intrinsic_small",
                    kind="intrinsic",
                    config_path=cfg,
                    cli_args={"operator": op, "dataset": "Cora", "beta": 10000, "seed": seed},
                    patches={"hop": hop, "results_dir": str(result_root)},
                    output_path=out,
                ))

    # Phase 3: Cora downstream
    cfg = EXP_DIR / "configs" / "phase_3_cora_downstream.yaml"
    for op in PHASE3_OPERATORS:
        for beta in [10000, 1]:
            for bb in ["GCN", "GAT"]:
                for seed in [0, 1, 2, 3, 4]:
                    out = EXP_DIR / "results" / "phase_3_cora_downstream" / "raw" / op / "cora" / f"beta{beta}_seed{seed}_{bb.lower()}.json"
                    steps.append(Step(
                        label=f"phase3_{op}_cora_beta{beta}_{bb.lower()}_seed{seed}",
                        phase="Phase 3",
                        policy="downstream",
                        kind="downstream",
                        config_path=cfg,
                        cli_args={"operator": op, "dataset": "Cora", "beta": beta, "backbone": bb, "seed": seed},
                        output_path=out,
                    ))
    steps.append(Step(
        label="phase3_compute_gaps",
        phase="Phase 3",
        policy="downstream",
        kind="compute_gaps",
        config_path=cfg,
    ))

    # Phase 4: homophilic reproduction
    cfg = EXP_DIR / "configs" / "phase_4_homophilic_reproduction.yaml"
    for ds in ["Citeseer", "Pubmed"]:
        for op in PHASE4_OPERATORS:
            for beta in [10000, 1]:
                for bb in ["GCN", "GAT"]:
                    for seed in [0, 1, 2, 3, 4]:
                        out = EXP_DIR / "results" / "phase_4_homophilic_reproduction" / "raw" / op / ds.lower() / f"beta{beta}_seed{seed}_{bb.lower()}.json"
                        steps.append(Step(
                            label=f"phase4_{op}_{ds.lower()}_beta{beta}_{bb.lower()}_seed{seed}",
                            phase="Phase 4",
                            policy="downstream",
                            kind="downstream",
                            config_path=cfg,
                            cli_args={"operator": op, "dataset": ds, "beta": beta, "backbone": bb, "seed": seed},
                            output_path=out,
                        ))
    steps.append(Step(
        label="phase4_compute_gaps",
        phase="Phase 4",
        policy="downstream",
        kind="compute_gaps",
        config_path=cfg,
    ))

    # Phase 5: OGBN-Arxiv intrinsic scalability
    cfg = EXP_DIR / "configs" / "phase_5_scalability_ogbn_arxiv.yaml"
    for op in ["adjacency", "asymmetric_random_walk", "diffusion", "chebyshev_diffusion", "appnp"]:
        for beta in [10000, 1]:
            for seed in [0, 1, 2]:
                out = EXP_DIR / "results" / "phase_5_scalability_ogbn_arxiv" / "raw" / op / "ogbn-arxiv" / f"beta{beta}_seed{seed}.json"
                steps.append(Step(
                    label=f"phase5_{op}_ogbn_beta{beta}_seed{seed}",
                    phase="Phase 5",
                    policy="intrinsic_large",
                    kind="intrinsic",
                    config_path=cfg,
                    cli_args={"operator": op, "dataset": "ogbn-arxiv", "beta": beta, "seed": seed},
                    output_path=out,
                ))

    # Phase 6: heterophily stress
    cfg = EXP_DIR / "configs" / "phase_6_heterophily_stress.yaml"
    for ds in ["Texas", "Wisconsin"]:
        for op in PHASE6_OPERATORS:
            for seed in [0, 1, 2, 3, 4]:
                out = EXP_DIR / "results" / "phase_6_heterophily_stress" / "raw" / op / ds.lower() / f"beta10000_seed{seed}_gcn.json"
                steps.append(Step(
                    label=f"phase6_{op}_{ds.lower()}_seed{seed}",
                    phase="Phase 6",
                    policy="downstream",
                    kind="downstream",
                    config_path=cfg,
                    cli_args={"operator": op, "dataset": ds, "beta": 10000, "backbone": "GCN", "seed": seed},
                    output_path=out,
                ))

    return steps


POLICIES = {
    "intrinsic_small": {
        "min_gpu_free_mb": 22000,
        "min_ram_avail_gb": 20,
        "max_load1": 18.0,
        "max_gpu_util": 40.0,
        "max_external_fedgnn_jobs": 2,
    },
    "heat_kernel_cpu": {
        "min_ram_avail_gb": 20,
        "max_load1": 16.0,
        "max_external_fedgnn_jobs": 2,
    },
    "intrinsic_large": {
        "min_gpu_free_mb": 26000,
        "min_ram_avail_gb": 24,
        "max_load1": 14.0,
        "max_gpu_util": 30.0,
        "max_external_fedgnn_jobs": 1,
    },
    "downstream": {
        "min_gpu_free_mb": 30000,
        "min_ram_avail_gb": 28,
        "max_load1": 10.0,
        "max_gpu_util": 20.0,
        "max_external_fedgnn_jobs": 0,
    },
}


def read_meminfo() -> dict[str, int]:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            out[key] = int(value.strip().split()[0])
    return out


def query_gpu() -> dict[str, float | int | None]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        used, free, util_gpu, util_mem = [float(x.strip()) for x in raw.split(",")]
        return {
            "gpu_used_mb": int(used),
            "gpu_free_mb": int(free),
            "gpu_util": util_gpu,
            "gpu_mem_util": util_mem,
        }
    except Exception:
        return {"gpu_used_mb": None, "gpu_free_mb": None, "gpu_util": None, "gpu_mem_util": None}


def count_external_fedgnn_jobs() -> int:
    try:
        raw = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    except Exception:
        return 0
    count = 0
    my_pid = os.getpid()
    for line in raw.splitlines()[1:]:
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid, cmd = parts
        try:
            pid = int(pid)
        except Exception:
            continue
        if pid == my_pid:
            continue
        if any(token in cmd for token in [
            "src.experiments.run_experiments",
            "ray::FLClient.train_client",
            "src.fedgnn.experiments.run_downstream_eval",
        ]):
            count += 1
    return count


def snapshot() -> dict[str, Any]:
    mem = read_meminfo()
    gpu = query_gpu()
    load1, load5, load15 = os.getloadavg()
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "ram_avail_gb": round(mem.get("MemAvailable", 0) / 1024 / 1024, 2),
        "ram_free_gb": round(mem.get("MemFree", 0) / 1024 / 1024, 2),
        "load1": round(load1, 2),
        "load5": round(load5, 2),
        "load15": round(load15, 2),
        "external_fedgnn_jobs": count_external_fedgnn_jobs(),
        **gpu,
    }


def is_safe(policy_name: str, snap: dict[str, Any]) -> tuple[bool, list[str]]:
    policy = POLICIES[policy_name]
    reasons = []
    if "min_gpu_free_mb" in policy and snap.get("gpu_free_mb") is not None and snap["gpu_free_mb"] < policy["min_gpu_free_mb"]:
        reasons.append(f"gpu_free_mb<{policy['min_gpu_free_mb']}")
    if "max_gpu_util" in policy and snap.get("gpu_util") is not None and snap["gpu_util"] > policy["max_gpu_util"]:
        reasons.append(f"gpu_util>{policy['max_gpu_util']}")
    if snap["ram_avail_gb"] < policy["min_ram_avail_gb"]:
        reasons.append(f"ram_avail_gb<{policy['min_ram_avail_gb']}")
    if snap["load1"] > policy["max_load1"]:
        reasons.append(f"load1>{policy['max_load1']}")
    if snap["external_fedgnn_jobs"] > policy["max_external_fedgnn_jobs"]:
        reasons.append(f"external_fedgnn_jobs>{policy['max_external_fedgnn_jobs']}")
    return (len(reasons) == 0, reasons)


def phase_counts(steps: list[Step]) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = {}
    for step in steps:
        if step.kind == "compute_gaps":
            continue
        done = 1 if step.done() else 0
        counts.setdefault(step.phase, [0, 0])
        counts[step.phase][0] += done
        counts[step.phase][1] += 1
    return {k: (v[0], v[1] - v[0]) for k, v in counts.items()}


def write_status(steps: list[Step], current: Step | None, state: str, details: str, snap: dict[str, Any]) -> None:
    counts = phase_counts(steps)
    done_total = sum(done for done, rem in counts.values())
    total_total = sum(done + rem for done, rem in counts.values())
    lines = [
        "# Sequential Runner Status\n",
        f"_Updated: {snap['time']}_\n",
        "## Runner overview\n",
        f"- State: **{state}**\n",
        f"- Current step: `{current.label}`\n" if current else "- Current step: _(none)_\n",
        f"- Details: {details}\n",
        f"- Overall progress: **{done_total}/{total_total}**\n",
        "\n## Resource snapshot\n",
        "| Metric | Value |\n",
        "|---|---|\n",
        f"| GPU free | {snap.get('gpu_free_mb')} MiB |\n",
        f"| GPU util | {snap.get('gpu_util')}% |\n",
        f"| RAM available | {snap.get('ram_avail_gb')} GiB |\n",
        f"| Load (1/5/15) | {snap.get('load1')} / {snap.get('load5')} / {snap.get('load15')} |\n",
        f"| External FedGNN jobs | {snap.get('external_fedgnn_jobs')} |\n",
        "\n## Phase progress\n",
        "| Phase | Done | Remaining |\n",
        "|---|---:|---:|\n",
    ]
    for phase, (done, remaining) in sorted(counts.items()):
        lines.append(f"| {phase} | {done} | {remaining} |\n")
    lines.extend([
        "\n## Next pending steps\n",
        "| Step | Phase | Output |\n",
        "|---|---|---|\n",
    ])
    shown = 0
    for step in steps:
        if step.kind == "compute_gaps":
            continue
        if step.done():
            continue
        out = str(step.output_path.relative_to(REPO_ROOT)) if step.output_path else "-"
        lines.append(f"| `{step.label}` | {step.phase} | `{out}` |\n")
        shown += 1
        if shown >= 12:
            break
    STATUS_PATH.write_text("".join(lines))


def make_temp_config(step: Step) -> Path:
    TMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    with open(step.config_path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update(step.patches)
    tmp_path = TMP_CFG_DIR / f"{slugify(step.label)}.yaml"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_path


def build_command(step: Step, config_path: Path) -> list[str]:
    if step.kind == "intrinsic":
        cmd = [str(PYTHON_BIN), "-u", "-m", "src.fedgnn.experiments.run_intrinsic_eval", "--config", str(config_path)]
        for key in ["operator", "dataset", "beta", "seed"]:
            if key in step.cli_args:
                cmd.extend([f"--{key}", str(step.cli_args[key])])
        return cmd
    if step.kind == "downstream":
        cmd = [str(PYTHON_BIN), "-u", "-m", "src.fedgnn.experiments.run_downstream_eval", "--config", str(config_path)]
        for key in ["operator", "dataset", "backbone", "beta", "seed"]:
            if key in step.cli_args:
                cmd.extend([f"--{key}", str(step.cli_args[key])])
        return cmd
    if step.kind == "compute_gaps":
        return [str(PYTHON_BIN), "-u", "-m", "src.fedgnn.experiments.run_downstream_eval", "--config", str(config_path), "--compute-gaps", "--resume"]
    raise ValueError(step.kind)


def run_step(step: Step) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = make_temp_config(step) if step.patches else step.config_path
    cmd = ["nice", "-n", "15", *build_command(step, config_path)]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    log_path = step_log_path(step)
    with open(log_path, "a") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===\n")
        log.write("CMD: " + shlex.join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run remaining propagator_eval items sequentially when resources are safe.")
    parser.add_argument("--dry-run", action="store_true", help="Write the status file and print next steps without launching anything.")
    parser.add_argument("--wait-seconds", type=int, default=600, help="How long to sleep between resource checks when waiting.")
    parser.add_argument("--start-phase", choices=["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6"], default=None)
    args = parser.parse_args()

    steps = build_steps()
    if args.start_phase:
        phase_order = ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6"]
        allowed = set(phase_order[phase_order.index(args.start_phase):])
        steps = [s for s in steps if s.phase in allowed]

    snap = snapshot()
    write_status(steps, None, "dry-run" if args.dry_run else "idle", "runner initialized", snap)

    pending = [s for s in steps if not s.done() or s.kind == "compute_gaps"]
    if args.dry_run:
        print(f"Total steps considered: {len(steps)}")
        print(f"Pending runnable steps: {sum(1 for s in pending if s.kind != 'compute_gaps')}")
        for step in pending[:20]:
            print(step.label)
        print(f"Status file: {STATUS_PATH}")
        return 0

    for step in steps:
        if step.kind != "compute_gaps" and step.done():
            continue
        while True:
            snap = snapshot()
            safe, reasons = is_safe(step.policy, snap)
            details = "safe to run" if safe else ", ".join(reasons)
            write_status(steps, step, "waiting" if not safe else "running", details, snap)
            if safe:
                break
            time.sleep(args.wait_seconds)

        code = run_step(step)
        snap = snapshot()
        details = f"exit_code={code}"
        write_status(steps, step, "step-failed" if code else "step-complete", details, snap)
        if code != 0:
            print(f"Step failed: {step.label} (see {step_log_path(step)})")
            return code

    snap = snapshot()
    write_status(steps, None, "complete", "all remaining steps finished", snap)
    print("All remaining steps completed.")
    print(f"Status file: {STATUS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
