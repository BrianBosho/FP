#!/usr/bin/env python3
"""Simple GAT ablation runner — runs each config once, monitors, reports."""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/home/bosho/FP")
STATE_F = BASE / "results/gat_ablations_state.json"
LOG_DIR = BASE / "results/gat_ablations/logs"
PYBIN = "/home/bosho/.conda/envs/fedgnn/bin/python"
PYMODULE = "src.experiments.run_experiments"

CONFIGS = [
    ("cora_nope_h2", "Cora GAT no-PE hop=2", "experiments/configs/R1b/R1b_cora_gat_nope_2hop.yaml"),
    ("cora_pe_h2",   "Cora GAT PE hop=2",    "experiments/configs/R1b/R1b_cora_gat_pe_2hop.yaml"),
    ("citeseer_nope_h2", "Citeseer GAT no-PE hop=2", "experiments/configs/R1b/R1b_citeseer_gat_nope_2hop.yaml"),
    ("citeseer_pe_h2",   "Citeseer GAT PE hop=2",    "experiments/configs/R1b/R1b_citeseer_gat_pe_2hop.yaml"),
    ("pubmed_nope_h2", "Pubmed GAT no-PE hop=2", "experiments/configs/R1b/R1b_pubmed_gat_nope_2hop.yaml"),
    ("pubmed_pe_h2",   "Pubmed GAT PE hop=2",    "experiments/configs/R1b/R1b_pubmed_gat_pe_2hop.yaml"),
]


def load_state():
    if STATE_F.exists():
        return json.loads(STATE_F.read_text())
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": {
            stage_id: {
                "status": "pending",
                "label": label,
                "config": str(BASE / cfg),
                "log": str(LOG_DIR / f"{stage_id}.log"),
                "result": None,
                "error": None,
                "attempts": 0,
            }
            for stage_id, label, cfg in CONFIGS
        },
        "current": None,
        "completed": [],
        "failed": [],
    }


def save_state(state):
    STATE_F.parent.mkdir(parents=True, exist_ok=True)
    STATE_F.write_text(json.dumps(state, indent=2))


def gpu_ok(threshold=15):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=5
        )
        return int(out.strip()) < threshold
    except:
        return False


def kill_stale():
    subprocess.run(
        "ps aux | grep -E 'run_experiments|Cora_diffusion|ray' | grep -v grep | awk '{print $2}' | xargs -r kill -9",
        shell=True, capture_output=True
    )
    time.sleep(3)


def run_stage(stage_id, info):
    log_path = info["log"]
    cfg_path = info["config"]

    print(f"\n[{'='*60}")
    print(f"[{stage_id}] Starting: {info['label']}")
    print(f"[{stage_id}] Config: {cfg_path}")

    kill_stale()

    for wait in range(30):
        if gpu_ok(threshold=15):
            break
        print(f"[{stage_id}] Waiting for GPU ({wait+1}/30)...")
        time.sleep(5)
    else:
        return {"accuracy": None, "error": "GPU timeout"}

    cmd = [PYBIN, "-m", PYMODULE, "--config", cfg_path]
    print(f"[{stage_id}] Running: {' '.join(cmd)}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    pid = proc.pid
    print(f"[{stage_id}] PID: {pid}")

    check_interval = 30
    max_runtime = 3600  # 60 min safety timeout per config
    start_time = time.time()

    while True:
        time.sleep(check_interval)
        elapsed = time.time() - start_time

        rc = proc.poll()
        if rc is not None:
            log_file.flush()
            log_file.close()
            print(f"[{stage_id}] Exited code {rc} after {elapsed:.0f}s")
            if rc == 0:
                return parse_result(log_path)
            else:
                return {"accuracy": None, "error": f"Exit code {rc}"}

        if elapsed > max_runtime:
            proc.kill()
            log_file.flush()
            log_file.close()
            return {"accuracy": None, "error": f"Timeout after {max_runtime}s"}

        print(f"[{stage_id}] elapsed={elapsed:.0f}s, running...")


def parse_result(log_path):
    import re
    try:
        content = Path(log_path).read_text()
    except Exception as e:
        return {"accuracy": None, "error": f"Can't read log: {e}"}

    m = re.search(r"Average Global Result[:\s]+([\d.]+)", content)
    if m:
        return {"accuracy": float(m.group(1))}

    # Fallback: search for "average global test results" pattern
    m = re.search(r"average global test results[:\s]+([\d.]+)", content, re.IGNORECASE)
    if m:
        return {"accuracy": float(m.group(1))}

    return {"accuracy": None, "error": "Couldn't parse accuracy from log"}


def main():
    state = load_state()
    print(f"GAT Ablations — {len(state['stages'])} stages total")
    print(f"Completed: {len(state.get('completed', []))}")
    print(f"Failed: {len(state.get('failed', []))}")

    pending = [s for s, i in state["stages"].items() if i["status"] == "pending"]
    print(f"Pending: {pending}\n")

    if not pending:
        print("All stages complete!")
        return

    for stage_id in pending:
        info = state["stages"][stage_id]
        state["current"] = stage_id
        info["status"] = "running"
        info["attempts"] += 1
        save_state(state)

        result = run_stage(stage_id, info)
        info["result"] = result.get("accuracy")
        info["error"] = result.get("error")

        if result.get("accuracy") is not None:
            info["status"] = "completed"
            state.setdefault("completed", []).append(stage_id)
            state["current"] = None
            print(f"[{stage_id}] DONE — accuracy: {info['result']:.4f}")
        else:
            info["status"] = "failed"
            state.setdefault("failed", []).append(stage_id)
            state["current"] = None
            print(f"[{stage_id}] FAILED: {info['error']}")

        save_state(state)
        time.sleep(5)

    print("\nAll GAT ablation stages complete!")


if __name__ == "__main__":
    main()
