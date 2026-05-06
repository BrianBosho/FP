#!/usr/bin/env bash
# Sequential smoke: S1 ogbn-arxiv, beta {10000,1} x {adjacency,diffusion}
# 5 FL rounds, 1 rep, 3 local epochs (from config), one run at a time.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/home/bosho/.conda/envs/fedgnn/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"

PYTHON=/home/bosho/.conda/envs/fedgnn/bin/python
CONFIG=experiments/configs/scalability/S1_ogbn_arxiv_iid_noniid.yaml
RESULTS=experiments/results/smoke/S1_ogbn_arxiv
LOG_DIR=experiments/results/smoke/S1_ogbn_arxiv/logs

mkdir -p "$LOG_DIR"

run_smoke() {
    local beta=$1
    local loader=$2
    local label="beta${beta}_${loader}"
    local log="$LOG_DIR/${label}.log"

    echo ""
    echo "========================================================"
    echo "START: $label  ($(date '+%H:%M:%S'))"
    echo "========================================================"

    "$PYTHON" -m src.fedgnn.experiments.run_experiments \
        --config "$CONFIG" \
        --rounds 5 \
        --repetitions 1 \
        --beta "$beta" \
        --data_loading "$loader" \
        --results_dir "${RESULTS}/${label}" \
        2>&1 | tee "$log"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "DONE:  $label  ($(date '+%H:%M:%S'))  exit=0"
    else
        echo "FAIL:  $label  ($(date '+%H:%M:%S'))  exit=$exit_code"
    fi
    return $exit_code
}

# Run all 4 conditions sequentially
run_smoke 10000 adjacency
run_smoke 10000 diffusion
run_smoke 1     adjacency
run_smoke 1     diffusion

echo ""
echo "========================================================"
echo "All smoke runs complete  ($(date '+%H:%M:%S'))"
echo "========================================================"

# Quick results summary
python3 - <<'EOF'
import json, glob, os

files = sorted(glob.glob("experiments/results/smoke/S1_ogbn_arxiv/**/results_*.json", recursive=True))
if not files:
    print("No result files found yet.")
else:
    print(f"\n{'Condition':<30} {'Global':>8} {'Client':>8} {'Part(s)':>8} {'Train(s)':>9} {'FL rds':>7}")
    print("-" * 75)
    for f in files:
        d = json.load(open(f))
        s = d["summary"]
        cfg = d["experiment_config"]
        label = f"{cfg['dataset']}/{cfg['data_loading_option']}/b{cfg['beta']}"
        rounds = d.get("rounds", [])
        if rounds:
            r0 = rounds[0]
            print(f"{label:<30} {s['average_global_result']:>8.4f} {s['average_client_result']:>8.4f} "
                  f"{r0.get('partition_time_s', 0):>8.1f} {r0.get('training_time_s', 0):>9.1f} "
                  f"{r0.get('rounds_executed', '?'):>7}")
        else:
            print(f"{label:<30} {'FAILED':>8}")
EOF
