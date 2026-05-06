#!/usr/bin/env bash
# Baseline runs: zero_hop (lower bound) and full (oracle upper bound)
# Both IID (beta=10000) and non-IID (beta=1), up to 600 FL rounds with early stopping.
# Strictly sequential — one condition at a time.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Ensure Ray workers load CUDA-enabled torch from fedgnn env, not CPU-only miniconda
export LD_LIBRARY_PATH="/home/bosho/.conda/envs/fedgnn/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"

PYTHON=/home/bosho/.conda/envs/fedgnn/bin/python
CONFIG=experiments/configs/scalability/S1_ogbn_arxiv_iid_noniid.yaml
RESULTS=experiments/results/baseline/S1_ogbn_arxiv
LOG_DIR=experiments/results/baseline/S1_ogbn_arxiv/logs
MASTER_LOG="$RESULTS/baseline_s1_master.log"

mkdir -p "$LOG_DIR"

echo "=== Baseline master log start ===" | tee "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

run_baseline() {
    local beta=$1
    local loader=$2
    local label="beta${beta}_${loader}"
    local log="$LOG_DIR/${label}.log"

    echo "" | tee -a "$MASTER_LOG"
    echo "========================================================" | tee -a "$MASTER_LOG"
    echo "START: $label  ($(date '+%H:%M:%S'))" | tee -a "$MASTER_LOG"
    echo "========================================================" | tee -a "$MASTER_LOG"

    # Stop any lingering Ray from previous condition
    ray stop --force 2>/dev/null || true

    "$PYTHON" -m src.fedgnn.experiments.run_experiments \
        --config "$CONFIG" \
        --rounds 600 \
        --repetitions 1 \
        --beta "$beta" \
        --data_loading "$loader" \
        --results_dir "${RESULTS}/${label}" \
        2>&1 | tee "$log" | tee -a "$MASTER_LOG"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "DONE:  $label  ($(date '+%H:%M:%S'))  exit=0" | tee -a "$MASTER_LOG"
    else
        echo "FAIL:  $label  ($(date '+%H:%M:%S'))  exit=$exit_code" | tee -a "$MASTER_LOG"
    fi
    return $exit_code
}

# Oracle baseline (full real features) for IID and non-IID
run_baseline 10000 full
run_baseline 1     full

echo "" | tee -a "$MASTER_LOG"
echo "========================================================" | tee -a "$MASTER_LOG"
echo "All baseline runs complete  ($(date '+%H:%M:%S'))" | tee -a "$MASTER_LOG"
echo "========================================================" | tee -a "$MASTER_LOG"

# Results summary
python3 - <<'EOF'
import json, glob

files = sorted(glob.glob("experiments/results/baseline/S1_ogbn_arxiv/**/results_*.json", recursive=True))
if not files:
    print("No result files found.")
else:
    print(f"\n{'Condition':<35} {'Global':>8} {'Client':>8} {'Part(s)':>8} {'Train(s)':>9} {'FL rds':>7} {'Duration':>10}")
    print("-" * 90)
    for f in files:
        d = json.load(open(f))
        s = d["summary"]
        cfg = d["experiment_config"]
        label = f"{cfg['dataset']}/{cfg['data_loading_option']}/b{cfg['beta']}"
        rounds = d.get("rounds", [])
        if rounds:
            r0 = rounds[0]
            duration = d.get("duration", "?")
            print(f"{label:<35} {s['average_global_result']:>8.4f} {s['average_client_result']:>8.4f} "
                  f"{r0.get('partition_time_s', 0):>8.1f} {r0.get('training_time_s', 0):>9.1f} "
                  f"{r0.get('rounds_executed', '?'):>7} {str(duration):>10}")
        else:
            print(f"{label:<35} {'FAILED':>8}")
EOF
