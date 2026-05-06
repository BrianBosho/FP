#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/scalability"
mkdir -p "$LOG_DIR"
TS=$(date +"%Y%m%d_%H%M%S")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="/home/bosho/.conda/envs/fedgnn/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduces fragmentation on large graphs
PYTHON=/home/bosho/.conda/envs/fedgnn/bin/python

run_phase() {
    local label="$1"
    local config="$2"
    local log="$LOG_DIR/${label}_${TS}.log"
    echo ""
    echo "========================================"
    echo " Phase: $label"
    echo " Config: $config"
    echo " Log: $log"
    echo " GPU: $CUDA_VISIBLE_DEVICES"
    echo "========================================"
    ray stop --force >/dev/null 2>&1 || true
    "$PYTHON" -m src.fedgnn.experiments.run_experiments --config "$config" 2>&1 | tee "$log"
    echo "[DONE] $label"
}

# Phase 1: smoke
run_phase "smoke"    "experiments/configs/scalability/S1_smoke.yaml"

# Phase 2: peak diagnostic
run_phase "peak"     "experiments/configs/scalability/S1_peak.yaml"

# Phase 4: full run (uncomment after Step 3 code changes are confirmed)
# run_phase "S1_full" "experiments/configs/scalability/S1_ogbn_arxiv_iid_noniid.yaml"

echo ""
echo "All phases complete. Results in experiments/results/scalability/"
