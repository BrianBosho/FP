#!/bin/bash
# Ablation study: Decomposing the 81% vs 70% GAT accuracy gap on Cora
# 6 configs × 5 reps × ~3 min/rep ≈ 90 min on a single GPU
#
# Reads:
#   A - C = pure aggregation effect (mean vs fedavg), short training, SGD
#   A - B = pure training-duration effect under mean
#   C - D = training-duration effect under fedavg_weighted + SGD
#   E - F = training-duration effect under fedavg_weighted + Adam
#   D - F = optimizer interaction within fedavg_weighted

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="$ROOT_DIR/experiments/results/ablation_gap/logs"

mkdir -p "$LOG_DIR"

CONFIGS=(
    A_baseline_mean_sgd_short
    B_mean_sgd_long
    C_fedavg_sgd_short
    D_fedavg_sgd_long
    E_fedavg_adam_short
    F_fedavg_adam_long
)

echo "=== GAT Accuracy Gap Ablation ==="
echo "Configs: ${#CONFIGS[@]}"
echo "Results: $ROOT_DIR/experiments/results/ablation_gap/"
echo ""

for config in "${CONFIGS[@]}"; do
    YAML="$SCRIPT_DIR/${config}.yaml"
    LOG="$LOG_DIR/${config}.log"

    if [ ! -f "$YAML" ]; then
        echo "MISSING: $YAML"
        continue
    fi

    echo "--- Running $config ---"
    echo "    Config: $YAML"
    echo "    Log:    $LOG"

    cd "$ROOT_DIR"
    /home/bosho/.conda/envs/fedgnn/bin/python -m src.experiments.run_experiments --config "$YAML" 2>&1 | tee "$LOG"

    echo "--- Done: $config ---"
    echo ""
done

echo "=== Ablation complete ==="
echo "Check results in: $ROOT_DIR/experiments/results/ablation_gap/"
echo ""
echo "Expected decomposition:"
echo "  A - C  = aggregation effect (SGD, short training)"
echo "  A - B  = training duration effect (mean)"
echo "  C - D  = training duration effect (fedavg_weighted, SGD)"
echo "  E - F  = training duration effect (fedavg_weighted, Adam)"
echo "  D - F  = optimizer effect (fedavg_weighted)"
