#!/bin/bash
# Launch Citeseer NoPE cleanly — 4 parallel jobs, one per data_loading type
# Cleans NaN result folders before launching.

set -e
cd /home/bosho/FP
mkdir -p logs/r1b

PYTHON="/home/bosho/.conda/envs/fedgnn/bin/python"
CONFIG="experiments/configs/R1b/R1b_citeseer_gat_nope.yaml"
RESULTS="experiments/results/R1b_citeseer_nope"

# --- Clean NaN folders ---
echo "Removing NaN result folders..."
rm -rf "${RESULTS}/Citeseer_zero_hop_GAT_beta10000_clients10_hop0_iter50_t0.1_alpha0.5"
rm -rf "${RESULTS}/Citeseer_zero_hop_GAT_beta10_clients10_hop0_iter50_t0.1_alpha0.5"
rm -rf "${RESULTS}/Citeseer_zero_hop_GAT_beta1_clients10_hop0_iter50_t0.1_alpha0.5"
rm -rf "${RESULTS}/Citeseer_adjacency_GAT_beta10000_clients10_hop0_iter50_t0.1_alpha0.5"
rm -rf "${RESULTS}/Citeseer_adjacency_GAT_beta10_clients10_hop0_iter50_t0.1_alpha0.5"
echo "Done."

# --- Launch 4 parallel jobs ---
launch() {
    local loading=$1
    echo "Launching Citeseer NoPE — data_loading=${loading}..."
    PYTHONPATH=/home/bosho/FP $PYTHON \
        -m src.experiments.run_experiments \
        --config "$CONFIG" \
        --data_loading "$loading" \
        > "logs/r1b/R1b_citeseer_nope_${loading}.log" 2>&1 &
    echo "  PID: $!"
}

launch zero_hop
launch adjacency
launch diffusion
launch full

echo ""
echo "4 Citeseer NoPE jobs running in parallel."
echo "Monitor: tail -f logs/r1b/R1b_citeseer_nope_*.log"
