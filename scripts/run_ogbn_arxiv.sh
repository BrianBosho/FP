#!/bin/bash
# Run ogbn-arxiv experiments sequentially with live progress tracking.
# Config 1: adjacency + diffusion  → experiments/results/ogbn-arxiv/adj_diff
# Config 2: zero_hop + full        → experiments/results/ogbn-arxiv/zero_full

set -e

PYTHON="/home/bosho/.conda/envs/fedgnn/bin/python"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="experiments/output/ogbn_arxiv_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

CONFIGS=(
    "conf/ogbn-arxiv_config.yaml"
    "conf/ogbn-arxiv_zero_full.yaml"
)

RESULT_DIRS=(
    "experiments/results/ogbn-arxiv/adj_diff"
    "experiments/results/ogbn-arxiv/zero_full"
)

# Expected: 2 strategies x 2 betas x 10 reps each config
EXPECTED_PER_CONFIG=40

progress() {
    local result_dir="$1"
    local done
    done=$(find "${result_dir}" -name "results_*.json" 2>/dev/null | wc -l)
    local combos
    combos=$(find "${result_dir}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  Completed runs : ${done} / ${EXPECTED_PER_CONFIG}"
    echo "  Combos started : ${combos} / 4  (strategy x beta)"
    if [ "${done}" -gt 0 ]; then
        echo "  Latest result  : $(find "${result_dir}" -name "results_*.json" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- | xargs basename 2>/dev/null)"
    fi
}

echo "========================================"
echo "ogbn-arxiv runner — started $(date)"
echo "Logs: ${LOG_DIR}"
echo "========================================"

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    result_dir="${RESULT_DIRS[$i]}"
    name=$(basename "${config}" .yaml)
    log="${LOG_DIR}/${name}.log"

    echo ""
    echo "----------------------------------------"
    echo "Config  : ${config}"
    echo "Results : ${result_dir}"
    echo "Log     : ${log}"
    echo "Started : $(date)"
    echo "----------------------------------------"

    "${PYTHON}" -u -m src.fedgnn.experiments.run_experiments \
        --config "${config}" 2>&1 | tee "${log}"

    status=${PIPESTATUS[0]}

    echo ""
    echo "Progress for ${name}:"
    progress "${result_dir}"

    if [ "${status}" -eq 0 ]; then
        echo "PASSED: ${name} at $(date)"
    else
        echo "FAILED: ${name} — see ${log}"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "All done at $(date)"
echo ""
echo "Final progress:"
for i in "${!CONFIGS[@]}"; do
    echo "  ${CONFIGS[$i]}:"
    progress "${RESULT_DIRS[$i]}"
done
echo "========================================"
