#!/bin/bash
# Close the 5 remaining R1 GCN gaps (3 missing, 2 incomplete).
#
# Gaps targeted:
#   Cora       diffusion  β=1      MISSING  → 10 reps
#   Cora       diffusion  β=10     8/10     → fresh 10-rep file (seed=300)
#   Citeseer   diffusion  β=1      MISSING  → 10 reps
#   Citeseer   zero_hop   β=1      3/10     → fresh 10-rep file (seed=300)
#   Pubmed     zero_hop   β=1      MISSING  → 10 reps
#
# Total: 5 conditions × 10 reps = 50 new runs (sequential, 1 GPU).
# After all runs, consolidate_results.py and generate_tables.py are re-run.
#
# Usage:
#   bash experiments/run_r1_gaps.sh

set -euo pipefail
cd /home/bosho/FP

PYTHON=/home/bosho/.conda/envs/fedgnn/bin/python
RUNNER="src.fedgnn.experiments.run_experiments"
LOG_DIR="logs/rerun"
mkdir -p "$LOG_DIR"

CONFIGS=(
    "experiments/configs/rerun/R1_cora_gaps.yaml"      # Cora diffusion β=1,10 (2×10 reps)
    "experiments/configs/rerun/R1_citeseer_gaps.yaml"  # Citeseer diffusion+zero_hop β=1 (2×10 reps)
    "experiments/configs/rerun/R1_pubmed_gaps.yaml"    # Pubmed zero_hop β=1 (1×10 reps)
)

run_config() {
    local config="$1"
    local name
    name=$(basename "$config" .yaml)
    local logfile="$LOG_DIR/${name}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  START: $name"
    echo "  Config: $config"
    echo "  Log:    $logfile"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    PYTHONPATH=/home/bosho/FP \
        $PYTHON -m $RUNNER --config "$config" \
        2>&1 | tee "$logfile"

    echo "  DONE: $name ($(date '+%Y-%m-%d %H:%M:%S'))"
}

START=$(date +%s)
echo "Starting R1 gap-fill at $(date '+%Y-%m-%d %H:%M:%S')"
echo "5 conditions × 10 reps = 50 runs"

for cfg in "${CONFIGS[@]}"; do
    run_config "$cfg"
done

echo ""
echo "═══ Consolidating results ═══"
PYTHONPATH=/home/bosho/FP $PYTHON experiments/consolidate_results.py --verbose

echo ""
echo "═══ Generating tables ═══"
PYTHONPATH=/home/bosho/FP $PYTHON experiments/generate_tables.py

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))
echo ""
echo "All done. Total elapsed: ${ELAPSED} minutes."
echo "Check completeness: experiments/output/completeness_R1.txt"
echo "Check tables:       experiments/output/tables/"
echo "Check logs:         $LOG_DIR/"
