#!/bin/bash
# Run all missing/incomplete R1 and R1b experiments sequentially.
#
# Usage:
#   bash experiments/run_missing.sh           # run all 8 configs
#   bash experiments/run_missing.sh --r1-only # only R1 GCN configs
#   bash experiments/run_missing.sh --r1b-only # only R1b GAT configs
#
# All jobs run sequentially (1 at a time) due to GPU/RAM constraints.
# Logs: logs/rerun/<config_name>.log
# After all runs complete, consolidate_results.py is re-run automatically.

set -euo pipefail

cd /home/bosho/FP

PYTHON=/home/bosho/.conda/envs/fedgnn/bin/python
RUNNER="src.experiments.run_experiments"
LOG_DIR="logs/rerun"
mkdir -p "$LOG_DIR"

# ── Config lists ────────────────────────────────────────────────────────────

R1_CONFIGS=(
    # Missing (0 reps) — run first, highest priority
    "experiments/configs/rerun/R1_cora_missing.yaml"        # Cora full+diff β=10 (2×10 reps)
    "experiments/configs/rerun/R1_citeseer_missing.yaml"    # Citeseer diff β=10 (1×10 reps)
    "experiments/configs/rerun/R1_pubmed_zerohop_missing.yaml"   # Pubmed zero_hop β=10 (1×10 reps)
    "experiments/configs/rerun/R1_pubmed_diffusion_missing.yaml" # Pubmed diff β=10,10000 (2×10 reps)
    # Top-up (incomplete) — run after missing
    "experiments/configs/rerun/R1_cora_topup.yaml"          # Cora full β=10000 (+5 reps)
    "experiments/configs/rerun/R1_citeseer_topup.yaml"      # Citeseer full β=10000,10 (+10 reps each)
)

R1B_CONFIGS=(
    "experiments/configs/rerun/R1b_citeseer_zerohop_topup.yaml"   # Citeseer GAT zero_hop β=10000 (+5)
    "experiments/configs/rerun/R1b_citeseer_diffusion_topup.yaml" # Citeseer GAT diff β=10 (+5)
)

# ── Argument parsing ─────────────────────────────────────────────────────────

RUN_R1=true
RUN_R1B=true
if [[ "${1:-}" == "--r1-only" ]]; then
    RUN_R1B=false
elif [[ "${1:-}" == "--r1b-only" ]]; then
    RUN_R1=false
fi

# ── Helper: run one config ───────────────────────────────────────────────────

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

# ── Main ─────────────────────────────────────────────────────────────────────

START=$(date +%s)
echo "Starting rerun at $(date '+%Y-%m-%d %H:%M:%S')"
echo "R1 GCN: $RUN_R1  |  R1b GAT: $RUN_R1B"

if $RUN_R1; then
    echo ""
    echo "═══ R1 GCN MISSING + TOP-UP (6 configs) ═══"
    for cfg in "${R1_CONFIGS[@]}"; do
        run_config "$cfg"
    done
fi

if $RUN_R1B; then
    echo ""
    echo "═══ R1b GAT TOP-UP (2 configs) ═══"
    for cfg in "${R1B_CONFIGS[@]}"; do
        run_config "$cfg"
    done
fi

# ── Re-consolidate ───────────────────────────────────────────────────────────

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
echo "Check tables: experiments/output/tables/"
echo "Check logs:   $LOG_DIR/"
