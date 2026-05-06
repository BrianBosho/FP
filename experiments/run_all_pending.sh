#!/bin/bash
# Sequential pipeline: R5a → R5b → R1/R1b missing runs
# Each step waits for the previous to complete.
# All logs go to logs/rerun/

set -euo pipefail
cd /home/bosho/FP

PYTHON="/home/bosho/.conda/envs/fedgnn/bin/python"
LOG_DIR="logs/rerun"
mkdir -p "$LOG_DIR"

START=$(date +%s)

# ── Step 1: R5a (zero_hop + adjacency) ──────────────────────────────────────
echo ""
echo "═══ Step 1/3: R5a — zero_hop + adjacency (clients 1→20) ═══"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
PYTHONPATH=/home/bosho/FP $PYTHON -m src.fedgnn.experiments.run_experiments \
  --config experiments/configs/R5/R5a_cora_adj_zero.yaml \
  2>&1 | tee "$LOG_DIR/R5_r5a_resume.log"
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"

# ── Step 2: R5b (diffusion + full) ──────────────────────────────────────────
echo ""
echo "═══ Step 2/3: R5b — diffusion + full (clients 20→1) ═══"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
PYTHONPATH=/home/bosho/FP $PYTHON -m src.fedgnn.experiments.run_experiments \
  --config experiments/configs/R5/R5b_cora_diff_full.yaml \
  2>&1 | tee "$LOG_DIR/R5_r5b_resume.log"
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"

# ── Step 3: R1 + R1b missing runs ───────────────────────────────────────────
echo ""
echo "═══ Step 3/3: R1/R1b missing experiments ═══"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
bash experiments/run_missing.sh
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"

# ── Summary ─────────────────────────────────────────────────────────────────
END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))
echo ""
echo "═══ All done. Total: ${ELAPSED} minutes ═══"
echo "Logs: $LOG_DIR/"
