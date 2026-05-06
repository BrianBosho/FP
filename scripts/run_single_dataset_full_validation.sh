#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/bosho/.conda/envs/fedgnn/bin/python
CFG="${1:-experiments/configs/scalability/S1_single_full_validation.yaml}"
LOG_DIR="$REPO_ROOT/logs/scalability"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/single_full_validation_${TS}.out"

echo "$TS" > /tmp/single_full_validation_ts
ray stop --force >/dev/null 2>&1 || true
exec env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON" -u -m src.fedgnn.experiments.run_experiments --config "$CFG" > "$LOG" 2>&1
