#!/usr/bin/env bash
set -euo pipefail

# OGBN-Arxiv ablation sweep: find the best accuracy × speed config.
# 11 configs × 3 reps × 200 rounds ≈ 8–12 hours total.
# Each config runs sequentially to avoid GPU contention.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs/scalability/ablation"
mkdir -p "$LOG_DIR"
TS=$(date +"%Y%m%d_%H%M%S")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="/home/bosho/.conda/envs/fedgnn/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/bosho/.conda/envs/fedgnn/bin/python

BASE="experiments/configs/scalability/ablation/_base.yaml"
ABLATION_DIR="experiments/configs/scalability/ablation"

run_config() {
    local name="$1"
    local override="$2"
    local log="$LOG_DIR/${name}_${TS}.log"

    echo ""
    echo "========================================"
    echo " Config: $name"
    echo " Override: $override"
    echo " Log: $log"
    echo "========================================"

    ray stop --force >/dev/null 2>&1 || true

    # Build merged config: base + ablation override
    # OmegaConf.merge happens in load_config, but the runner takes --config as single file.
    # Instead, we create a temp merged YAML.
    TMP=$(mktemp /tmp/ablation_XXXXXX.yaml)
    "$PYTHON" -c "
from omegaconf import OmegaConf
base = OmegaConf.load('$BASE')
override = OmegaConf.load('$ABLATION_DIR/$override')
merged = OmegaConf.merge(base, override)
OmegaConf.save(merged, '$TMP')
"

    "$PYTHON" -m src.fedgnn.experiments.run_experiments --config "$TMP" 2>&1 | tee "$log"
    rm -f "$TMP"
    echo "[DONE] $name"
}

echo "========================================"
echo " OGBN-Arxiv Ablation Sweep"
echo " $(date)"
echo " 11 configs, 3 reps each, 200 rounds"
echo "========================================"

# --- A: Training HPs (adjacency, IID) ---
# These find the best lr/optimizer/wd/dropout/epochs
run_config "A01" "A01_adam_lr01_wd0_drop5.yaml"      # Adam default
run_config "A02" "A02_adam_lr01_wd0_drop3.yaml"      # less dropout
run_config "A03" "A03_adam_lr01_wd5e4_drop5.yaml"    # add weight decay
run_config "A04" "A04_adam_lr005_wd0_drop5.yaml"     # lower lr
run_config "A05" "A05_adam_lr005_wd5e4_drop5.yaml"   # lower lr + wd
run_config "A06" "A06_sgd_lr05_wd5e4_drop5.yaml"     # SGD
run_config "A07" "A07_adam_lr01_wd0_drop5_ep1.yaml"  # 1 local epoch (3x faster)
run_config "A08" "A08_adam_lr01_wd0_drop5_ep5.yaml"  # 5 local epochs

# --- B: FP iterations (adjacency) ---
# These find the minimum FP iterations needed
run_config "B01" "B01_fp10.yaml"   # 10 iters (fastest)
run_config "B02" "B02_fp25.yaml"   # 25 iters
# B03 = A01 (50 iters, already run above)

# --- C: Diffusion mode × FP iterations ---
# Does diffusion beat adjacency with fewer iterations?
run_config "C01" "C01_diff_fp10.yaml"   # diffusion 10 iters
run_config "C02" "C02_diff_fp25.yaml"   # diffusion 25 iters
run_config "C03" "C03_diff_fp50.yaml"   # diffusion 50 iters

echo ""
echo "========================================"
echo " Ablation complete. Reading results..."
echo "========================================"

# Print a summary table
"$PYTHON" -c "
import json, glob, os
results = []
for d in sorted(glob.glob('experiments/results/scalability/ogbn_arxiv_ablation/*/')):
    name = os.path.basename(d.rstrip('/'))
    jsons = sorted(glob.glob(os.path.join(d, '**/results_*.json'), recursive=True), key=os.path.getmtime, reverse=True)
    if not jsons:
        results.append((name, 'NO RESULT', '', ''))
        continue
    try:
        d = json.load(open(jsons[0]))
        s = d.get('summary', {})
        acc = s.get('average_client_result', float('nan'))
        std = s.get('std_client', float('nan'))
        n = len(d.get('rounds', []))
        results.append((name, f'{acc:.4f}', f'{std:.4f}', str(n)))
    except:
        results.append((name, 'ERROR', '', ''))

print(f'{'Config':<45} {'Accuracy':>10} {'Std':>8} {'Reps':>5}')
print('-' * 70)
for name, acc, std, n in results:
    print(f'{name:<45} {acc:>10} {std:>8} {n:>5}')
"

echo ""
echo "All done. Pick the config with best accuracy, then check speed."
