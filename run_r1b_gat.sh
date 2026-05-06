#!/bin/bash
# Run all R1b GAT ablation configs sequentially
set -e

ROOT=/home/bosho/FP
PYTHON=/home/bosho/.conda/envs/fedgnn/bin/python
LOGDIR=$ROOT/experiments/results/R1b/logs
mkdir -p $LOGDIR

CONFIGS=(
  "experiments/configs/R1b/R1b_cora_gat_nope.yaml"
  "experiments/configs/R1b/R1b_cora_gat_nope_2hop.yaml"
  "experiments/configs/R1b/R1b_cora_gat_pe.yaml"
  "experiments/configs/R1b/R1b_citeseer_gat_nope.yaml"
  "experiments/configs/R1b/R1b_citeseer_gat_nope_2hop.yaml"
  "experiments/configs/R1b/R1b_citeseer_gat_pe.yaml"
  "experiments/configs/R1b/R1b_pubmed_gat_nope.yaml"
  "experiments/configs/R1b/R1b_pubmed_gat_nope_2hop.yaml"
)

echo "=== R1b GAT Ablations ==="
echo "Configs: ${#CONFIGS[@]}"
echo ""

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  log="$LOGDIR/${name}.log"
  
  echo "--- Running $name ---"
  echo "    Config: $cfg"
  echo "    Log:    $log"
  
  cd "$ROOT"
  $PYTHON -m src.experiments.run_experiments --config "$cfg" 2>&1 | tee "$log"
  
  echo "--- Done: $name ---"
  echo ""
done

echo "=== All R1b GAT configs complete ==="
