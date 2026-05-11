#!/bin/bash
# Resume incomplete R1b Pubmed GAT experiments
# NoPE: full_beta1 missing (1 config)
# PE-adj: beta10, beta1 missing (2 configs)
# PE-diff: beta10, beta1 missing (2 configs)
set -e

ROOT=/home/bosho/FP
PYTHON=/home/bosho/.conda/envs/fedgnn/bin/python
LOGDIR=$ROOT/logs/r1b

cd "$ROOT"

echo "=== [1/3] NoPE resume (full_beta1) ==="
$PYTHON -m src.fedgnn.experiments.run_experiments \
  --config experiments/configs/R1b/R1b_pubmed_gat_nope.yaml \
  2>&1 | tee "$LOGDIR/R1b_pubmed_gat_nope_resume.log"
echo "--- NoPE done ---"

echo "=== [2/3] PE-adj resume (beta10, beta1) ==="
$PYTHON -m src.fedgnn.experiments.run_experiments \
  --config experiments/configs/R1b/R1b_pubmed_gat_pe_adj.yaml \
  2>&1 | tee "$LOGDIR/R1b_pubmed_gat_pe_adj_resume.log"
echo "--- PE-adj done ---"

echo "=== [3/3] PE-diff resume (beta10, beta1) ==="
$PYTHON -m src.fedgnn.experiments.run_experiments \
  --config experiments/configs/R1b/R1b_pubmed_gat_pe_diff.yaml \
  2>&1 | tee "$LOGDIR/R1b_pubmed_gat_pe_diff_resume.log"
echo "--- PE-diff done ---"

echo "=== All resume runs complete ==="
