#!/bin/sh
set -eu

cd /home/bosho/FP

LOG_DIR=/home/bosho/FP/experiments/propagator_eval/results/phase_1_cora_intrinsic/logs
RAW_DIR=/home/bosho/FP/experiments/propagator_eval/results/phase_1_cora_intrinsic/raw
mkdir -p "$LOG_DIR"

OPERATORS="adjacency asymmetric_random_walk diffusion chebyshev_diffusion appnp"
BETAS="10000 10 1"
SEEDS="0 1 2"

for op in $OPERATORS; do
  for beta in $BETAS; do
    for seed in $SEEDS; do
      out="$RAW_DIR/$op/cora/beta${beta}_seed${seed}.json"
      log="$LOG_DIR/phase1__cora__${op}__beta${beta}__seed${seed}.log"
      if [ -f "$out" ]; then
        echo "SKIP exists: $out"
        continue
      fi
      echo "RUN $op beta=$beta seed=$seed"
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      nice -n 15 /home/bosho/.conda/envs/fedgnn/bin/python -u -m src.fedgnn.experiments.run_intrinsic_eval \
        --config experiments/propagator_eval/configs/phase_1_cora_intrinsic.yaml \
        --operator "$op" --dataset Cora --beta "$beta" --seed "$seed" \
        > "$log" 2>&1 || echo "FAILED $op beta=$beta seed=$seed (see $log)"
    done
  done
done

echo "Phase 1 Cora queue complete"
