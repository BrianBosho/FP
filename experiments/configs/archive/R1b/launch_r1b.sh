#!/bin/bash
# Launch all 6 R1b configs in parallel

cd /home/bosho/FP
mkdir -p logs/r1b

# Clean up test configs first
rm -f experiments/configs/R1b/test_*.yaml

launch_experiment() {
    local config=$1
    local name=$(basename "$config" .yaml)
    echo "Launching $name..."
    PYTHONPATH=/home/bosho/FP /home/bosho/.conda/envs/fedgnn/bin/python \
        -m src.experiments.run_experiments \
        --config "$config" \
        > "logs/r1b/${name}.log" 2>&1 &
    echo "  PID: $!"
}

# Launch all 6 configs
launch_experiment "experiments/configs/R1b/R1b_cora_gat_pe.yaml"
launch_experiment "experiments/configs/R1b/R1b_cora_gat_nope.yaml"
launch_experiment "experiments/configs/R1b/R1b_citeseer_gat_pe.yaml"
launch_experiment "experiments/configs/R1b/R1b_citeseer_gat_nope.yaml"
launch_experiment "experiments/configs/R1b/R1b_pubmed_gat_pe.yaml"
launch_experiment "experiments/configs/R1b/R1b_pubmed_gat_nope.yaml"

echo ""
echo "All 6 R1b experiments launched in parallel!"
echo "Monitor progress: tail -f logs/r1b/*.log"
echo "Check GPU: watch -n 5 nvidia-smi"
