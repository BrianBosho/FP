#!/bin/bash

# Activate conda environment (replace 'your_env_name' with actual environment name)
# conda activate your_env_name

# Create logs directory if it doesn't exist
mkdir -p logs

# Run all experiments in background with output redirected to log files

nohup python -m src.experiments.run_experiments --config conf/ablation/cora_gcn_no_pe.yaml > logs/nohup_nope_cora_ablation_10k.log 2>&1 &
nohup python -m src.experiments.run_experiments --config conf/ablation/cora_gcn_pe.yaml > logs/nohup_pe_cora_ablation_10k.log 2>&1 &

# Wait a moment and show running processes
sleep 2
echo "Started 4 experiment processes. Process IDs:"
ps aux | grep "run_experiments" | grep -v grep

echo "Logs are being written to the logs/ directory"
echo "Use 'ps aux | grep run_experiments' to check if processes are still running"
echo "Use 'tail -f logs/[logfile]' to monitor progress"