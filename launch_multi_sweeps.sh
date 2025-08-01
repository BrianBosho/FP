#!/bin/bash

echo "Starting multi-dataset WandB sweeps..."

# Launch Cora sweep
echo "Creating Cora sweep..."
CORA_SWEEP_ID=$(wandb sweep conf/sweep_cora.yaml 2>&1 | grep -o 'wandb: Run sweep agent with: wandb agent.*' | sed 's/wandb: Run sweep agent with: wandb agent //')
echo "Cora sweep ID: $CORA_SWEEP_ID"

# Launch Citeseer sweep  
echo "Creating Citeseer sweep..."
CITESEER_SWEEP_ID=$(wandb sweep conf/sweep_citeseer.yaml 2>&1 | grep -o 'wandb: Run sweep agent with: wandb agent.*' | sed 's/wandb: Run sweep agent with: wandb agent //')
echo "Citeseer sweep ID: $CITESEER_SWEEP_ID"

# Launch Pubmed sweep
echo "Creating Pubmed sweep..."
PUBMED_SWEEP_ID=$(wandb sweep conf/sweep_pubmed.yaml 2>&1 | grep -o 'wandb: Run sweep agent with: wandb agent.*' | sed 's/wandb: Run sweep agent with: wandb agent //')
echo "Pubmed sweep ID: $PUBMED_SWEEP_ID"

echo ""
echo "All sweeps created! Now launching agents in background..."
echo ""

# Start agents in background
nohup wandb agent $CORA_SWEEP_ID > logs/cora_sweep.log 2>&1 &
echo "Cora agent started in background (PID: $!)"

nohup wandb agent $CITESEER_SWEEP_ID > logs/citeseer_sweep.log 2>&1 &
echo "Citeseer agent started in background (PID: $!)"

nohup wandb agent $PUBMED_SWEEP_ID > logs/pubmed_sweep.log 2>&1 &
echo "Pubmed agent started in background (PID: $!)"

echo ""
echo "All sweep agents are running in background!"
echo "Check logs in logs/ directory for progress"
echo "Use 'ps aux | grep wandb' to see running processes"
echo "Use 'kill PID' to stop individual agents"