#!/bin/sh

# Wrapper script to run all dataset experiments sequentially in the background
# Each dataset starts 20 minutes after the previous one begins
# This script is designed to continue running even if disconnected from the VM

LOG_DIR="logs/experiment_runs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting all experiments at $(date)"
echo "Logs will be saved to $LOG_DIR"

# Run Cora experiments
echo "Starting Cora experiments at $(date)"
nohup ./scripts/run_cora_manual.sh > "$LOG_DIR/cora_${TIMESTAMP}.log" 2>&1 &
CORA_PID=$!
echo "Cora experiments started with PID: $CORA_PID"

# Wait 20 minutes before starting Pubmed
echo "Waiting 20 minutes before starting Pubmed experiments..."
sleep 1200  # 20 minutes = 1200 seconds

# Run Pubmed experiments
echo "Starting Pubmed experiments at $(date)"
nohup ./scripts/run_pubmed_manual.sh > "$LOG_DIR/pubmed_${TIMESTAMP}.log" 2>&1 &
PUBMED_PID=$!
echo "Pubmed experiments started with PID: $PUBMED_PID"

# Wait 20 minutes before starting Citeseer
echo "Waiting 20 minutes before starting Citeseer experiments..."
sleep 1200  # 20 minutes = 1200 seconds

# Run Citeseer experiments
echo "Starting Citeseer experiments at $(date)"
nohup ./scripts/run_citeseer_manual.sh > "$LOG_DIR/citeseer_${TIMESTAMP}.log" 2>&1 &
CITESEER_PID=$!
echo "Citeseer experiments started with PID: $CITESEER_PID"

echo "All experiments have been launched!"
echo "Process IDs:"
echo "  Cora:     $CORA_PID"
echo "  Pubmed:   $PUBMED_PID"
echo "  Citeseer: $CITESEER_PID"
echo ""
echo "Monitor logs with:"
echo "  tail -f $LOG_DIR/cora_${TIMESTAMP}.log"
echo "  tail -f $LOG_DIR/pubmed_${TIMESTAMP}.log"
echo "  tail -f $LOG_DIR/citeseer_${TIMESTAMP}.log"
echo ""
echo "Check running processes with: ps aux | grep python"
echo "Finished launching at $(date)"
