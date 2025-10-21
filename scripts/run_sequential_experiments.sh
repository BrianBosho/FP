#!/bin/bash

# Run multiple experiments sequentially
# Each experiment will use the full GPU and complete before the next starts

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/experiments_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "Starting sequential experiments at $(date)"
echo "Logs will be saved to ${LOG_DIR}"

# Array of config files to run
CONFIGS=(
    "conf/ogbn-arxiv_config.yaml"
    "conf/photos_config.yaml"
    "conf/cora_config.yaml"
    "conf/citeseer_config.yaml"
)

# Run each experiment sequentially
for config in "${CONFIGS[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting experiment: ${config}"
    echo "Time: $(date)"
    echo "========================================"
    
    # Extract config name for log file
    config_name=$(basename "${config}" .yaml)
    log_file="${LOG_DIR}/${config_name}.log"
    
    # Run the experiment and wait for completion
    python -m src.experiments.run_experiments --config "${config}" 2>&1 | tee "${log_file}"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ ${config} completed successfully"
    else
        echo "✗ ${config} failed with error code $?"
    fi
    
    # Optional: Add delay between experiments
    sleep 5
done

echo ""
echo "========================================"
echo "All experiments completed at $(date)"
echo "========================================"
