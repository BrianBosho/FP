#!/bin/bash

# Run the 4 Cora experiments sequentially: GCN/GAT × beta=1/10000, PE=True

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/cora_manual_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Manual Cora Experiments Runner"
echo "========================================"
echo "Started at: $(date)"
echo "Logs: ${LOG_DIR}"
echo ""

CONFIGS=(
    "conf/cora/cora_gcn_beta1.yaml"
    "conf/cora/cora_gcn_beta10000.yaml"
    "conf/cora/cora_gat_beta1.yaml"
    "conf/cora/cora_gat_beta10000.yaml"
)

for config in "${CONFIGS[@]}"; do
    config_name=$(basename "${config}" .yaml)
    log_file="${LOG_DIR}/${config_name}.log"
    
    echo "[$(date '+%H:%M:%S')] Starting: ${config_name}"
    echo "  → Config: ${config}"
    echo "  → Log: ${log_file}"
    
    python -m src.experiments.run_experiments --config "${config}" > "${log_file}" 2>&1
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ Completed: ${config_name}"
    else
        echo "[$(date '+%H:%M:%S')] ✗ Failed: ${config_name} (Exit: ${exit_code})"
    fi
    echo ""
done

echo "========================================"
echo "All Experiments Complete"
echo "========================================"
echo "Finished at: $(date)"
echo "Logs: ${LOG_DIR}"
echo "========================================"