#!/bin/bash

# Utility script for managing experiments
# Provides common operations for experiment management

case "$1" in
    stop)
        echo "Stopping all experiments and Ray processes..."
        pkill -f "run_experiments"
        ray stop
        echo "✓ All processes stopped"
        ;;
    
    status)
        echo "=== Experiment Status ==="
        echo ""
        echo "Running Python experiments:"
        ps aux | grep run_experiments | grep -v grep || echo "  None"
        echo ""
        echo "Ray processes:"
        ps aux | grep ray | grep -v grep | head -5 || echo "  None"
        echo ""
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
        echo ""
        echo "Ray ports in use:"
        netstat -tuln 2>/dev/null | grep -E "637[0-9]" || ss -tuln 2>/dev/null | grep -E "637[0-9]" || echo "  Cannot check ports (netstat/ss not available)"
        ;;
    
    logs)
        echo "=== Recent Experiment Logs ==="
        echo ""
        if [ -d "logs" ]; then
            echo "Latest log directories:"
            ls -lth logs/ | head -10
            echo ""
            if [ ! -z "$2" ]; then
                echo "Showing tail of $2:"
                tail -n 50 "$2"
            else
                echo "Usage: $0 logs <log_file>"
                echo "Example: $0 logs logs/parallel_ray_experiments_*/cora_port6379.log"
            fi
        else
            echo "No logs directory found"
        fi
        ;;
    
    clean)
        echo "Cleaning up experiment artifacts..."
        echo ""
        
        # Stop processes
        echo "1. Stopping processes..."
        pkill -f "run_experiments" 2>/dev/null
        ray stop 2>/dev/null
        
        # Clear CUDA cache (if Python available)
        echo "2. Clearing CUDA cache..."
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || echo "  (Python/PyTorch not available)"
        
        # Clean Ray temp files
        echo "3. Cleaning Ray temp files..."
        rm -rf /tmp/ray/* 2>/dev/null
        
        echo ""
        echo "✓ Cleanup complete"
        echo "  Run 'nvidia-smi' to verify GPU memory is freed"
        ;;
    
    monitor)
        echo "Starting GPU monitor (Ctrl+C to stop)..."
        echo ""
        watch -n 2 nvidia-smi
        ;;
    
    results)
        echo "=== Recent Results ==="
        echo ""
        if [ -d "results" ]; then
            echo "Results by dataset:"
            for dataset in results/*/; do
                if [ -d "$dataset" ]; then
                    count=$(find "$dataset" -name "*.csv" 2>/dev/null | wc -l)
                    echo "  $(basename $dataset): ${count} CSV files"
                fi
            done
            echo ""
            echo "Latest result files:"
            find results/ -name "*.csv" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -10 | cut -d' ' -f2-
        else
            echo "No results directory found"
        fi
        ;;
    
    test)
        echo "=== Running Quick Test ==="
        echo "This will run a single experiment to test your setup"
        echo ""
        
        # Check if config exists
        if [ ! -f "conf/cora_config.yaml" ]; then
            echo "✗ Error: conf/cora_config.yaml not found"
            exit 1
        fi
        
        # Run test
        echo "Running test experiment with Cora dataset..."
        python -m src.experiments.run_experiments --config conf/cora_config.yaml
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Test completed successfully!"
        else
            echo ""
            echo "✗ Test failed - check errors above"
        fi
        ;;
    
    help|--help|-h|"")
        cat << 'EOF'
Experiment Management Utility

Usage: ./scripts/experiment_utils.sh <command> [args]

Commands:
  stop      Stop all running experiments and Ray processes
  status    Show status of experiments, GPU, and Ray
  logs      List recent logs or show specific log file
            Usage: logs [log_file]
  clean     Clean up processes and temporary files
  monitor   Start real-time GPU monitoring (watch nvidia-smi)
  results   Show summary of experiment results
  test      Run a quick test experiment
  help      Show this help message

Examples:
  ./scripts/experiment_utils.sh status
  ./scripts/experiment_utils.sh stop
  ./scripts/experiment_utils.sh logs logs/experiment.log
  ./scripts/experiment_utils.sh clean
  ./scripts/experiment_utils.sh test

For running experiments, see:
  ./scripts/run_sequential_experiments.sh
  ./scripts/run_parallel_with_ports.sh

For detailed help, see: EXPERIMENT_RUNNER_GUIDE.md
EOF
        ;;
    
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
