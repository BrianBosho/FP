#!/bin/bash

# This script runs all experiments with the new directory structure

# Change to the scripts directory
cd "$(dirname "$0")"

# Run all experiments
python run_experiments.py --mode all --output_dir ../results

echo "All experiments completed!" 