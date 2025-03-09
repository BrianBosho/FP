#!/bin/bash

# This script runs a single experiment with the new directory structure

# Change to the scripts directory
cd "$(dirname "$0")"

# Default values
DATASET="Cora"
MODEL="GCN"
DATA_LOADING="zero_hop"
NUM_CLIENTS=""
HOP=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --data_loading)
      DATA_LOADING="$2"
      shift 2
      ;;
    --num_clients)
      NUM_CLIENTS="--num_clients $2"
      shift 2
      ;;
    --hop)
      HOP="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Run the experiment
python run_experiments.py --mode single --dataset "$DATASET" --model "$MODEL" --data_loading "$DATA_LOADING" $NUM_CLIENTS --hop "$HOP" --output_dir ../results

echo "Experiment completed!" 