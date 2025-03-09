#!/bin/bash

# This script helps run experiments with the correct path setup

# Change to the project root directory
cd "$(dirname "$0")"

# Add the project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the experiment script from the src directory with all arguments passed to this script
python -m src.run_experiments "$@" 