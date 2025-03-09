#!/bin/bash

# This script sets up the FP project structure and cleans up unnecessary files

echo "Setting up FP project structure..."

# Create necessary directories if they don't exist
mkdir -p src/dataprocessing src/result_processing scripts tests config docs notebooks results logs

# Ask if user wants to clean up duplicated files and directories
read -p "Do you want to clean up duplicated files and directories? (y/n) " cleanup_duplicates

if [[ $cleanup_duplicates == "y" || $cleanup_duplicates == "Y" ]]; then
    echo "Cleaning up duplicated files and directories..."
    if [ -f scripts/cleanup_duplicates.sh ]; then
        scripts/cleanup_duplicates.sh
    else
        echo "Warning: cleanup_duplicates.sh script not found."
    fi
fi

# Ask if user wants to organize results
read -p "Do you want to organize all result folders into results_archive? (y/n) " organize_results

if [[ $organize_results == "y" || $organize_results == "Y" ]]; then
    echo "Organizing results..."
    if [ -f scripts/organize_results.sh ]; then
        scripts/organize_results.sh
    else
        echo "Warning: organize_results.sh script not found."
    fi
fi

# Run cleanup script for nohup files
if [ -f scripts/cleanup.sh ]; then
    echo "Running cleanup script for nohup files..."
    scripts/cleanup.sh
fi

echo "Setup complete!"
echo "You can now run experiments using the scripts in the 'scripts' directory:"
echo "  - ./scripts/run_all.sh: Run all experiments"
echo "  - ./scripts/run_single.sh: Run a single experiment"
echo "  - ./scripts/organize_results.sh: Organize result folders"
echo "  - ./scripts/cleanup_duplicates.sh: Clean up duplicated files and directories" 