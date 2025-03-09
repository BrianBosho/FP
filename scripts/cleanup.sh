#!/bin/bash

# This script removes unnecessary files from the FP project

# Remove nohup.out files
echo "Removing nohup.out files..."
find .. -name "nohup*.out" -type f -delete
find .. -name "old_nohup.out" -type f -delete

echo "Cleanup complete!" 