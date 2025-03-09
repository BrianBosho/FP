#!/bin/bash

# This script moves all result-related folders to a results_archive folder to reduce clutter

# Navigate to the project root
cd "$(dirname "$0")/.."

echo "Creating results_archive directory..."
mkdir -p results_archive

echo "Moving result folders to results_archive..."

# List of result folders to move
result_folders=(
  "results"
  "results_1_client"
  "old_results"
  "old1_results"
  "New_results"
  "Central_results"
  "propagation_results"
  "consolidated_results"
  "Monte carlo results"
)

# Move each folder if it exists
for folder in "${result_folders[@]}"; do
  if [ -d "$folder" ]; then
    echo "Moving $folder..."
    mv "$folder" "results_archive/"
  else
    echo "Folder $folder not found, skipping."
  fi
done

# Create a new clean results directory for future experiments
echo "Creating new clean results directory..."
mkdir -p results

# Move result CSV files to results_archive/csv_files
echo "Moving result CSV files..."
mkdir -p results_archive/csv_files
find . -maxdepth 1 -name "*result*.csv" -exec mv {} results_archive/csv_files/ \;
find . -maxdepth 1 -name "reduced_df.csv" -exec mv {} results_archive/csv_files/ \;
find . -maxdepth 1 -name "extracted_results.csv" -exec mv {} results_archive/csv_files/ \;
find . -maxdepth 1 -name "extracted_results.xlsx" -exec mv {} results_archive/csv_files/ \;

echo "Results organization complete!"
echo "All result folders and files have been moved to the results_archive directory."
echo "A new clean results directory has been created for future experiments." 