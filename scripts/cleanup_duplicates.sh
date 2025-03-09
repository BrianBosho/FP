#!/bin/bash

# This script cleans up duplicated files and directories in the FP repository
# It removes duplicates while ensuring no data is lost

# Navigate to the project root
cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

echo "Current directory: $ROOT_DIR"
echo "Starting cleanup of duplicated files and directories..."

# 1. Handle conf vs config duplication
echo "Fixing conf vs config duplication..."
if [ -d "conf" ] && [ -d "config" ]; then
    # Check if they have the same files
    CONF_FILES=$(ls -1 conf | sort)
    CONFIG_FILES=$(ls -1 config | sort)
    
    if [ "$CONF_FILES" == "$CONFIG_FILES" ]; then
        echo "conf and config directories contain the same files. Removing conf..."
        rm -rf conf
    else
        echo "conf and config directories have different files. Merging..."
        cp -rn conf/* config/
        rm -rf conf
    fi
else
    if [ -d "conf" ] && [ ! -d "config" ]; then
        echo "Only conf directory exists. Renaming to config..."
        mv conf config
    elif [ ! -d "conf" ] && [ -d "config" ]; then
        echo "Only config directory exists. No action needed."
    fi
fi

# 2. Handle duplicated dataprocessing directories
echo "Fixing duplicated dataprocessing directories..."
if [ -d "dataprocessing" ] && [ -d "src/dataprocessing" ]; then
    echo "Found duplicated dataprocessing directories. Checking for unique files..."
    # Create a temporary directory to store merged files
    mkdir -p tmp_dataprocessing
    
    # Copy all files from both directories to the temporary directory
    cp -rn dataprocessing/* tmp_dataprocessing/
    cp -rn src/dataprocessing/* tmp_dataprocessing/
    
    # Replace both directories with the merged one
    rm -rf dataprocessing
    rm -rf src/dataprocessing
    mv tmp_dataprocessing src/dataprocessing
    
    echo "Merged dataprocessing directories into src/dataprocessing"
fi

# 3. Handle duplicated result_processing directories
echo "Fixing duplicated result_processing directories..."
if [ -d "result_processing" ] && [ -d "src/result_processing" ]; then
    echo "Found duplicated result_processing directories. Checking for unique files..."
    # Create a temporary directory to store merged files
    mkdir -p tmp_result_processing
    
    # Copy all files from both directories to the temporary directory
    cp -rn result_processing/* tmp_result_processing/
    cp -rn src/result_processing/* tmp_result_processing/
    
    # Replace both directories with the merged one
    rm -rf result_processing
    rm -rf src/result_processing
    mv tmp_result_processing src/result_processing
    
    echo "Merged result_processing directories into src/result_processing"
fi

# 4. Move all Python files from root to src if they exist in both places
echo "Checking for Python files duplicated in root and src..."
for py_file in *.py; do
    if [ -f "$py_file" ] && [ -f "src/$py_file" ]; then
        echo "Found duplicated file: $py_file"
        
        # Compare file sizes and modification times
        ROOT_SIZE=$(stat -c%s "$py_file")
        SRC_SIZE=$(stat -c%s "src/$py_file")
        ROOT_MTIME=$(stat -c%Y "$py_file")
        SRC_MTIME=$(stat -c%Y "src/$py_file")
        
        if [ "$ROOT_MTIME" -gt "$SRC_MTIME" ] || [ "$ROOT_SIZE" -gt "$SRC_SIZE" ]; then
            echo "Root file is newer or larger. Replacing src/$py_file..."
            cp -f "$py_file" "src/$py_file"
        fi
        
        # Remove the file from root after ensuring it's in src
        rm -f "$py_file"
        echo "Removed duplicated file from root: $py_file"
    elif [ -f "$py_file" ] && [ ! -f "src/$py_file" ] && [ "$py_file" != "__init__.py" ]; then
        # If the file exists only in root, move it to src
        echo "Moving $py_file to src directory..."
        mv "$py_file" "src/"
    fi
done

# 5. Move notebook files to notebooks directory
echo "Moving notebook files to notebooks directory..."
for ipynb_file in *.ipynb; do
    if [ -f "$ipynb_file" ]; then
        echo "Moving $ipynb_file to notebooks directory..."
        mv "$ipynb_file" "notebooks/"
    fi
done

# 6. Move test files to tests directory
echo "Moving test files to tests directory..."
for test_file in test_*.py debug_*.py; do
    if [ -f "$test_file" ]; then
        echo "Moving $test_file to tests directory..."
        mv "$test_file" "tests/"
    fi
done

# 7. Move run_experiments.py to scripts directory if not already there
echo "Checking run_experiments.py location..."
if [ -f "run_experiments.py" ] && [ ! -f "scripts/run_experiments.py" ]; then
    echo "Moving run_experiments.py to scripts directory..."
    mv "run_experiments.py" "scripts/"
elif [ -f "run_experiments.py" ] && [ -f "scripts/run_experiments.py" ]; then
    echo "Found duplicated run_experiments.py. Checking which one to keep..."
    
    # Compare file sizes and modification times
    ROOT_SIZE=$(stat -c%s "run_experiments.py")
    SCRIPT_SIZE=$(stat -c%s "scripts/run_experiments.py")
    ROOT_MTIME=$(stat -c%Y "run_experiments.py")
    SCRIPT_MTIME=$(stat -c%Y "scripts/run_experiments.py")
    
    if [ "$ROOT_MTIME" -gt "$SCRIPT_MTIME" ] || [ "$ROOT_SIZE" -gt "$SCRIPT_SIZE" ]; then
        echo "Root file is newer or larger. Replacing scripts/run_experiments.py..."
        cp -f "run_experiments.py" "scripts/run_experiments.py"
    fi
    
    # Remove the file from root
    rm -f "run_experiments.py"
    echo "Removed duplicated run_experiments.py from root"
fi

# 8. Move documentation files to docs directory
echo "Moving documentation files to docs directory..."
if [ -f "run_experiments_documentation.md" ] && [ ! -f "docs/run_experiments_documentation.md" ]; then
    echo "Moving run_experiments_documentation.md to docs directory..."
    mv "run_experiments_documentation.md" "docs/"
elif [ -f "run_experiments_documentation.md" ] && [ -f "docs/run_experiments_documentation.md" ]; then
    echo "Found duplicated documentation file. Removing from root..."
    rm -f "run_experiments_documentation.md"
fi

# 9. Clean up any empty directories
echo "Cleaning up any empty directories..."
find . -type d -empty -not -path "*/\.*" -not -path "./results" -delete

echo "Cleanup complete! The repository structure is now much cleaner."
echo "All duplicated files and directories have been handled." 