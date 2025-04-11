# Technical Commands and Shortcuts

This document contains useful commands and shortcuts for this project.

## Bash and Environment Commands

### Bash Setup
```bash
# Source your bash configuration
source ~/.bashrc

# If you need to edit your bashrc
nano ~/.bashrc
```

### Conda Commands
```bash
# Activate conda base environment
conda activate

# Activate a specific conda environment
conda activate [environment_name]

# List all conda environments
conda env list

# Create a new conda environment
conda create -n [environment_name] python=[version]

# Install packages in current environment
conda install [package_name]

# Update conda
conda update conda
```

## Git Commands

```bash
# Check git status
git status

# Add files to staging
git add [file_name]
git add .  # add all files

# Commit changes
git commit -m "Your commit message"

# Push to remote repository
git push origin [branch_name]

# Pull latest changes
git pull origin [branch_name]

# Create and switch to a new branch
git checkout -b [new_branch_name]

# Switch branches
git checkout [branch_name]
```

## Project-Specific Commands

```bash
# Run experiments
python run_experiments.py

# Run the server
python server.py

# Run the client
python client.py
```

## Useful Shortcuts

### Terminal Shortcuts
- `Ctrl+C`: Interrupt/kill the current command
- `Ctrl+Z`: Suspend the current process
- `Ctrl+D`: Exit the current shell
- `Ctrl+R`: Search command history
- `Ctrl+L`: Clear the terminal screen

### File Navigation
```bash
# List files and directories
ls -la

# Change directory
cd [directory_path]

# Go up one directory
cd ..

# Go to home directory
cd ~

# Print working directory
pwd
```

Feel free to add more commands and shortcuts as needed for your workflow.