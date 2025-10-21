# Quick Start: Running Multiple Experiments

## TL;DR - Just Run This

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Test single experiment first
./scripts/experiment_utils.sh test

# Run multiple experiments in parallel (RECOMMENDED)
./scripts/run_parallel_with_ports.sh

# Monitor progress
./scripts/experiment_utils.sh status
```

## Commands Cheat Sheet

```bash
# Run experiments (choose one)
./scripts/run_parallel_with_ports.sh     # Parallel (2-3 at once) - BEST
./scripts/run_sequential_experiments.sh   # One at a time - SAFEST

# Monitor
./scripts/experiment_utils.sh status      # Check what's running
./scripts/experiment_utils.sh monitor     # Live GPU view
watch -n 2 nvidia-smi                     # Alternative GPU monitor

# Manage
./scripts/experiment_utils.sh stop        # Stop everything
./scripts/experiment_utils.sh clean       # Clean up after stopping
./scripts/experiment_utils.sh results     # Show results summary

# Debug
./scripts/experiment_utils.sh logs        # List log files
tail -f logs/parallel_*/cora*.log         # Follow specific log
```

## Before You Start

1. **Check GPU is available:**
   ```bash
   nvidia-smi
   ```

2. **Make sure no other experiments are running:**
   ```bash
   ./scripts/experiment_utils.sh status
   ```

3. **If processes are stuck, clean up:**
   ```bash
   ./scripts/experiment_utils.sh stop
   ./scripts/experiment_utils.sh clean
   ```

## Customizing What Runs

Edit the experiment list in your chosen script:

```bash
# Edit parallel runner
nano scripts/run_parallel_with_ports.sh

# Find this section and modify:
EXPERIMENTS=(
    "conf/ogbn-arxiv_config.yaml"
    "conf/photos_config.yaml"
    # Add or remove configs here
)
```

## When Things Go Wrong

```bash
# Kill everything
pkill -f run_experiments
ray stop

# Check what's still running
ps aux | grep python

# Force clean GPU
./scripts/experiment_utils.sh clean

# Check GPU memory is freed
nvidia-smi
```

## Understanding Your GPU

Based on your nvidia-smi output:
- **Total GPU Memory:** 49,152 MiB (49 GB)
- **Per Experiment:** ~6-7 GB
- **Safe Parallel Limit:** 2-3 experiments

## Reading the Output

### Successful experiment log shows:
```
Starting experiment: dataset_model_beta_clients
Round 1/5: Testing...
Round 2/5: Testing...
...
✓ Experiment completed successfully
```

### Failed experiment shows:
```
Error: Out of memory
OR
Error: Address already in use (port conflict)
```

## Pro Tips

1. **Start Small:** Test with 1-2 experiments first
2. **Use Sequential for Overnight:** Let it run unattended safely
3. **Use Parallel for Speed:** When you can monitor
4. **Always Check Logs:** If something fails, the log tells you why
5. **Clean Between Major Runs:** `./scripts/experiment_utils.sh clean`

## Need More Help?

Read the full guide: `EXPERIMENT_RUNNER_GUIDE.md`
