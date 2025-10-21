# Running Multiple Experiments Guide

This guide explains how to run multiple experiments without crashes or conflicts.

## Problem Overview

When running multiple federated learning experiments in parallel, you may encounter:
- **GPU memory conflicts**: Multiple processes trying to use the same GPU memory
- **Ray port conflicts**: Ray framework uses port 6379 by default, causing conflicts
- **Process interference**: Multiple experiments modifying shared resources

## Solutions

We provide 4 different approaches, choose based on your needs:

---

## Solution 1: Sequential Execution (Safest, Simplest)

**Best for**: Running experiments overnight without worrying about conflicts.

```bash
# Make executable
chmod +x scripts/run_sequential_experiments.sh

# Run all experiments one after another
./scripts/run_sequential_experiments.sh
```

**Pros:**
- No conflicts possible
- Simple to understand
- Uses full GPU per experiment
- Easy to debug

**Cons:**
- Slower (no parallelism)
- GPU may be underutilized

**How it works:**
Runs each experiment to completion before starting the next one.

---

## Solution 2: Parallel with Custom Ray Ports (Recommended)

**Best for**: Running 2-3 experiments in parallel on a single GPU with 49GB memory.

```bash
# Make executable
chmod +x scripts/run_parallel_with_ports.sh

# Edit MAX_PARALLEL in the script if needed (default: 2)
nano scripts/run_parallel_with_ports.sh  # Change MAX_PARALLEL=2 to desired value

# Run experiments in parallel
./scripts/run_parallel_with_ports.sh
```

**Pros:**
- Faster than sequential (2-3x speedup)
- Safe (unique Ray ports prevent conflicts)
- Automatic queueing
- Good GPU utilization

**Cons:**
- Each experiment gets less GPU memory
- Requires code modification (already done)

**How it works:**
- Assigns unique Ray port to each experiment (6379, 6380, 6381, etc.)
- Runs MAX_PARALLEL experiments simultaneously
- Automatically queues remaining experiments

**GPU Memory Estimation:**
Based on your nvidia-smi output:
- Each experiment uses ~570-635 MiB per client + ~500 MiB base
- With 10 clients: ~6-6.5 GB per experiment
- Your 49GB GPU can handle ~7 experiments simultaneously
- **Recommended MAX_PARALLEL: 2-3** (to leave headroom)

---

## Solution 3: Queue-based Execution

**Best for**: Running many experiments with controlled parallelism.

```bash
# Make executable
chmod +x scripts/run_queued_experiments.sh

# Edit MAX_PARALLEL if needed
nano scripts/run_queued_experiments.sh

# Run with queue management
./scripts/run_queued_experiments.sh
```

**Pros:**
- Handles large experiment lists
- Automatic job management
- Simple parallelism control

**Cons:**
- No custom Ray ports (may have conflicts)
- Less sophisticated than Solution 2

---

## Solution 4: Manual Parallel Execution

**Best for**: Quick parallel runs of 2-3 specific experiments.

```bash
# Terminal 1
python -m src.experiments.run_experiments \
    --config conf/ogbn-arxiv_config.yaml \
    --ray_port 6379 \
    > logs/arxiv.log 2>&1 &

# Terminal 2
python -m src.experiments.run_experiments \
    --config conf/photos_config.yaml \
    --ray_port 6380 \
    > logs/photos.log 2>&1 &

# Check status
jobs
tail -f logs/arxiv.log
```

**Pros:**
- Maximum control
- Easy to start/stop individual experiments
- Good for testing

**Cons:**
- Manual management
- Error-prone for many experiments

---

## Monitoring Running Experiments

### Check GPU Usage
```bash
# Watch GPU usage in real-time
watch -n 2 nvidia-smi

# Or one-time check
nvidia-smi
```

### Check Experiment Progress
```bash
# View all experiment logs
ls -lh logs/

# Follow a specific experiment
tail -f logs/parallel_ray_experiments_*/cora_port6379.log

# Check for errors
grep -i "error\|failed\|exception" logs/parallel_ray_experiments_*/*.log
```

### Check Running Processes
```bash
# List all Python experiment processes
ps aux | grep run_experiments

# Count running experiments
ps aux | grep run_experiments | wc -l

# Kill all experiment processes (if needed)
pkill -f run_experiments
```

### Monitor Ray Processes
```bash
# Check Ray processes
ps aux | grep ray

# Check if Ray ports are in use
netstat -tuln | grep 637

# Kill Ray if stuck
ray stop
```

---

## Troubleshooting

### Issue: "Address already in use" error
**Cause:** Ray trying to use same port  
**Solution:** Use scripts with custom Ray ports (Solution 2)

```bash
# Or manually kill Ray
ray stop
pkill -9 -f ray
```

### Issue: Out of GPU Memory
**Cause:** Too many experiments running in parallel  
**Solution:** Reduce MAX_PARALLEL in scripts

```bash
# Edit script
nano scripts/run_parallel_with_ports.sh
# Change: MAX_PARALLEL=2 to MAX_PARALLEL=1
```

### Issue: Experiments hanging/frozen
**Cause:** Ray or GPU process stuck  
**Solution:** Kill and restart

```bash
# Kill all Python experiments
pkill -f run_experiments

# Stop Ray completely
ray stop

# Clean GPU memory
nvidia-smi --gpu-reset  # Use with caution!

# Or reboot if nothing else works
```

### Issue: Results not saving
**Cause:** Permission or path issues  
**Solution:** Check results directory

```bash
# Check if results directory exists and is writable
ls -ld results/
mkdir -p results/

# Check disk space
df -h
```

---

## Best Practices

1. **Start with Sequential:** Test your experiments work individually first
2. **Monitor Initially:** Watch nvidia-smi and logs when starting parallel runs
3. **Leave Headroom:** Don't use 100% of GPU memory (use MAX_PARALLEL=2-3, not 7)
4. **Use Logs:** Always redirect output to log files for debugging
5. **Clean Between Runs:** Run `ray stop` between major experiment batches
6. **Backup Results:** Copy results directory before large experiment runs

---

## Configuration Tips

### Reducing Memory Per Experiment

Edit your config files to use less memory:

```yaml
# In conf/your_config.yaml

# Reduce concurrent clients (default: 5)
max_concurrent_clients: 2

# Use fewer clients
num_clients: [5]  # instead of [10]

# Reduce batch size if applicable
batch_size: 32  # instead of 64
```

### Optimizing for Parallel Runs

```yaml
# Disable wandb for parallel runs (reduces overhead)
use_wandb: false

# Reduce repetitions for testing
repetitions: 1

# Use fewer rounds for quick tests
num_rounds: 3
```

---

## Example Workflows

### Workflow 1: Quick Test
```bash
# Test with 2 experiments in parallel
./scripts/run_parallel_with_ports.sh
```

### Workflow 2: Overnight Run
```bash
# Run all experiments sequentially
nohup ./scripts/run_sequential_experiments.sh > overnight.log 2>&1 &

# Check progress next morning
tail overnight.log
ls -lh results/
```

### Workflow 3: Multiple Configs, Same Dataset
```bash
# Edit script to only run specific configs
nano scripts/run_parallel_with_ports.sh
# Modify EXPERIMENTS array to your needs

./scripts/run_parallel_with_ports.sh
```

---

## Summary Table

| Solution | Speed | Safety | Complexity | Best For |
|----------|-------|--------|------------|----------|
| Sequential | ★☆☆ | ★★★ | ★☆☆ | Overnight runs |
| Parallel (ports) | ★★★ | ★★★ | ★★☆ | **Most users** |
| Queue-based | ★★☆ | ★★☆ | ★★☆ | Many experiments |
| Manual | ★★☆ | ★☆☆ | ★★★ | Testing |

**Recommendation:** Start with **Sequential** for testing, then use **Parallel (ports)** for production runs.

---

## Getting Help

If you encounter issues:
1. Check the log files in `logs/` directory
2. Run `nvidia-smi` to check GPU status
3. Run `ps aux | grep ray` to check for stuck processes
4. Try `ray stop` and restart your experiment

For persistent issues, share:
- Log files from `logs/` directory
- Output of `nvidia-smi`
- Your config file
- Error messages
