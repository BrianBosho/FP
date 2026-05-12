# R1b Re-Run Instructions — Fixing Short Repetitions

**Date:** 2026-04-29
**Issue:** Several completed experiments have fewer than the required 10 repetitions due to OOM crashes and process kills during the original run. These need to be re-run with only the missing reps, then merged into the existing results.

---

## Current State — What's Short

### Cora NoPE

| Combo | Got | Need | Config | Results Dir |
|-------|-----|------|--------|-------------|
| zero_hop beta10000 | 4 | +6 | `R1b_cora_gat_nope.yaml` | `experiments/results/R1b_cora_nope/` |
| diffusion beta10000 | 9 | +1 | `R1b_cora_gat_nope.yaml` | `experiments/results/R1b_cora_nope/` |

### Citeseer NoPE

| Combo | Got | Need | Config | Results Dir |
|-------|-----|------|--------|-------------|
| zero_hop beta10000 | 4 | +6 | `R1b_citeseer_gat_nope.yaml` | `experiments/results/R1b_citeseer_nope/` |
| diffusion beta10 | 9 | +1 | `R1b_citeseer_gat_nope.yaml` | `experiments/results/R1b_citeseer_nope/` |

### All Other Cora/Citeseer Combos — 10/10 ✅

No issues with:
- Cora NoPE: zero_hop beta10/beta1, adjacency (all), diffusion beta10/beta1, full (all)
- Cora PE: adjacency, diffusion, full (all at iter50 t=0.1)
- Citeseer NoPE: zero_hop beta10/beta1, adjacency (all), diffusion beta10000/beta1, full (all)
- Citeseer PE: adjacency, diffusion, full (all at iter50 t=0.1)

---

## Existing Results File Locations

The existing (incomplete) result files that will need merging:

| # | File |
|---|------|
| 1 | `experiments/results/R1b_cora_nope/Cora_zero_hop_GAT_beta10000_clients10_hop2_iter50_t0.1_alpha0.5/results_...txt` |
| 2 | `experiments/results/R1b_cora_nope/Cora_diffusion_GAT_beta10000_clients10_hop2_iter50_t0.1_alpha0.5/results_...txt` |
| 3 | `experiments/results/R1b_citeseer_nope/Citeseer_zero_hop_GAT_beta10000_clients10_hop2_iter50_t0.1_alpha0.5/results_...txt` |
| 4 | `experiments/results/R1b_citeseer_nope/Citeseer_diffusion_GAT_beta10_clients10_hop2_iter50_t0.1_alpha0.5/results_...txt` |

---

## Launch Commands

```bash
cd /home/bosho/FP
PYTHON=/home/bosho/.conda/envs/fedgnn/bin/python

# 1. Cora NoPE zero_hop beta10000 (6 missing reps)
nohup $PYTHON -m src.experiments.run_experiments \
  --config experiments/configs/R1b/R1b_cora_gat_nope.yaml \
  --data_loading zero_hop --beta 10000 --repetitions 6 \
  > logs/r1b/rerun_cora_nope_zerohop_b10000.log 2>&1 &

# 2. Cora NoPE diffusion beta10000 (1 missing rep)
nohup $PYTHON -m src.experiments.run_experiments \
  --config experiments/configs/R1b/R1b_cora_gat_nope.yaml \
  --data_loading diffusion --beta 10000 --repetitions 1 \
  > logs/r1b/rerun_cora_nope_diff_b10000.log 2>&1 &

# 3. Citeseer NoPE zero_hop beta10000 (6 missing reps)
nohup $PYTHON -m src.experiments.run_experiments \
  --config experiments/configs/R1b/R1b_citeseer_gat_nope.yaml \
  --data_loading zero_hop --beta 10000 --repetitions 6 \
  > logs/r1b/rerun_citeseer_nope_zerohop_b10000.log 2>&1 &

# 4. Citeseer NoPE diffusion beta10 (1 missing rep)
nohup $PYTHON -m src.experiments.run_experiments \
  --config experiments/configs/R1b/R1b_citeseer_gat_nope.yaml \
  --data_loading diffusion --beta 10 --repetitions 1 \
  > logs/r1b/rerun_citeseer_nope_diff_b10.log 2>&1 &
```

> **Note:** Can only run 1 at a time due to GPU/RAM constraints (node has 62GB RAM, 10 Ray clients + experiment runner uses ~12GB each). Running 2 simultaneously caused the OOM crashes in the first place.

---

## After Completion — Merge Instructions

The runner creates new result files with timestamps. We need to merge the new rep results into the originals and recalculate mean/std.

For each of the 4 re-runs:

1. **Find the new result file:**
   ```bash
   ls -t experiments/results/R1b_<dataset>_nope/<Combo>_GAT_beta<*_clients10_hop2_iter50_t0.1_alpha0.5/results_*.txt | head -1
   ```

2. **Extract the new accuracy values** from `The global test results: [...]`

3. **Append to the original file's** `The global test results: [...]` list

4. **Recalculate** `average global test results` and `standard deviation global` over the combined 10 values

5. **Update RESULTS.md** with the corrected mean±std

### Example merge for Cora NoPE zero_hop beta10000:

Original had: `[0.613, 0.615, 0.621, 0.616]` (4 values, mean 0.6162)
New run will produce 6 more values, e.g. `[0.610, 0.619, 0.622, 0.614, 0.617, 0.620]`
Combined: 10 values → recalculate mean and std

---

## Full Experiment Status (from RESULTS.md)

### Cora — NoPE ✅ (after re-runs)

| Data Loading | Beta 10000 | Beta 10 | Beta 1 | Done |
|-------------|------------|---------|--------|------|
| zero_hop | 0.6162±0.0038 | 0.6145±0.0066 | 0.6717±0.0072 | ⚠️ b10000=4reps |
| adjacency | 0.7453±0.0224 | 0.7483±0.0144 | 0.8018±0.0075 | ✅ |
| diffusion | 0.7014±0.0089 | 0.7017±0.0196 | 0.7688±0.0111 | ⚠️ b10000=9reps |
| full | 0.8105±0.0086 | 0.8098±0.0091 | 0.8138±0.0063 | ✅ |

### Citeseer — NoPE ✅ (after re-runs)

| Data Loading | Beta 10000 | Beta 10 | Beta 1 | Done |
|-------------|------------|---------|--------|------|
| zero_hop | 0.5865±0.0063 | 0.5650±0.0074 | 0.6004±0.0082 | ⚠️ b10000=4reps |
| adjacency | 0.6550±0.0169 | 0.6481±0.0184 | 0.6800±0.0101 | ✅ |
| diffusion | 0.6178±0.0179 | 0.6099±0.0122 | 0.6378±0.0178 | ⚠️ b10=9reps |
| full | 0.6832±0.0090 | 0.6898±0.0089 | 0.6914±0.0099 | ✅ |

### Pubmed — NoPE 🔄 (still running + pending)

| Data Loading | Beta 10000 | Beta 10 | Beta 1 | Done |
|-------------|------------|---------|--------|------|
| zero_hop | 0.6433±0.0092 | 0.6747±0.0052 | 0.6367±0.0190 | ✅ |
| adjacency | 0.7936±0.0060 | 0.7734±0.0090 | 0.7194±0.0040 | ✅ |
| diffusion | 0.7563±0.0094 | | | 🔄 running |
| full | | | | ❌ pending |

### Pubmed — PE ❌ (all pending)

All 9 combos still need to run.
