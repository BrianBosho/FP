# OGBN-Arxiv Full-Mode Concurrency Sweep — Findings

## Executive Summary

We ran a full-mode concurrency sweep for `ogbn-arxiv` with 10 total FL clients and tested `max_concurrent_clients` from **2** through **10**.

### Final conclusion
- **Highest safe concurrency:** **7**
- **Failure boundary:** **8+**
- **Observed failure mode at 8, 9, 10:** **Ray worker kills due to host-memory pressure (OOM)** leading to **NaN final metrics**.

In this setup, **7 concurrent full-mode clients** is the highest confirmed safe value for production use.

---

## Sweep Setup

### Fixed settings
- Dataset: `ogbn-arxiv`
- Data loading: `full`
- Model: `GCN`
- Total federated clients: `10`
- Rounds: `5`
- Local epochs: `3`
- Beta: `10000`
- Hop: `2`
- `use_pe: false`
- `device: cuda`
- `feature_prop_device: cpu`
- `keep_data_on_gpu: false`
- `use_amp: true`
- `prop_dtype: bfloat16`
- `client_num_gpus: 0.1`

### Important runner detail
This codepath creates **all 10 Ray actors up front**, then uses `max_concurrent_clients` to batch training/evaluation/testing. Because of that, actor GPU reservation had to stay small (`client_num_gpus: 0.1`) so all client actors could be placed, while batching controlled the concurrency actually under test.

---

## Results Table

| Target concurrency | Status | Avg global acc | Duration | Interpretation |
|---:|---|---:|---:|---|
| 2 | ok | 0.5638 | 00:32:33 | Safe |
| 3 | ok | 0.5649 | 00:26:40 | Safe |
| 4 | ok | 0.5669 | 00:20:59 | Safe |
| 5 | ok | 0.5648 | 00:15:17 | Safe |
| 6 | ok | 0.5666 | 00:15:26 | Safe |
| 7 | ok | 0.5644 | 00:15:43 | Safe |
| 8 | oom_or_pressure | NaN | 00:01:47 | Unsafe |
| 9 | oom_or_pressure | NaN | 00:01:44 | Unsafe |
| 10 | oom_or_pressure | NaN | 00:01:44 | Unsafe |

---

## What Failed at 8+

For `N=8`, `N=9`, and `N=10`, logs show clear host-memory failure signals:

- Ray reported workers **killed due to memory pressure (OOM)**
- `ray.exceptions.OutOfMemoryError` was raised
- result aggregation degraded to `np.nanmean(...)`
- final summary rows contained **NaN** values for global/client metrics

Representative log evidence:
- `N8`: `2 Workers ... killed due to memory pressure (OOM)`
- `N9`: `3 Workers ... killed due to memory pressure (OOM)`
- `N10`: `3 Workers ... killed due to memory pressure (OOM)`

This means the limiting factor is **host RAM pressure**, not simply GPU scheduler admission.

---

## Interpretation

### Safe region
`N=2` through `N=7` all completed with valid metrics and normal runtimes.

### Unsafe region
`N=8` and above consistently failed with OOM-pressure behavior and invalid outputs.

### Practical recommendation
Use:
- **`max_concurrent_clients: 7`** for the most aggressive confirmed-safe configuration

If extra safety margin is desired for longer or less controlled runs, use:
- **`max_concurrent_clients: 6`** as a more conservative production setting

---

## Recommended Production Setting

### Aggressive safe setting
```yaml
max_concurrent_clients: 7
client_num_gpus: 0.1
device: cuda
feature_prop_device: cpu
keep_data_on_gpu: false
use_amp: true
prop_dtype: bfloat16
convergence_check_interval: 5
ray_num_gpus: 1
ray_object_store_memory_bytes: 4294967296
```

### Conservative setting
```yaml
max_concurrent_clients: 6
client_num_gpus: 0.1
```

---

## Questions Answered

1. **What is the highest safe concurrency?**
   - **7**

2. **Where does failure begin?**
   - **8**

3. **What fails first?**
   - **Host memory / Ray memory pressure**

4. **Did lower GPU reservation alone solve the problem?**
   - No. It helped actor placement, but the ultimate limit remained host RAM.

5. **Can we safely run 10 concurrent full-mode clients on this machine?**
   - No.

6. **What production setting should we recommend?**
   - **7** if maximizing throughput, **6** if keeping some headroom.

---

## Evidence Paths

### Summary files
- `logs/scalability/full_concurrency_sweep/sweep_20260505_175508_summary.tsv`
- `logs/scalability/full_concurrency_sweep/sweep_resume_5_10_20260505_202959_summary.tsv`

### Failure logs
- `logs/scalability/full_concurrency_sweep/N8_20260505_202959.log`
- `logs/scalability/full_concurrency_sweep/N9_20260505_202959.log`
- `logs/scalability/full_concurrency_sweep/N10_20260505_202959.log`

### Result artifacts
- `experiments/results/scalability/full_concurrency_sweep/N2/`
- `experiments/results/scalability/full_concurrency_sweep/N3/`
- `experiments/results/scalability/full_concurrency_sweep/N4/`
- `experiments/results/scalability/full_concurrency_sweep/N5/`
- `experiments/results/scalability/full_concurrency_sweep/N6/`
- `experiments/results/scalability/full_concurrency_sweep/N7/`
- `experiments/results/scalability/full_concurrency_sweep/N8/`
- `experiments/results/scalability/full_concurrency_sweep/N9/`
- `experiments/results/scalability/full_concurrency_sweep/N10/`
