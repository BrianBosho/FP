# FL benchmark harness

A thin wrapper around `src.experiments.run_experiments` that lets us:

1. run one experiment *variant* (baseline or any fix candidate) across N seeds,
2. capture the full output in a structured, machine-readable form,
3. compare two variants with mean/std, deltas, and a significance test.

No code under `src/` is modified — this is pure orchestration, so it stays
backward-compatible with the current training pipeline.

## Directory layout

```
scripts/bench/
    run_variant.py       # run a variant (baseline or candidate)
    compare_variants.py  # A/B compare two variant runs
    common.py            # shared helpers (provenance, parsing, stats)

conf/bench/              # auto-generated per-seed configs (gitignored)

bench/
    runs/<variant>_<ts>/ # one directory per invocation of run_variant.py
        manifest.json
        runs.csv
        config_used.yaml
        seed_<N>/
            config.yaml
            stdout.log
            stderr.log
            <run_experiments output tree>
    compare/<base>_vs_<cand>_<ts>/
        comparison.md
        comparison.csv
        per_round.csv
        provenance.json
```

## 1. Run a variant

Reproduce the `cora_minimal` run (the smoke-test you've been using) as the
*baseline* variant, N=3 seeds:

```bash
python scripts/bench/run_variant.py \
    --base-config conf/cora_minimal.yaml \
    --variant baseline \
    --seeds 0,1,2
```

A single seed is fine for a fast sanity check:

```bash
python scripts/bench/run_variant.py \
    --base-config conf/cora_minimal.yaml \
    --variant baseline \
    --seeds 0
```

### Overrides

Any config key can be overridden from the CLI. Values are parsed as YAML so
lists and bools work as expected:

```bash
python scripts/bench/run_variant.py \
    --base-config conf/cora_minimal.yaml \
    --variant lr_0.1 \
    --seeds 0-2 \
    --override lr=0.1
```

FedAvg-weighted aggregation is available behind `aggregation=fedavg_weighted`.
For an A/B comparison against a legacy mean baseline, the *candidate* run is:

```bash
python scripts/bench/run_variant.py \
    --base-config conf/cora_minimal.yaml \
    --variant fedavg_weighted \
    --seeds 0,1,2 \
    --override aggregation=fedavg_weighted
```

Forced per-seed:

* `results_dir` — set to the seed's directory so runs don't overwrite each other.
* `repetitions` — forced to 1. Cross-seed variance is produced by running
  separate subprocesses, not by the inner `repetitions` loop. This also keeps
  the per-seed timing meaningful.
* `experiment_seed` — passed through to partitioning, model initialization,
  client training, and RFP generation.

### What gets captured

For each seed we log:

* `seed_<N>/config.yaml` — exact config file the subprocess loaded.
* `seed_<N>/stdout.log`, `stderr.log` — full training output and any warnings.
* `seed_<N>/**/results_*.json` — the per-combination result file written by
  `run_experiments` (contains `experiment_config`, per-round `global_result` /
  `client_result`, summary stats, and duration).
* `seed_<N>/**/training_*.csv` — per-client per-epoch loss/accuracy table
  written by `save_results_to_csv`.
* `manifest.json` — variant name, overrides, seeds, git commit + dirty flag,
  Python / torch / PyG / Ray versions, GPU info, and per-seed status + duration.
* `runs.csv` — one flat row per `(seed, experiment-combination)` with final
  metrics and a pointer to the raw `results_*.json`.

## 2. Compare two variants

```bash
python scripts/bench/compare_variants.py \
    --baseline  bench/runs/baseline_20260420_153000 \
    --candidate bench/runs/fedavg_weighted_20260420_154500
```

The comparison is grouped by the *experiment key* —
`(dataset, data_loading_option, model_type, num_clients, beta, hop, use_pe, fulltraining_flag)`.
Two runs are compared only when their keys match exactly; heterogeneous
configurations are not silently averaged.

For each matching key you get, per metric (`avg_global_acc`, `avg_client_acc`):

* baseline mean ± std (n)
* candidate mean ± std (n)
* Δ = candidate − baseline
* Welch's two-sample t-statistic and p-value (scipy)
* Cohen's d effect size

Per-round curves for both variants are written to `per_round.csv` in long form
so any plotting tool can draw convergence curves without coupling this harness
to matplotlib.

## Notes, caveats, and known limitations

* **Seeding is now end-to-end for the main experiment path.** The harness writes
  `experiment_seed` into each seed config; `src/` uses it for partitioning,
  server model initialization, client training, per-client RFP, and global PE.
  Keep `repetitions: 1` in this harness so each subprocess maps cleanly to one
  seed.

* **Each seed runs in its own subprocess.** This isolates Ray and CUDA state
  between seeds and makes `run_experiments`'s internal `ray.shutdown()` /
  `ray.init()` dance much less brittle.  The trade-off is process-startup
  overhead on the order of a few seconds per seed — negligible for anything
  larger than the `cora_minimal` smoke test.

* **Memory optimization knobs travel with the config.** Keys like
  `feature_prop_device`, `keep_data_on_gpu`, `max_concurrent_clients`,
  `client_num_gpus`, and `ray_num_gpus` are
  preserved from the base config and can be overridden per variant, so a
  memory-only change can be A/B tested the same way as a correctness fix.

* **The comparison is variant-level, not seed-paired.** We use Welch's t-test
  rather than a paired test. Seed N in variant A and seed N in variant B are
  intended to share partition/model/training/RFP seeds, but variant-level
  failures can still make sample sizes asymmetric.
