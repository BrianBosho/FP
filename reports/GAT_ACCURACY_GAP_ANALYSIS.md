# GAT Accuracy Gap Analysis: 81% vs 70% on Cora

**Date:** 2026-04-28
**Question:** Both runs use `data_loading: full` (the federated upper bound). Why does one reach 81% and the other 70%? What is causing the 10-12 point gap?

This document is a **delta accounting** of the two configs and a **hypothesis ranking** for what is responsible.

---

## 1. The two runs

Both runs are GAT on Cora with:

- `num_clients: 10`
- `beta: 10000` (near-IID Dirichlet)
- `data_loading: full` (federated upper bound — every client has full features)
- `hop: 2`
- 8 heads, hidden_dim=8, dropout=0.6, no normalization

So the model architecture, partition, and feature regime are identical. The differences are entirely in the FL training and aggregation hyperparameters.

| Parameter | R1b (~70%) | Working (81%) | Source of "Working" default |
|-----------|-----------|---------------|------------------------------|
| `aggregation` | `fedavg_weighted` | `mean` | `base.yaml:22` (default) |
| `num_rounds` | 400 | 200 | `cora_gat_beta10000.yaml:15` |
| `early_stopping_patience` | 30 | 10 | `base.yaml:15` (default) |
| `optimizer` / `lr` | varies (Adam/0.005, SGD/0.5) | SGD / 0.5 | `cora_gat_beta10000.yaml` |
| `experiment_seed` | 52 | `null` (unseeded) | `base.yaml:35` (default) |
| `repetitions` | 2 | 5 | `base.yaml:12` (default) |
| `num_iterations` | 50 | 80 | `base.yaml:78` |
| `diffusion_t` | 0.1 | 1.0 | `base.yaml:109` |

### Important: which deltas are inert under `data_loading: full`?

`data_loading: full` triggers `imputation_method == "full"` in `loaders.py:78-80`:

```python
elif imputation_method == "full":
    use_feature_prop = False
    full_data = True
```

`use_feature_prop = False` means **no feature propagation runs**. Therefore:

- `num_iterations` — **inert** (no FP iterations executed)
- `diffusion_t` — **inert** (no diffusion kernel built)
- `feature_prop_tolerance` — **inert**

Setting these in R1b configs has no effect on the outcome. They are red herrings for this gap.

The deltas that **can** affect the outcome are:

- `aggregation` (fedavg_weighted vs mean)
- `num_rounds` × `early_stopping_patience` (training duration / when to stop)
- `optimizer` / `lr` (only matters when optimizer differs — SGD/0.5 vs Adam/0.005)
- `experiment_seed` (run-to-run variance baseline)
- `repetitions` (Monte Carlo estimate quality, not the underlying mean)

---

## 2. Hypothesis ranking

### H1 (primary): `aggregation: fedavg_weighted` is the dominant cause

**Evidence:**
- Switching only `aggregation: mean` → `fedavg_weighted` while holding everything else equal moves results from 81% to 70.75% (R1b Adam) and 63.1% (R1b SGD). The investigation report holds this as the primary effect.
- Mechanistically plausible: GAT's attention coefficients adapt to local subgraph topology. Weighted aggregation by `|D_k|` lets larger clients' attention patterns dominate the aggregated `a` vector, biasing it toward the geometry of large subgraphs and degrading generalization on smaller ones.

**What we don't yet know:**
- Whether the effect persists when training duration is matched. The R1b configs train for 400 rounds with patience 30; the working config trains for 200 with patience 10. Longer training under `fedavg_weighted` could be amplifying overfit damage, not the aggregation per se.

**Confidence:** High that this is part of the gap. Medium that this is the entire gap.

### H2: Training-duration × overfit interaction

**Evidence:**
- Investigation report notes GAT hits 100% training accuracy within 2-3 rounds.
- 200 rounds + patience 10 → likely stops early (well before 200) and locks in mid-training accuracy.
- 400 rounds + patience 30 → trains much longer, and any per-client gradient drift continues compounding.

**Mechanistically:** Once a client overfits its local subgraph, longer training pulls global params further toward an over-specialized point in weight space. With `fedavg_weighted`, the over-specialized point is biased toward large clients. With `mean`, it's a more balanced point.

**Confidence:** Medium-high that this contributes. The R1b configs were built to push for higher accuracy (longer training, larger patience) but in a regime where GAT has already overfit, this hurts.

### H3: SGD/0.5 vs Adam/0.005

**Evidence:**
- R1b SGD/0.5 + fedavg_weighted: 63.1%
- R1b Adam/0.005 + fedavg_weighted: 70.75%
- Working SGD/0.5 + mean: 81.28%

When `aggregation = mean`, SGD/0.5 works. When `aggregation = fedavg_weighted`, SGD/0.5 is **worse** than Adam/0.005. This suggests the optimizer × aggregation interaction is real: SGD's higher effective step compounds the "drift toward large clients" problem; Adam's adaptive per-parameter scaling partially compensates.

**Confidence:** Medium. The gap between SGD-fedavg and Adam-fedavg (~7 points) is comparable to part of the gap to baseline. But this is a within-fedavg comparison and doesn't directly explain the 10-12 point gap from 81% baseline.

### H4: Variance from `experiment_seed`

**Evidence:**
- Working config is unseeded (`experiment_seed: null`), so 5 reps draw fresh inits each time. Reported numbers: `82.3, 79.6, 81.9, 82.4, 80.2` — std ~1.1%.
- R1b uses `experiment_seed: 52` over 2 reps. The two reps are correlated (only initial-weight seed and dropout RNG vary).

**Effect on the gap:** A 1-2 point variance is plausible but is much smaller than the observed 10-12 point gap. Not a primary cause.

**Confidence:** Low that this explains more than 1-2 points.

### H5: `repetitions: 2` vs `5` Monte Carlo noise

The R1b numbers are averaged over 2 reps, the working over 5. The expected sampling noise of a 2-rep mean is √(5/2) ≈ 1.6× larger than a 5-rep mean. If true mean were 75%, observed 70% on 2 reps vs 81% on 5 reps would be ~3 sigma — possible but unlikely at the magnitudes reported.

**Confidence:** Low. Insufficient sample size to fully rule out, but not the main story.

---

## 3. What we actually know vs. assume

| Claim | Status |
|-------|--------|
| `aggregation: mean` in the working config drives most of the lift | **Strong, but not isolated.** The working config also has shorter training duration and unseeded reps. |
| `data_loading: full` is the upper bound regime in both runs | **Confirmed by code reading.** `use_feature_prop=False, full_data=True`. |
| FP-related parameters (`num_iterations`, `diffusion_t`) are inert in this comparison | **Confirmed by code reading.** No FP runs under `full`. |
| GAT overfits within 2-3 rounds | **Reported in the investigation document**, but not verified in this analysis. |
| The aggregation effect is GAT-specific (GCN is robust) | **Reported, not verified.** The mechanism (attention coefficients depending on local topology) is plausible. |

---

## 4. Minimal ablation to decompose the gap cleanly

To isolate each contributor, run these six configurations on Cora-GAT-`full`-hop2-beta10000 with **everything else identical**, 5 reps, seeded:

| ID | aggregation | optimizer | lr | num_rounds | patience | Hypothesized effect |
|----|-------------|-----------|----|------------|---------:|----------------------|
| A | mean | SGD | 0.5 | 200 | 10 | Baseline (working config) — expect 81% |
| B | mean | SGD | 0.5 | 400 | 30 | Test long-training effect under `mean` |
| C | fedavg_weighted | SGD | 0.5 | 200 | 10 | Aggregation alone, short training |
| D | fedavg_weighted | SGD | 0.5 | 400 | 30 | Full R1b SGD config (currently 63%) |
| E | fedavg_weighted | Adam | 0.005 | 200 | 10 | Aggregation + Adam, short training |
| F | fedavg_weighted | Adam | 0.005 | 400 | 30 | Full R1b Adam config (currently 70.75%) |

**Expected reads:**
- `A − C` = pure aggregation effect, short training, SGD
- `A − B` = pure training-duration effect under `mean`
- `C − D` = training-duration effect under `fedavg_weighted` + SGD
- `E − F` = training-duration effect under `fedavg_weighted` + Adam
- `D − F` = optimizer interaction within `fedavg_weighted`

This pins each hypothesis to a specific number. If the report's claim is right, A − C should be the largest single contributor.

**Cost:** 6 configs × 5 reps × ~3 min/rep ≈ **90 minutes on a single GPU**. Cheap.

---

## 5. Why this matters before publishing R1b

The R1b table is supposed to populate a GAT row across `[full, zero_hop, adjacency, diffusion]`. If the GAT baselines are tuned with `fedavg_weighted` × long training, every cell will be 8-15 points below the achievable ceiling, and the FP comparison rows will be reported against an unfair baseline.

The fix is **not** to silently switch all GAT configs to `mean`. The fix is:

1. **Verify** the gap decomposition with the ablation in §4 so we know what the controlling variables are.
2. **Choose** an aggregation strategy with a documented rationale (e.g., "all GAT runs use `mean` because aggregation × attention interacts in non-IID settings, see §X").
3. **Apply** the same choice across all data_loading modes for fair comparison.
4. **Report** the choice in the paper / report so reviewers can audit it.

If `fedavg_weighted` is the FedAvg-paper-canonical choice, then either:
- Document explicitly that this configuration causes a known accuracy loss for GAT and report results anyway, or
- Use an aggregation variant that is robust to attention coefficients (one option: aggregate `W` weighted but `a` uniformly; another: use FedAvgM).

---

## 6. Immediate next step

Run the six-config ablation in §4. The ranking emerging from those numbers determines the GAT baseline strategy for R1b.

Until that is run, the safest claim is: **"The working config achieves 81-82% on Cora-GAT-full. The R1b configs reach only 63-71%. Switching `aggregation` from `fedavg_weighted` to `mean` is the largest single known contributor; the contribution of training duration and optimizer choice has not yet been isolated."**

---

*Companion document: `CODEBASE_REVIEW.md` covers the broader stack review.*
