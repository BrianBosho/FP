# Phase-Based Config Map

These configs implement the anchor-dataset-first execution order.

## Authoritative execution order

1. `phase_1_cora_intrinsic.yaml`
2. `phase_1_cora_intrinsic_heat_kernel.yaml`
3. `phase_2_cora_ablation.yaml` (manifest)
4. `phase_2_cora_ablation_appnp_alpha.yaml`
5. `phase_2_cora_ablation_epsilon_cora.yaml`
6. `phase_2_cora_ablation_hop_depth.yaml`
7. `phase_3_cora_downstream.yaml`
8. `phase_4_homophilic_reproduction.yaml`
9. `phase_5_scalability_ogbn_arxiv.yaml`
10. `phase_6_heterophily_stress.yaml`

## Backward compatibility

The old layer-based configs are still present:

- `L1_L3_primary.yaml`
- `L1_L3_heat_kernel_ref.yaml`
- `L1_L3_hop2_ablation.yaml`
- `L4_baselines.yaml`
- `L4_downstream_operators.yaml`
- `L4_downstream_heterophilic.yaml`
- `ablations_intrinsic.yaml`

They remain useful as references and for ad-hoc debugging, but they are no longer the preferred execution order.

## Runner notes

- `run_intrinsic_eval.py` accepts **flat grids** only.
- `run_downstream_eval.py` accepts **flat grids** only.
- Therefore `phase_2_cora_ablation.yaml` is a manifest, and its companion files are the runnable configs.
