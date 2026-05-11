#!/bin/bash
export WANDB_MODE=disabled
/home/bosho/.conda/envs/fedgnn/bin/python3 -m src.experiments.run_experiments --config conf/cora_split/cora_GCN_diffusion_beta10000.yaml
/home/bosho/.conda/envs/fedgnn/bin/python3 -m src.experiments.run_experiments --config conf/cora_split/cora_GAT_diffusion_beta10000.yaml
/home/bosho/.conda/envs/fedgnn/bin/python3 -m src.experiments.run_experiments --config conf/cora_split/cora_GCN_full_beta10000.yaml
/home/bosho/.conda/envs/fedgnn/bin/python3 -m src.experiments.run_experiments --config conf/cora_split/cora_GAT_full_beta10000.yaml
/home/bosho/.conda/envs/fedgnn/bin/python3 -m src.experiments.run_experiments --config conf/cora_split/cora_GCN_zero_hop_beta10000.yaml
/home/bosho/.conda/envs/fedgnn/bin/python3 -m src.experiments.run_experiments --config conf/cora_split/cora_GAT_zero_hop_beta10000.yaml
/home/bosho/.conda/envs/fedgnn/bin/python3 -m src.experiments.run_experiments --config conf/cora_split/cora_GCN_adjacency_beta10000.yaml
/home/bosho/.conda/envs/fedgnn/bin/python3 -m src.experiments.run_experiments --config conf/cora_split/cora_GAT_adjacency_beta10000.yaml
