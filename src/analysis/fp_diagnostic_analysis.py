import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_propagation_stats(stats_dir):
    """
    Load all propagation diagnostic JSONs and aggregate metrics per experiment.
    """
    all_stats = []
    stats_path = Path(stats_dir)
    
    if not stats_path.exists():
        print(f"Stats directory not found: {stats_dir}")
        return pd.DataFrame()

    for file in stats_path.glob("*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            
        exp_id = data['experiment_id']
        mode = data['propagation_mode']
        
        # Aggregate across clients
        client_metrics = []
        for client in data['clients']:
            # Use the last value of each metric for final state analysis
            metrics = {
                'client_id': client['client_id'],
                'final_residual': client['residuals'][-1] if client['residuals'] else np.nan,
                'final_variance': client['variances'][-1] if client['variances'] else np.nan,
                'final_norm_drift': client['norm_drifts'][-1] if client['norm_drifts'] else np.nan,
                'iterations': client['iterations'],
                'nodes_unknown_pct': client['nodes_unknown'] / client['nodes_total']
            }
            # Initial vs Final Energy (Smoothness recovery)
            if client.get('energies'):
                metrics['initial_energy'] = client['energies'][0]['per_node']
                metrics['final_energy'] = client['energies'][-1]['per_node']
                metrics['energy_reduction'] = (metrics['initial_energy'] - metrics['final_energy']) / metrics['initial_energy']
            
            client_metrics.append(metrics)
            
        # Mean metrics for the whole graph partition
        df_clients = pd.DataFrame(client_metrics)
        agg_stats = {
            'stats_id': exp_id,
            'mode': mode,
            'avg_residual': df_clients['final_residual'].mean(),
            'avg_variance': df_clients['final_variance'].mean(),
            'avg_norm_drift': df_clients['final_norm_drift'].mean(),
            'avg_energy_reduction': df_clients.get('energy_reduction', pd.Series([np.nan])).mean(),
            'max_residual': df_clients['final_residual'].max(),
            'min_variance': df_clients['final_variance'].min()
        }
        all_stats.append(agg_stats)
        
    return pd.DataFrame(all_stats)

def analyze_correlations(results_df, stats_df):
    """
    Merge accuracy results with structural stats and compute correlations.
    """
    # Note: Linking needs to be robust to experiment naming conventions
    # For now, we'll try to match by dataset, mode, and iterations
    
    # This is a placeholder for the merging logic which depends on exact path matching
    # In a real run, we should ensure the experiment runner logs the stats_id in the main results JSON.
    pass

if __name__ == "__main__":
    # Example usage (to be run after experiments complete)
    STATS_DIR = "results/ablation/fp_diagnostics/propagation_stats"
    RESULTS_DIR = "results/ablation/fp_diagnostics"
    
    stats = load_propagation_stats(STATS_DIR)
    if not stats.empty:
        print("\n=== Aggregated Structural Diagnostics ===")
        print(stats.head())
        
        # Plotting distributions
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=stats, x='avg_residual', y='avg_variance', hue='mode')
        plt.title("Structural Conscience: Residual vs Variance")
        plt.savefig("results/ablation/fp_diagnostics/structural_conscience.png")
