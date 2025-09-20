#!/usr/bin/env python3
"""
Framework for analyzing propagation vs training time in federated GNN experiments.

This script creates a comprehensive analysis of:
1. Propagation time (preprocessing cost) 
2. Training time (federated learning cost)
3. Efficiency ratios and convergence analysis
"""

import pandas as pd
import json
import glob
import numpy as np
from pathlib import Path
import os
import re
from typing import Dict, List, Tuple, Optional

def extract_experiment_params(filename: str) -> Dict:
    """Extract experiment parameters from filename patterns."""
    params = {}
    
    # Extract from propagation stats filename
    # Format: prop_exp_20250730-203943_diffusion_beta_10_hop_1.json
    if 'prop_exp' in filename:
        match = re.search(r'prop_exp_(\d{8}-\d{6})_([^_]+)_beta_(\d+)_hop_(\d+)', filename)
        if match:
            params['timestamp'] = match.group(1)
            params['propagation_mode'] = match.group(2)
            params['beta'] = int(match.group(3))
            params['hop'] = int(match.group(4))
    
    # Extract from training results filename  
    # Format: results_Pubmed_adjacency_GCN_beta10_clients10_20250730_202650.json
    elif 'results_' in filename:
        match = re.search(r'results_([^_]+)_([^_]+)_([^_]+)_beta(\d+)_clients(\d+)_(\d{8}_\d{6})', filename)
        if match:
            params['dataset'] = match.group(1)
            params['data_loading_option'] = match.group(2)
            params['model_type'] = match.group(3)
            params['beta'] = int(match.group(4))
            params['num_clients'] = int(match.group(5))
            params['timestamp'] = match.group(6)
    
    return params

def analyze_propagation_stats(prop_file: str) -> Dict:
    """Analyze propagation statistics from a single experiment."""
    with open(prop_file, 'r') as f:
        data = json.load(f)
    
    # Extract client runtimes
    client_runtimes = []
    clients_converged = 0
    total_iterations = 0
    
    for client in data.get('clients', []):
        runtime = client.get('runtime', 0)
        converged = client.get('converged', False)
        iterations = client.get('iterations', 0)
        
        client_runtimes.append(runtime)
        if converged:
            clients_converged += 1
        total_iterations += iterations
    
    # Calculate propagation metrics
    metrics = {
        'experiment_id': data.get('experiment_id', ''),
        'propagation_mode': data.get('propagation_mode', ''),
        'num_clients': data.get('num_clients', 0),
        'beta': data.get('beta', 0),
        'hop': data.get('hop', 0),
        'use_pe': data.get('use_pe', False),
        
        # Timing metrics
        'total_propagation_time': sum(client_runtimes),
        'avg_propagation_time': np.mean(client_runtimes) if client_runtimes else 0,
        'max_propagation_time': max(client_runtimes) if client_runtimes else 0,
        'min_propagation_time': min(client_runtimes) if client_runtimes else 0,
        'propagation_std': np.std(client_runtimes) if client_runtimes else 0,
        
        # Convergence metrics
        'clients_converged': clients_converged,
        'convergence_rate': clients_converged / len(client_runtimes) if client_runtimes else 0,
        'avg_iterations': total_iterations / len(client_runtimes) if client_runtimes else 0,
        'max_iterations': max([c.get('iterations', 0) for c in data.get('clients', [])]),
        
        # Raw data for matching
        'prop_file': prop_file,
        'client_runtimes': client_runtimes
    }
    
    return metrics

def analyze_training_results(train_file: str) -> Dict:
    """Analyze training results from a single experiment.""" 
    with open(train_file, 'r') as f:
        data = json.load(f)
    
    config = data.get('experiment_config', {})
    duration = data.get('duration', {})
    summary = data.get('summary', {})
    rounds = data.get('rounds', [])
    
    metrics = {
        'dataset': config.get('dataset', ''),
        'data_loading_option': config.get('data_loading_option', ''),
        'model_type': config.get('model_type', ''),
        'num_clients': config.get('num_clients', 0),
        'beta': config.get('beta', 0),
        'hop': config.get('hop', 1),
        'fulltraining_flag': config.get('fulltraining_flag', False),
        
        # Training metrics
        'training_duration_seconds': duration.get('seconds', 0),
        'training_duration_formatted': duration.get('formatted', ''),
        'avg_global_result': summary.get('average_global_result', 0),
        'avg_client_result': summary.get('average_client_result', 0),
        'std_global': summary.get('std_global', 0),
        'std_client': summary.get('std_client', 0),
        
        # Round analysis
        'training_rounds': len(rounds),
        'first_round': rounds[0].get('round', 1) if rounds else 1,
        'last_round': rounds[-1].get('round', 1) if rounds else 1,
        'is_complete_log': len(rounds) >= 10,  # Assuming 10 rounds expected
        
        # Raw data
        'train_file': train_file,
        'rounds_data': rounds
    }
    
    return metrics

def match_propagation_to_training(prop_metrics: List[Dict], train_metrics: List[Dict]) -> List[Dict]:
    """Match propagation stats with corresponding training results."""
    matched_experiments = []
    
    for train in train_metrics:
        # Find matching propagation experiment
        matching_prop = None
        
        for prop in prop_metrics:
            # Match on key parameters
            if (prop['beta'] == train['beta'] and 
                prop['num_clients'] == train['num_clients'] and
                prop['hop'] == train['hop']):
                
                # Check if data loading option matches propagation mode
                data_option = train['data_loading_option']
                prop_mode = prop['propagation_mode']
                
                # Map data loading options to propagation modes
                if ((data_option == 'diffusion' and prop_mode == 'diffusion') or
                    (data_option == 'adjacency' and prop_mode == 'adjacency') or  
                    (data_option == 'full' and prop_mode in ['full', 'diffusion']) or
                    (data_option.startswith('zero') and prop_mode in ['zero_hop', 'diffusion'])):
                    matching_prop = prop
                    break
        
        if matching_prop:
            # Combine metrics
            combined = {
                # Experiment identifiers
                'experiment_group': Path(train['train_file']).parent.parent.name,
                'dataset': train['dataset'],
                'model_type': train['model_type'], 
                'data_loading_option': train['data_loading_option'],
                'propagation_mode': matching_prop['propagation_mode'],
                'beta': train['beta'],
                'num_clients': train['num_clients'],
                'hop': train['hop'],
                'use_pe': matching_prop['use_pe'],
                
                # Training metrics
                'training_duration_seconds': train['training_duration_seconds'],
                'training_duration_formatted': train['training_duration_formatted'],
                'avg_global_result': train['avg_global_result'],
                'avg_client_result': train['avg_client_result'],
                'training_rounds': train['training_rounds'],
                'is_complete_log': train['is_complete_log'],
                
                # Propagation metrics
                'total_propagation_time': matching_prop['total_propagation_time'],
                'avg_propagation_time': matching_prop['avg_propagation_time'],
                'max_propagation_time': matching_prop['max_propagation_time'],
                'min_propagation_time': matching_prop['min_propagation_time'],
                'propagation_std': matching_prop['propagation_std'],
                'clients_converged': matching_prop['clients_converged'],
                'convergence_rate': matching_prop['convergence_rate'],
                'avg_iterations': matching_prop['avg_iterations'],
                'max_iterations': matching_prop['max_iterations'],
                
                # Efficiency ratios
                'propagation_to_training_ratio': (matching_prop['total_propagation_time'] / 
                                                train['training_duration_seconds'] if train['training_duration_seconds'] > 0 else 0),
                'avg_prop_to_training_ratio': (matching_prop['avg_propagation_time'] / 
                                             train['training_duration_seconds'] if train['training_duration_seconds'] > 0 else 0),
                'propagation_overhead_pct': (matching_prop['total_propagation_time'] / 
                                           train['training_duration_seconds'] * 100 if train['training_duration_seconds'] > 0 else 0),
                
                # Time per round analysis
                'propagation_time_per_round': (matching_prop['total_propagation_time'] / 
                                            train['training_rounds'] if train['training_rounds'] > 0 else 0),
                'training_time_per_round': (train['training_duration_seconds'] / 
                                          train['training_rounds'] if train['training_rounds'] > 0 else 0),
                
                # File references
                'prop_file': matching_prop['prop_file'],
                'train_file': train['train_file']
            }
            
            matched_experiments.append(combined)
    
    return matched_experiments

def analyze_results_directory(results_dir: str) -> pd.DataFrame:
    """Analyze all experiments in a results directory."""
    
    # Find all propagation stats files
    prop_files = glob.glob(f"{results_dir}/**/propagation_stats/prop_exp_*.json", recursive=True)
    print(f"Found {len(prop_files)} propagation stats files")
    
    # Find all training results files  
    train_files = glob.glob(f"{results_dir}/**/results_*.json", recursive=True)
    print(f"Found {len(train_files)} training results files")
    
    # Analyze propagation stats
    prop_metrics = []
    for prop_file in prop_files:
        try:
            metrics = analyze_propagation_stats(prop_file)
            prop_metrics.append(metrics)
            print(f"Analyzed propagation: {os.path.basename(prop_file)}")
        except Exception as e:
            print(f"Error analyzing {prop_file}: {e}")
    
    # Analyze training results
    train_metrics = []
    for train_file in train_files:
        try:
            metrics = analyze_training_results(train_file)
            train_metrics.append(metrics)
            print(f"Analyzed training: {os.path.basename(train_file)}")
        except Exception as e:
            print(f"Error analyzing {train_file}: {e}")
    
    # Match propagation with training
    matched_experiments = match_propagation_to_training(prop_metrics, train_metrics)
    print(f"Matched {len(matched_experiments)} experiments")
    
    # Create DataFrame
    df = pd.DataFrame(matched_experiments)
    
    return df

# Example usage and analysis
if __name__ == "__main__":
    results_dir = "results"
    
    print("Starting propagation vs training time analysis...")
    df = analyze_results_directory(results_dir)
    
    if not df.empty:
        print(f"\nAnalyzed {len(df)} experiments")
        print("\nSample results:")
        print(df[['dataset', 'data_loading_option', 'beta', 'use_pe', 
                 'total_propagation_time', 'training_duration_seconds', 
                 'propagation_overhead_pct', 'convergence_rate']].head())
        
        # Save results
        output_file = "propagation_training_analysis.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print(f"Average propagation overhead: {df['propagation_overhead_pct'].mean():.2f}%")
        print(f"Average convergence rate: {df['convergence_rate'].mean():.2f}%")
        print(f"Experiments with complete logs: {df['is_complete_log'].sum()}/{len(df)}")
        
    else:
        print("No matched experiments found!")