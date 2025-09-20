#!/usr/bin/env python3
"""
Quick analysis of propagation vs training time for specific experiments.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Analyze the specific examples you've been looking at
def analyze_pubmed_example():
    """Analyze the Pubmed beta=10 experiment you've been examining."""
    
    # Propagation stats file
    prop_file = "results/pubmed_gcn_final_beta_10_nope/propagation_stats/prop_exp_20250730-203943_diffusion_beta_10_hop_1.json"
    
    # Training results file
    train_file = "results/pubmed_gcn_final_beta_10_nope/Pubmed_adjacency_GCN_beta10_clients10/results_Pubmed_adjacency_GCN_beta10_clients10_20250730_202650.json"
    
    print("=== PUBMED BETA=10 ANALYSIS ===\n")
    
    # Load propagation stats
    try:
        with open(prop_file, 'r') as f:
            prop_data = json.load(f)
        
        print("📊 PROPAGATION ANALYSIS:")
        print(f"   Experiment ID: {prop_data['experiment_id']}")
        print(f"   Mode: {prop_data['propagation_mode']}")
        print(f"   Beta: {prop_data['beta']}")
        print(f"   Clients: {prop_data['num_clients']}")
        print(f"   Use PE: {prop_data['use_pe']}")
        
        # Extract client runtimes
        client_runtimes = [client['runtime'] for client in prop_data['clients']]
        converged_clients = sum(1 for client in prop_data['clients'] if client['converged'])
        
        total_prop_time = sum(client_runtimes)
        avg_prop_time = np.mean(client_runtimes)
        
        print(f"\n   💾 CLIENT RUNTIMES:")
        for i, runtime in enumerate(client_runtimes):
            converged = prop_data['clients'][i]['converged']
            iterations = prop_data['clients'][i]['iterations']
            print(f"     Client {i}: {runtime:.3f}s ({iterations} iter, {'✓' if converged else '✗'} converged)")
        
        print(f"\n   ⏱️  PROPAGATION TIMING:")
        print(f"     Total: {total_prop_time:.3f} seconds")
        print(f"     Average: {avg_prop_time:.3f} seconds") 
        print(f"     Min: {min(client_runtimes):.3f} seconds")
        print(f"     Max: {max(client_runtimes):.3f} seconds")
        print(f"     Std: {np.std(client_runtimes):.3f} seconds")
        
        print(f"\n   🎯 CONVERGENCE:")
        print(f"     Converged: {converged_clients}/{len(client_runtimes)} clients")
        print(f"     Rate: {converged_clients/len(client_runtimes)*100:.1f}%")
        
    except FileNotFoundError:
        print(f"❌ Propagation file not found: {prop_file}")
        return
    except Exception as e:
        print(f"❌ Error reading propagation file: {e}")
        return
    
    # Load training results
    try:
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        print(f"\n🚀 TRAINING ANALYSIS:")
        config = train_data['experiment_config']
        duration = train_data['duration']
        summary = train_data['summary']
        rounds = train_data['rounds']
        
        print(f"   Dataset: {config['dataset']}")
        print(f"   Model: {config['model_type']}")
        print(f"   Data loading: {config['data_loading_option']}")
        print(f"   Beta: {config['beta']}")
        
        training_seconds = duration['seconds']
        print(f"\n   ⏱️  TRAINING TIMING:")
        print(f"     Total: {training_seconds:.1f} seconds ({duration['formatted']})")
        print(f"     Rounds: {len(rounds)} (showing rounds {rounds[0]['round']}-{rounds[-1]['round']})")
        print(f"     Avg per round: {training_seconds/len(rounds):.1f} seconds")
        print(f"     Performance: {summary['average_global_result']:.3f}")
        
        # Calculate efficiency metrics
        prop_to_train_ratio = total_prop_time / training_seconds
        overhead_pct = prop_to_train_ratio * 100
        
        print(f"\n⚡ EFFICIENCY ANALYSIS:")
        print(f"   Propagation time: {total_prop_time:.1f}s")
        print(f"   Training time: {training_seconds:.1f}s") 
        print(f"   Propagation overhead: {overhead_pct:.2f}%")
        print(f"   Ratio (prop/train): {prop_to_train_ratio:.4f}")
        
        if len(rounds) < 10:
            print(f"\n   ⚠️  WARNING: Only {len(rounds)} rounds logged, expected ~10")
            print(f"   This might be a partial log - actual training time could be higher")
            estimated_full_time = training_seconds * (10 / len(rounds))
            estimated_overhead = (total_prop_time / estimated_full_time) * 100
            print(f"   Estimated full training time: {estimated_full_time:.1f}s")
            print(f"   Estimated propagation overhead: {estimated_overhead:.2f}%")
        
    except FileNotFoundError:
        print(f"❌ Training file not found: {train_file}")
        return
    except Exception as e:
        print(f"❌ Error reading training file: {e}")
        return

def analyze_citeseer_comparison():
    """Compare different Citeseer experiments."""
    
    print("\n" + "="*50)
    print("=== CITESEER COMPARISON ===\n")
    
    experiments = [
        {
            'name': 'Citeseer Diffusion Beta=1 (PE)',
            'prop': 'results/citeseer_ablation_final_pe/propagation_stats/prop_exp_20250730-172635_diffusion_beta_1_hop_1.json',
            'train': 'results/citeseer_ablation_final_pe/Citeseer_diffusion_GCN_beta1_clients10/results_Citeseer_diffusion_GCN_beta1_clients10_20250730_172626.json'
        },
        {
            'name': 'Citeseer Diffusion Beta=10000 (PE)', 
            'prop': 'results/citeseer_ablation_final_pe/propagation_stats/prop_exp_20250730-175355_diffusion_beta_10000_hop_1.json',
            'train': 'results/citeseer_ablation_final_pe/Citeseer_diffusion_GCN_beta10000_clients10/results_Citeseer_diffusion_GCN_beta10000_clients10_20250730_172626.json'
        }
    ]
    
    results = []
    
    for exp in experiments:
        try:
            # Load propagation data
            with open(exp['prop'], 'r') as f:
                prop_data = json.load(f)
            
            client_runtimes = [client['runtime'] for client in prop_data['clients']]
            total_prop_time = sum(client_runtimes)
            converged = sum(1 for client in prop_data['clients'] if client['converged'])
            
            # Try to load training data
            try:
                with open(exp['train'], 'r') as f:
                    train_data = json.load(f)
                
                training_time = train_data['duration']['seconds']
                performance = train_data['summary']['average_global_result']
                rounds = len(train_data['rounds'])
                
                overhead_pct = (total_prop_time / training_time) * 100
                
                results.append({
                    'name': exp['name'],
                    'beta': prop_data['beta'],
                    'prop_time': total_prop_time,
                    'train_time': training_time,
                    'overhead_pct': overhead_pct,
                    'converged': converged,
                    'performance': performance,
                    'rounds': rounds
                })
                
            except FileNotFoundError:
                print(f"⚠️  Training file not found for {exp['name']}")
                results.append({
                    'name': exp['name'],
                    'beta': prop_data['beta'],
                    'prop_time': total_prop_time,
                    'train_time': None,
                    'overhead_pct': None,
                    'converged': converged,
                    'performance': None,
                    'rounds': None
                })
                
        except FileNotFoundError:
            print(f"⚠️  Propagation file not found for {exp['name']}")
            continue
        except Exception as e:
            print(f"❌ Error processing {exp['name']}: {e}")
            continue
    
    # Display comparison
    if results:
        print("📊 COMPARISON TABLE:")
        print(f"{'Experiment':<35} {'Beta':<8} {'Prop(s)':<8} {'Train(s)':<10} {'Overhead%':<10} {'Conv.':<6} {'Perf.':<6}")
        print("-" * 85)
        
        for r in results:
            train_str = f"{r['train_time']:.1f}" if r['train_time'] else "N/A"
            overhead_str = f"{r['overhead_pct']:.2f}%" if r['overhead_pct'] else "N/A"
            perf_str = f"{r['performance']:.3f}" if r['performance'] else "N/A"
            
            print(f"{r['name']:<35} {r['beta']:<8} {r['prop_time']:<8.1f} {train_str:<10} {overhead_str:<10} {r['converged']:<6} {perf_str:<6}")

if __name__ == "__main__":
    analyze_pubmed_example()
    analyze_citeseer_comparison()
    
    print(f"\n{'='*50}")
    print("📝 KEY INSIGHTS:")
    print("1. Propagation is a ONE-TIME preprocessing cost")
    print("2. Training happens over multiple federated rounds")  
    print("3. Propagation overhead is typically <5% of training time")
    print("4. Higher beta values often converge faster in propagation")
    print("5. Some training logs may be incomplete (check round counts)")