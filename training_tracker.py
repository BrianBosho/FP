import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd

class TrainingTracker:
    def __init__(self, experiment_name=None, save_dir='./training_logs'):
        """
        Initialize a training tracker to monitor convergence across rounds.
        
        Args:
            experiment_name (str): Name for this experiment run
            save_dir (str): Directory to save logs and visualizations
        """
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # History containers
        self.round_history = []
        self.global_metrics = {
            'round': [],
            'test_accuracy': [],
            'avg_client_accuracy': []
        }
        
        # Client metrics
        self.client_metrics = {}
        
        # CSV log file
        self.log_file = os.path.join(self.save_dir, f"{self.experiment_name}_log.csv")
        self.client_log_file = os.path.join(self.save_dir, f"{self.experiment_name}_client_log.csv")
        
    def log_round(self, round_num, round_metrics):
        """Log metrics for a training round"""
        self.round_history.append({
            'round': round_num,
            **round_metrics
        })
        
        # Save updated logs
        self._save_logs()
        
    def log_global_metrics(self, round_num, test_accuracy, avg_client_accuracy):
        """Log global metrics after a round"""
        self.global_metrics['round'].append(round_num)
        self.global_metrics['test_accuracy'].append(test_accuracy)
        self.global_metrics['avg_client_accuracy'].append(avg_client_accuracy)
        
        # Save to CSV
        pd.DataFrame(self.global_metrics).to_csv(self.log_file, index=False)
        
    def log_client_metrics(self, round_num, client_id, train_loss, train_acc, val_loss=None, val_acc=None):
        """Log metrics for an individual client"""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = {
                'round': [], 
                'train_loss': [], 
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
        
        self.client_metrics[client_id]['round'].append(round_num)
        self.client_metrics[client_id]['train_loss'].append(train_loss)
        self.client_metrics[client_id]['train_acc'].append(train_acc)
        
        if val_loss is not None:
            self.client_metrics[client_id]['val_loss'].append(val_loss)
        if val_acc is not None:
            self.client_metrics[client_id]['val_acc'].append(val_acc)
            
        # Save client logs
        self._save_client_logs()
            
    def _save_logs(self):
        """Save all logs to disk"""
        df = pd.DataFrame(self.round_history)
        df.to_csv(self.log_file, index=False)
        
    def _save_client_logs(self):
        """Save client logs to disk"""
        all_client_data = []
        for client_id, metrics in self.client_metrics.items():
            for i in range(len(metrics['round'])):
                row = {'client_id': client_id}
                for key in metrics:
                    if i < len(metrics[key]):
                        row[key] = metrics[key][i]
                all_client_data.append(row)
                
        pd.DataFrame(all_client_data).to_csv(self.client_log_file, index=False)
        
    def plot_convergence(self, metric='train_acc', save=True):
        """Plot convergence of specified metric across rounds"""
        plt.figure(figsize=(10, 6))
        
        # Plot client metrics
        for client_id, metrics in self.client_metrics.items():
            if metric in metrics and len(metrics[metric]) > 0:
                plt.plot(metrics['round'], metrics[metric], 
                         alpha=0.3, label=f'Client {client_id}')
        
        # Plot global metrics if applicable
        if metric == 'test_accuracy' and len(self.global_metrics['round']) > 0:
            plt.plot(self.global_metrics['round'], self.global_metrics['test_accuracy'], 
                     'k-', linewidth=2, label='Global Model')
        
        plt.title(f'Convergence of {metric.replace("_", " ").title()}')
        plt.xlabel('Round')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save:
            plt.savefig(os.path.join(self.save_dir, f"{metric}_convergence.png"), dpi=300)
            
        plt.close()
        
    def plot_all_metrics(self):
        """Plot all relevant metrics to visualize convergence"""
        self.plot_convergence(metric='train_loss')
        self.plot_convergence(metric='train_acc')
        self.plot_convergence(metric='val_loss')
        self.plot_convergence(metric='val_acc')
        self.plot_convergence(metric='test_accuracy')
        
        # Plot global vs client average
        if len(self.global_metrics['round']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.global_metrics['round'], self.global_metrics['test_accuracy'], 
                    'b-', linewidth=2, label='Global Model')
            plt.plot(self.global_metrics['round'], self.global_metrics['avg_client_accuracy'], 
                    'r--', linewidth=2, label='Avg Client')
            plt.title('Global vs Average Client Accuracy')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(self.save_dir, f"global_vs_client_accuracy.png"), dpi=300)
            plt.close()
