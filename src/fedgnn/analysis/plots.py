"""Plotting utilities for federated learning experiment results."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple


def plot_round_loss(df: pd.DataFrame, title: str = "Loss per Round", 
                     client_col: str = None, ax: plt.Axes = None) -> plt.Figure:
    """Plot loss over rounds for one or more clients."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    if client_col and client_col in df.columns:
        for col in df.columns:
            if col != 'round' and col != 'step':
                ax.plot(df.index, df[col], label=col, alpha=0.7)
    else:
        ax.plot(df.index, df.values, alpha=0.7)
    
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_round_accuracy(df: pd.DataFrame, title: str = "Accuracy per Round",
                        client_col: str = None, ax: plt.Axes = None) -> plt.Figure:
    """Plot accuracy over rounds for one or more clients."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    if client_col and client_col in df.columns:
        for col in df.columns:
            if col != 'round' and col != 'step':
                ax.plot(df.index, df[col], label=col, alpha=0.7)
    else:
        ax.plot(df.index, df.values, alpha=0.7)

    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_training_curves(metrics: dict, title: str = "Training Curves") -> plt.Figure:
    """
    Plot training curves from a metrics dict returned by parse_client_csv.
    
    Args:
        metrics: dict with keys like 'avg_loss_df', 'avg_acc_df', 'epoch_loss_df', 'epoch_acc_df'
        title: plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if 'avg_loss_df' in metrics:
        ax = axes[0]
        for col in metrics['avg_loss_df'].columns:
            ax.plot(metrics['avg_loss_df'].index, metrics['avg_loss_df'][col], label=col)
        ax.set_title(f"{title} — Loss")
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if 'avg_acc_df' in metrics:
        ax = axes[1]
        for col in metrics['avg_acc_df'].columns:
            ax.plot(metrics['avg_acc_df'].index, metrics['avg_acc_df'][col], label=col)
        ax.set_title(f"{title} — Accuracy")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_federated_comparison(results_df: pd.DataFrame, metric_col: str,
                               title: str, groupby_cols: Optional[List[str]] = None,
                               ax: plt.Axes = None) -> plt.Figure:
    """
    Bar chart comparing aggregated results across configs/datasets.
    
    Args:
        results_df: DataFrame with experiment results
        metric_col: column name to compare (e.g., 'test_accuracy')
        title: plot title
        groupby_cols: columns to group by (e.g., ['dataset', 'model'])
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if groupby_cols:
        grouped = results_df.groupby(groupby_cols)[metric_col].mean().sort_values()
    else:
        grouped = results_df[metric_col].sort_values()

    grouped.plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel(metric_col)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    return fig


def plot_energy_dynamics(energy_df: pd.DataFrame, title: str = "Energy Dynamics",
                          per_node: bool = True, ax: plt.Axes = None) -> plt.Figure:
    """Plot Dirichlet energy dynamics over FP iterations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    for col in energy_df.columns:
        ax.plot(energy_df.index, energy_df[col], label=col, alpha=0.7)
    
    ax.set_xlabel("Iteration")
    suffix = "per node" if per_node else "raw"
    ax.set_ylabel(f"Energy ({suffix})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_missing_rate(fp_stats_df: pd.DataFrame, ax: plt.Axes = None) -> plt.Figure:
    """Plot missing rate over clients/rounds."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    if 'client_id' in fp_stats_df.columns and 'missing_rate' in fp_stats_df.columns:
        fp_stats_df.plot(x='client_id', y='missing_rate', kind='bar', ax=ax, alpha=0.8)
    else:
        fp_stats_df['missing_rate'].plot(ax=ax, alpha=0.8)
    
    ax.set_title("Feature Propagation — Missing Rate")
    ax.set_ylabel("Missing Rate")
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    return fig