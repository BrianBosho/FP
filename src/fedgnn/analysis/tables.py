"""Table formatting and export utilities for federated learning results."""

import pandas as pd
from typing import List, Optional


def format_results_table(df: pd.DataFrame, metric_cols: List[str],
                          groupby: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Format a results DataFrame for publication-ready table output.

    Args:
        df: experiment results DataFrame
        metric_cols: list of metric columns to include (e.g., ['test_accuracy', 'test_loss'])
        groupby: optional grouping columns (e.g., ['dataset', 'model'])

    Returns:
        Formatted DataFrame with mean ± std across runs
    """
    agg_dict = {}
    for col in metric_cols:
        agg_dict[col] = ['mean', 'std']

    if groupby:
        grouped = df.groupby(groupby)[metric_cols].agg(agg_dict)
    else:
        grouped = df[metric_cols].agg(agg_dict)

    # Flatten column multi-index
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    # Format as "mean ± std" strings
    formatted = grouped.copy()
    for col in formatted.columns:
        mean_col = col.replace('_std', '_mean').replace('_std', '')
        std_col = col.replace('_mean', '_std')
        if f"{col.replace('_std','_mean')}" in formatted.columns and col.endswith('_std'):
            pass  # handled below
        if col.endswith('_mean'):
            base = col.replace('_mean', '')
            std_col_name = base + '_std'
            if std_col_name in formatted.columns:
                formatted[col] = formatted.apply(
                    lambda row: f"{row[col]:.4f} ± {row[std_col_name]:.4f}" 
                    if pd.notna(row[col]) and pd.notna(row[std_col_name]) else "—",
                    axis=1
                )
        elif col.endswith('_std'):
            pass  # skip std columns in final output

    # Drop std columns
    formatted = formatted[[c for c in formatted.columns if not c.endswith('_std')]]
    return formatted


def summary_table(df: pd.DataFrame, rounds: Optional[int] = None) -> pd.DataFrame:
    """
    Build a summary table showing best / final metrics per experiment config.
    
    Args:
        df: results DataFrame
        rounds: if set, only consider results up to this round
    """
    subset = df[df['round'] <= rounds] if rounds and 'round' in df.columns else df

    summary = subset.groupby('experiment_config').agg(
        best_test_accuracy=('test_accuracy', 'max'),
        final_test_accuracy=('test_accuracy', 'last'),
        best_round=('test_accuracy', 'idxmax'),
        num_rounds=('round', 'max'),
    ).round(4)
    
    return summary


def export_table(df: pd.DataFrame, path: str, **kwargs) -> None:
    """
    Export a DataFrame to CSV or markdown.

    Args:
        df: DataFrame to export
        path: output file path (.csv or .md)
        **kwargs: passed to to_csv or to_markdown
    """
    if path.endswith('.csv'):
        df.to_csv(path, **kwargs)
    elif path.endswith('.md'):
        df.to_markdown(path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported export format: {path}")