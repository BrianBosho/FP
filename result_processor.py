import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_client_results(
    df, 
    dataset_col="dataset", 
    option_col="data_loading_option", 
    avg_col="avg_client_results", 
    std_col="std_client",
    chart_title="Average Client Results by Dataset and Loading Option",
    xlabel="Dataset",
    ylabel="Average Client Results",
    order_options=None,   # if None, sorted unique values will be used
    order_datasets=None,  # if None, sorted unique datasets will be used
    fig_size=(8,6),
    show_plot=True
):
    """
    Groups the input dataframe by dataset and loading option (using the provided column names),
    computes the mean of the average and standard deviation values, and produces:
    
    1) A grouped bar chart with error bars.
    2) A pivot table where each cell is formatted as "avg ± std".

    Parameters:
    - df: The input DataFrame.
    - dataset_col: Column name for dataset.
    - option_col: Column name for loading options.
    - avg_col: Column name for average results.
    - std_col: Column name for standard deviation.
    - chart_title, xlabel, ylabel: Strings for chart labeling.
    - order_options: Optional list to specify ordering of loading options.
    - order_datasets: Optional list to specify ordering of datasets.
    - fig_size: Tuple to specify the size of the matplotlib figure.
    - show_plot: Boolean to decide if plt.show() should be called.

    Returns:
    - result_table: A pivot table DataFrame with formatted "avg ± std" values.
    """

    # Standardize the option column values to strings (e.g., to handle case sensitivity)
    df[option_col] = df[option_col].astype(str)
    
    # Group by dataset and loading option, and compute means for avg and std
    grouped = df.groupby([dataset_col, option_col]).agg({
        avg_col: "mean",
        std_col: "mean"
    }).reset_index()
    
    # Determine ordering if not provided
    if order_options is None:
        order_options = sorted(grouped[option_col].unique())
    if order_datasets is None:
        order_datasets = sorted(grouped[dataset_col].unique())
    
    # Create a new column that formats the result as "avg ± std"
    grouped["result_str"] = grouped.apply(
        lambda row: f"{row[avg_col]:.3f} ± {row[std_col]:.3f}", axis=1
    )
    
    # Create a pivot table for the results table
    result_table = grouped.pivot(index=dataset_col, columns=option_col, values="result_str")
    result_table = result_table.reindex(columns=order_options)
    
    # Prepare the data for plotting: pivot to get numeric values
    pivot_numeric = grouped.pivot(index=dataset_col, columns=option_col)
    avg_data = pivot_numeric[avg_col].reindex(order_datasets).reindex(columns=order_options)
    std_data = pivot_numeric[std_col].reindex(order_datasets).reindex(columns=order_options)
    
    # Set up the bar chart parameters
    x = np.arange(len(order_datasets))  # positions for datasets
    width = 0.8 / len(order_options)  # bar width, adjusted based on number of options
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    for i, option in enumerate(order_options):
        values = avg_data[option]
        errors = std_data[option]
        ax.bar(x + i * width, values, width, yerr=errors, capsize=5, label=option)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(chart_title)
    ax.set_xticks(x + width * (len(order_options)-1) / 2)
    ax.set_xticklabels(order_datasets)
    ax.legend(title=option_col)
    
    plt.tight_layout()
    if show_plot:
        plt.show()
    
    return result_table

import os
import json
import pandas as pd
from glob import glob

def extract_data_from_json(json_file):
    """Extract relevant data from a JSON result file and format it for DataFrame."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract experiment configuration
    config = data['experiment_config']
    
    # Extract summary statistics
    summary = data['summary']
    
    # Build a dictionary with all the data we want to include
    row_data = {
        # Configuration data
        'device': config.get('device', None),
        'data_loading_option': config.get('data_loading_option', None),
        'model_type': config.get('model_type', None),
        'dataset': config.get('dataset', None),
        'num_clients': config.get('num_clients', None),
        'beta': config.get('beta', None),
        'hop': config.get('hop', None),
        
        # Summary statistics
        'avg_global_result': summary.get('average_global_result', None),
        'avg_client_result': summary.get('average_client_result', None),
        'std_global': summary.get('std_global', None),
        'std_client': summary.get('std_client', None),
        
        # Additional metadata
        'experiment_subfolder': os.path.basename(os.path.dirname(json_file)),
        'json_filename': os.path.basename(json_file)
    }
    
    # Extract the last round results
    if data.get('rounds'):
        last_round = data['rounds'][-1]
        row_data['final_round'] = last_round.get('round', None)
        row_data['final_global_result'] = last_round.get('global_result', None)
        row_data['final_client_result'] = last_round.get('client_result', None)
    
    # Parse experiment parameters from the folder name
    folder_parts = row_data['experiment_subfolder'].split('_')
    if len(folder_parts) >= 3:
        # Override dataset and model_type if we can extract from folder name
        # This acts as a cross-check against the JSON data
        for part in folder_parts:
            if part in ['Citeseer', 'Cora', 'Pubmed']:
                row_data['dataset_from_folder'] = part
            if part in ['GAT', 'GCN']:
                row_data['model_type_from_folder'] = part
            if part in ['split_dataset', 'with_feature_prop', 'with_khop']:
                row_data['data_loading_from_folder'] = part
    
    return row_data

def process_results_folder(results_folder):
    """Process a specific results folder and extract data from all JSON files."""
    all_data = []
    
    # Get all experiment subfolders (Citeseer_split_dataset_GAT, etc.)
    experiment_dirs = [d for d in os.listdir(results_folder) 
                      if os.path.isdir(os.path.join(results_folder, d))]
    
    for exp_dir in experiment_dirs:
        exp_path = os.path.join(results_folder, exp_dir)
        
        # Find all JSON result files in this experiment subfolder
        json_files = glob(os.path.join(exp_path, "*results*.json"))
        
        for json_file in json_files:
            try:
                row_data = extract_data_from_json(json_file)
                all_data.append(row_data)
                print(f"Processed: {json_file}")
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    # Convert to DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        return df
    else:
        print("No data found in the results folder.")
        return pd.DataFrame()

def get_results_df(results_folder):
    # Specific results folder to process
    results_folder = "/home/brian_bosho/FP/FP/old1_results/results_20250302_094016"
    
    # Check if the folder exists
    if not os.path.exists(results_folder):
        print(f"Error: The folder '{results_folder}' does not exist.")
        return
    
    # Process the results folder and create DataFrame
    results_df = process_results_folder(results_folder)
    
    if not results_df.empty:
        # Save the DataFrame to CSV
        output_file = f"{os.path.basename(results_folder)}_summary.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"Successfully processed {len(results_df)} result files.")
        print(f"DataFrame shape: {results_df.shape}")
        print(f"Results saved to {output_file}")
        print("First few rows of the DataFrame:")
        print(results_df.head())
    else:
        print("No data was processed.")

    return results_df


def main():
    # Specific results folder to process
    results_folder = "/home/brian_bosho/FP/FP/old1_results/results_20250302_094016"
    
    # Check if the folder exists
    if not os.path.exists(results_folder):
        print(f"Error: The folder '{results_folder}' does not exist.")
        return
    
    # Process the results folder and create DataFrame
    results_df = process_results_folder(results_folder)
    
    if not results_df.empty:
        # Save the DataFrame to CSV
        output_file = f"{os.path.basename(results_folder)}_summary.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"Successfully processed {len(results_df)} result files.")
        print(f"DataFrame shape: {results_df.shape}")
        print(f"Results saved to {output_file}")
        print("First few rows of the DataFrame:")
        print(results_df.head())
    else:
        print("No data was processed.")

if __name__ == "__main__":
    main()