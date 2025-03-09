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
    show_plot=True,
    model_type_col="model_type"  # new parameter to control splitting by model type
):
    """
    If the model_type_col exists in the DataFrame, this function will produce separate plots
    and pivot tables for each model type. Otherwise, it works on the whole DataFrame.
    
    Returns:
      - If splitting by model type: a dictionary mapping model type to its result pivot table.
      - Otherwise: the pivot table for the aggregated data.
    """
    # If the model_type column is present, split the data and process each subset
    if model_type_col in df.columns:
        result_tables = {}
        for mtype in sorted(df[model_type_col].unique()):
            sub_df = df[df[model_type_col] == mtype].copy()
            sub_chart_title = f"{chart_title} ({mtype})"
            
            # Standardize the option column values to strings
            sub_df[option_col] = sub_df[option_col].astype(str)
            
            # Group by dataset and loading option, then compute the mean of avg and std
            grouped = sub_df.groupby([dataset_col, option_col]).agg({
                avg_col: "mean",
                std_col: "mean"
            }).reset_index()
            
            # Determine ordering if not provided
            curr_order_options = order_options if order_options is not None else sorted(grouped[option_col].unique())
            curr_order_datasets = order_datasets if order_datasets is not None else sorted(grouped[dataset_col].unique())
            
            # Create a new column that formats the result as "avg ± std"
            grouped["result_str"] = grouped.apply(
                lambda row: f"{row[avg_col]:.3f} ± {row[std_col]:.3f}", axis=1
            )
            
            # Create a pivot table for the results table
            result_table = grouped.pivot(index=dataset_col, columns=option_col, values="result_str")
            result_table = result_table.reindex(columns=curr_order_options)
            
            # Prepare the data for plotting: pivot to get numeric values
            pivot_numeric = grouped.pivot(index=dataset_col, columns=option_col)
            avg_data = pivot_numeric[avg_col].reindex(curr_order_datasets).reindex(columns=curr_order_options)
            std_data = pivot_numeric[std_col].reindex(curr_order_datasets).reindex(columns=curr_order_options)
            
            # Set up the bar chart parameters
            x = np.arange(len(curr_order_datasets))  # positions for datasets
            width = 0.8 / len(curr_order_options)  # bar width, adjusted based on number of options
            
            fig, ax = plt.subplots(figsize=fig_size)
            
            for i, option in enumerate(curr_order_options):
                values = avg_data[option]
                errors = std_data[option]
                ax.bar(x + i * width, values, width, yerr=errors, capsize=5, label=option)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(sub_chart_title)
            ax.set_xticks(x + width * (len(curr_order_options)-1) / 2)
            ax.set_xticklabels(curr_order_datasets)
            ax.legend(title=option_col)
            
            plt.tight_layout()
            if show_plot:
                plt.show()
            
            result_tables[mtype] = result_table
        return result_tables
    else:
        # Process the entire DataFrame if no model_type column is present
        df[option_col] = df[option_col].astype(str)
        grouped = df.groupby([dataset_col, option_col]).agg({
            avg_col: "mean",
            std_col: "mean"
        }).reset_index()
        
        if order_options is None:
            order_options = sorted(grouped[option_col].unique())
        if order_datasets is None:
            order_datasets = sorted(grouped[dataset_col].unique())
        
        grouped["result_str"] = grouped.apply(
            lambda row: f"{row[avg_col]:.3f} ± {row[std_col]:.3f}", axis=1
        )
        result_table = grouped.pivot(index=dataset_col, columns=option_col, values="result_str")
        result_table = result_table.reindex(columns=order_options)
        
        pivot_numeric = grouped.pivot(index=dataset_col, columns=option_col)
        avg_data = pivot_numeric[avg_col].reindex(order_datasets).reindex(columns=order_options)
        std_data = pivot_numeric[std_col].reindex(order_datasets).reindex(columns=order_options)
        
        x = np.arange(len(order_datasets))
        width = 0.8 / len(order_options)
        
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

# --- Sample usage ---
if __name__ == "__main__":
    # Example DataFrame
    data = {
        "dataset": ["Citeseer", "Cora", "Pubmed", "Citeseer", "Cora", "Citeseer", "Pubmed", "Cora", "Cora", "Cora",
                    "Pubmed", "Citeseer", "Pubmed", "Pubmed", "Citeseer", "Citeseer", "Cora", "Pubmed"],
        "data_loading_option": ["LD", "kH", "LD", "kH", "FP", "kH", "FP", "LD", "kH", "LD",
                                "LD", "FP", "kH", "FP", "LD", "FP", "FP", "kH"],
        "avg_client_results": [0.614921, 0.651326, 0.691414, 0.598590, 0.717688, 0.592352, 0.777658, 0.632224,
                               0.656579, 0.589714, 0.749047, 0.506083, 0.623059, 0.775815, 0.551824, 0.513958,
                               0.723095, 0.603874],
        "std_client": [0.003430, 0.005179, 0.043085, 0.003607, 0.003577, 0.005954, 0.005778, 0.002890,
                       0.006303, 0.005199, 0.003819, 0.012014, 0.012287, 0.011504, 0.001556, 0.004552,
                       0.006471, 0.100978],
        "model_type": ["GCN", "GCN", "GAT", "GCN", "GCN", "GAT", "GCN", "GCN", "GAT", "GAT",
                       "GCN", "GAT", "GCN", "GAT", "GAT", "GCN", "GAT", "GAT"]
    }
    sample_df = pd.DataFrame(data)
    
    # Call the function. Because the DataFrame includes "model_type", this will produce separate plots
    # and tables for each model type.
    results_tables = plot_client_results(sample_df)
    
    # Print the resulting pivot tables for each model type
    for model, table in results_tables.items():
        print(f"Results Table for model type '{model}':")
        print(table)
        print("\n")
