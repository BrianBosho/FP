import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def get_client_durations_dataframe(directory_path: str) -> pd.DataFrame:
    """
    Takes a directory path containing propagation stats JSON files and returns a DataFrame
    where each column represents a client and each row represents a different file.
    The values are the runtime durations for each client.
    
    Args:
        directory_path (str): Path to directory containing JSON files with propagation stats
        
    Returns:
        pd.DataFrame: DataFrame with files as rows and clients as columns, values are runtimes
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    # Dictionary to store data: {file_name: {client_id: runtime}}
    data_dict = {}
    
    # Get all JSON files in the directory
    json_files = list(directory.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in directory: {directory_path}")
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            file_name = json_file.name
            client_runtimes = {}
            
            # Extract runtime for each client
            if 'clients' in data:
                for client in data['clients']:
                    client_id = client.get('client_id')
                    runtime = client.get('runtime')
                    
                    if client_id is not None and runtime is not None:
                        client_runtimes[f"client_{client_id}"] = runtime
            
            if client_runtimes:
                data_dict[file_name] = client_runtimes
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not process file {json_file.name}: {e}")
            continue
    
    if not data_dict:
        raise ValueError("No valid data found in any JSON files")
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    
    # Sort columns by client number
    if not df.empty:
        # Extract client numbers and sort columns
        client_cols = [col for col in df.columns if col.startswith('client_')]
        client_nums = [(col, int(col.split('_')[1])) for col in client_cols]
        client_nums.sort(key=lambda x: x[1])
        sorted_cols = [col for col, _ in client_nums]
        
        # Reorder columns
        df = df[sorted_cols]
    
    # Sort rows by filename
    df = df.sort_index()
    
    return df


def analyze_client_durations(directory_path: str, save_to_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze client durations and optionally save to CSV.
    
    Args:
        directory_path (str): Path to directory containing JSON files
        save_to_csv (str, optional): Path to save CSV file
        
    Returns:
        pd.DataFrame: DataFrame with client durations
    """
    df = get_client_durations_dataframe(directory_path)
    
    if save_to_csv:
        df.to_csv(save_to_csv)
        print(f"DataFrame saved to: {save_to_csv}")
    
    # Print summary statistics
    print(f"DataFrame shape: {df.shape}")
    print(f"Files processed: {len(df)}")
    print(f"Clients found: {len(df.columns)}")
    print("\nSummary statistics:")
    print(df.describe())
    
    return df


# Example usage
if __name__ == "__main__":
    # Example path
    example_path = "/home/brian_bosho/FP/FP/federated-gnn/results/pubmed_gcn_final_beta_10_with_pe2/propagation_stats"
    
    try:
        df = analyze_client_durations(example_path)
        print("\nFirst few rows:")
        print(df.head())
        
        # Optionally save to CSV
        # df.to_csv("client_durations.csv")
        
    except Exception as e:
        print(f"Error: {e}")