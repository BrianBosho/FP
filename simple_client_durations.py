import json
import pandas as pd
from pathlib import Path


def get_client_durations_dataframe(directory_path: str) -> pd.DataFrame:
    """
    Simple function to extract client durations from propagation stats JSON files.
    
    Args:
        directory_path (str): Path to directory containing JSON files
        
    Returns:
        pd.DataFrame: DataFrame where rows are files, columns are clients, values are runtimes
    """
    directory = Path(directory_path)
    data_dict = {}
    
    # Process each JSON file
    for json_file in directory.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            client_runtimes = {}
            
            # Extract runtime for each client
            if 'clients' in data:
                for client in data['clients']:
                    client_id = client.get('client_id')
                    runtime = client.get('runtime')
                    
                    if client_id is not None and runtime is not None:
                        client_runtimes[f"client_{client_id}"] = runtime
            
            if client_runtimes:
                data_dict[json_file.name] = client_runtimes
                
        except Exception as e:
            print(f"Warning: Could not process {json_file.name}: {e}")
    
    # Convert to DataFrame and sort
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    
    # Sort columns by client number
    if not df.empty:
        client_cols = sorted([col for col in df.columns if col.startswith('client_')], 
                           key=lambda x: int(x.split('_')[1]))
        df = df[client_cols]
    
    return df.sort_index()


# Example usage:
# df = get_client_durations_dataframe("/path/to/propagation_stats")
# print(df.head())