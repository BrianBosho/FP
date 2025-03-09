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
    # results_folder = "/home/brian_bosho/FP/FP/old1_results/results_20250302_094016"
    
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
        result_dir = os.path.dirname(results_folder)
        output_file = f"{os.path.basename(results_folder)}_summary.csv"
        save_dir = os.path.join(result_dir, output_file)
        results_df.to_csv(save_dir, index=False)
        
        print(f"Successfully processed {len(results_df)} result files.")
        print(f"DataFrame shape: {results_df.shape}")
        print(f"Results saved to {output_file}")
        print("First few rows of the DataFrame:")
        print(results_df.head())
    else:
        print("No data was processed.")

if __name__ == "__main__":
    main()