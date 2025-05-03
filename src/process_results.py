import os
import json
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def process_results_folder_json(main_folder_path):
    """
    Processes experiment results stored in .json files inside a results folder.

    Args:
        main_folder_path (str): Path to the main results folder.

    Returns:
        pandas.DataFrame: Compiled results DataFrame.
    """
    all_results_data = []

    if not os.path.isdir(main_folder_path):
        logging.error(f"Folder not found: {main_folder_path}")
        return pd.DataFrame()

    logging.info(f"Scanning folder: {main_folder_path}")

    for root, dirs, files in os.walk(main_folder_path):
        for file in files:
            if not file.endswith(".json"):
                continue

            file_path = os.path.join(root, file)
            logging.info(f"Reading: {file_path}")

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    logging.warning(f"File is not a valid JSON dict: {file_path}")
                    continue

                config = data.get('experiment_config', {})
                summary = data.get('summary', {})

                if not config or not summary:
                    logging.warning(f"Missing keys in: {file_path}")
                    continue

                row = config.copy()
                row.update(summary)
                row['experiment_id'] = os.path.splitext(file)[0]  # filename without extension

                all_results_data.append(row)

            except Exception as e:
                logging.warning(f"Error reading '{file_path}': {e}")
                continue

    if not all_results_data:
        logging.warning("No valid experiment results found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results_data)

    logging.info(f"Successfully created DataFrame with {len(df)} rows.")
    return df
