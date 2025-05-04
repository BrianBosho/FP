import pandas as pd
import ast
import json
def parse_client_csv(file_path):
    """
    Parses a federated training CSV with embedded epoch logs into structured DataFrames.

    Returns a dictionary of:
        - loss_df, acc_df: MultiIndex (round, epoch), columns = client IDs
        - loss_df_step, acc_df_step: step-based DataFrames (for plotting over time)
        - avg_loss_df, avg_acc_df: average per round
        - epoch_loss_df, epoch_acc_df: average per epoch
    """
    df_raw = pd.read_csv(file_path)
    records = []

    # Parse and flatten 'epochs_data'
    for _, row in df_raw.iterrows():
        client_id = row['client_id']
        if not row['epochs_data'] or row['epochs_data'] == '[]':
            continue
        data = ast.literal_eval(row['epochs_data'])
        for entry in data:
            entry['client_id'] = client_id
            records.append(entry)

    flat_df = pd.DataFrame(records)
    flat_df = flat_df.sort_values(by=['round', 'epoch', 'client_id'])

    # Build main loss and accuracy DataFrames
    loss_df = flat_df.pivot(index=['round', 'epoch'], columns='client_id', values='loss')
    acc_df = flat_df.pivot(index=['round', 'epoch'], columns='client_id', values='accuracy')

    # Add step column for plotting over training progression
    loss_df_step = loss_df.reset_index().copy()
    acc_df_step = acc_df.reset_index().copy()
    loss_df_step['step'] = range(len(loss_df_step))
    acc_df_step['step'] = range(len(acc_df_step))
    # make the step column the index
    loss_df_step = loss_df_step.set_index('step')
    acc_df_step = acc_df_step.set_index('step')
    loss_df_step = loss_df_step.drop(columns=['round', 'epoch'])
    acc_df_step = acc_df_step.drop(columns=['round', 'epoch'])

    # Round-wise averages
    avg_loss_df = loss_df.groupby('round').mean()
    avg_acc_df = acc_df.groupby('round').mean()

    # Epoch-wise averages
    epoch_loss_df = loss_df.groupby('epoch').mean()
    epoch_acc_df = acc_df.groupby('epoch').mean()

    # Final (last-epoch) metrics per round
    last_epoch_df = flat_df[flat_df['epoch'] == flat_df['epoch'].max()]

    final_loss_df = last_epoch_df.pivot(index='round', columns='client_id', values='loss')
    final_acc_df = last_epoch_df.pivot(index='round', columns='client_id', values='accuracy')

    return {
        'loss_df': loss_df,
        'acc_df': acc_df,
        'loss_df_step': loss_df_step,
        'acc_df_step': acc_df_step,
        'avg_loss_df': avg_loss_df,
        'avg_acc_df': avg_acc_df,
        'epoch_loss_df': epoch_loss_df,
        'epoch_acc_df': epoch_acc_df,
        'final_loss_df': final_loss_df,     # <--- NEW
        'final_acc_df': final_acc_df 
    }



def process_fp_logs(filepath):
    # Load JSON from file
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    clients = data.get("clients", [])
    
    summary_data = []
    deltas_data = {}
    energy_raw_data = {}
    energy_per_node_data = {}
    energy_per_edge_data = {}

    for client in clients:
        client_id = client["client_id"]

        # Deltas
        deltas = client.get("deltas", [])
        deltas_data[client_id] = deltas

        # Energies
        energies = client.get("energies", [])
        energy_raw_data[client_id] = [e["raw"] for e in energies]
        energy_per_node_data[client_id] = [e["per_node"] for e in energies]
        energy_per_edge_data[client_id] = [e["per_edge"] for e in energies]

        # Summary (excluding deltas and energies)
        summary_entry = {k: v for k, v in client.items() if k not in ["deltas", "energies"]}
        summary_data.append(summary_entry)
    
    # Create DataFrames
    fp_stats_df = pd.DataFrame(summary_data)
    # find missing rate by dividign num_missing by num_nodes
    fp_stats_df["missing_rate"] = fp_stats_df["nodes_unknown"] / fp_stats_df["nodes_total"]
        
    deltas_df = pd.DataFrame.from_dict(deltas_data, orient='columns')
    deltas_df.index.name = 'iteration'

    energy_raw_df = pd.DataFrame.from_dict(energy_raw_data, orient='columns')
    energy_raw_df.index.name = 'iteration'

    energy_per_node_df = pd.DataFrame.from_dict(energy_per_node_data, orient='columns')
    energy_per_node_df.index.name = 'iteration'

    energy_per_edge_df = pd.DataFrame.from_dict(energy_per_edge_data, orient='columns')
    energy_per_edge_df.index.name = 'iteration'

    # Return as dictionary of DataFrames
    return {
        "fp_stats_df": fp_stats_df,
        "deltas_df": deltas_df,
        "energy_raw_df": energy_raw_df,
        "energy_per_node_df": energy_per_node_df,
        "energy_per_edge_df": energy_per_edge_df
    }
