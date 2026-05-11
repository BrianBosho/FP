import yaml
import copy
import os
import itertools

with open('conf/cora_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Find keys with lists that need to be expanded
list_keys = ['data_loading', 'models', 'num_clients', 'beta', 'use_pe']

# Extract the lists for these keys
lists = []
for k in list_keys:
    if k in config and isinstance(config[k], list):
        lists.append([(k, v) for v in config[k]])
    else:
        lists.append([(k, config.get(k))])

# Generate combinations
combinations = list(itertools.product(*lists))

os.makedirs('conf/cora_split', exist_ok=True)

scripts = []
for i, combo in enumerate(combinations):
    new_config = copy.deepcopy(config)
    name_parts = []
    for k, v in combo:
        # Some values might need to be kept as list if that's what the code expects, 
        # but typically single experiment expects single value or list of 1.
        # If the code expects a list for data_loading or models, we'll make it a list of 1.
        if k in ['data_loading', 'models', 'num_clients', 'beta', 'use_pe']:
            new_config[k] = [v]
        else:
            new_config[k] = v
        name_parts.append(f"{k}_{v}")
    
    # Let's simplify the filename
    model = dict(combo).get('models', 'unknown')
    dl = dict(combo).get('data_loading', 'unknown')
    beta = dict(combo).get('beta', 'unknown')
    
    filename = f"cora_{model}_{dl}_beta{beta}.yaml"
    filepath = os.path.join('conf/cora_split', filename)
    
    with open(filepath, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
        
    scripts.append(f"/home/bosho/.conda/envs/fedgnn/bin/python3 -m src.experiments.run_experiments --config {filepath}")

with open('run_cora_splits.sh', 'w') as f:
    f.write("#!/bin/bash\n")
    for s in scripts:
        f.write(s + "\n")

print(f"Generated {len(scripts)} config files in conf/cora_split/")
print("Created run_cora_splits.sh to run them.")
