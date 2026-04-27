#!/usr/bin/env python3
"""Update Linear issues with current progress."""

import json
import urllib.request
from datetime import datetime
from pathlib import Path

FP_ROOT = Path("/home/bosho/FP")

# Load API key
with open("/home/bosho/davout/.env") as f:
    api_key = None
    for line in f:
        if line.startswith("LINEAR_API_KEY="):
            api_key = line.split("=", 1)[1].strip()
            break

if not api_key:
    print("No API key found")
    exit(1)

# Load data
queue = json.loads((FP_ROOT / "experiments/scheduler/queue.json").read_text())
status = json.loads((FP_ROOT / "experiments/scheduler/task_status.json").read_text())
linear_state_path = FP_ROOT / "experiments/scheduler/linear_state.json"
linear_state = json.loads(linear_state_path.read_text()) if linear_state_path.exists() else {}

issue_map = {
    'R1_cora_GCN':             'fceea407-f167-4ddc-913b-804694813a6f',   # BOS-37
    'R1_citeseer_GCN':         'bb5b47fb-8fac-4228-9cc0-dc814ed6d19e',   # BOS-38
    'R1_pubmed_GCN':           '678cca7b-70e8-4747-b558-ccb04d636cad',   # BOS-39
    'R1_ogbn_arxiv_GCN_arxiv': 'e84ab5f1-76f1-4783-ae73-caeb9b7063f2',  # BOS-40
    'R1b_cora_GAT':            'a8f8eb13-89d0-4cfc-993a-4d7d027c3283',   # BOS-41
    'R1b_citeseer_GAT':        '0f283228-fc5d-4a4e-9cf4-20c1e8f11bfe',   # BOS-42
    'R1b_pubmed_GAT':          'e68966ed-c5c5-4b3a-9b03-da6764f90d72',   # BOS-43
    'R4_cora_GCN':             'c6587ba7-83be-4793-8c8f-89b0db495af3',   # BOS-44
    'R5_cora_GCN':             'b6354de6-6902-42c4-8a72-9474b6d16027',   # BOS-45
    'R6_texas_GCN':            '94817798-fab2-4be7-b25a-ced9e5eee74d',   # BOS-46
    'R6_wisconsin_GCN':        'fca52611-6bc6-4206-9a10-c2e4a8c86207',   # BOS-47
    'R7_computers_GCN':        'cf91b158-6dc3-4dba-832e-e341147fc435',   # BOS-48
    'R7_photo_GCN':            '34f83e3a-3597-4647-9164-dcde1018c04b',   # BOS-49
}

updated = 0
for task in queue['tasks']:
    task_id = task['task_id']
    issue_id = issue_map.get(task_id)
    if not issue_id:
        continue
    if 'REPLACE_WITH_ACTUAL' in issue_id:
        print(f"Skipping {task_id}: no real Linear issue ID yet")
        continue
    
    completed = sum(1 for c in task['children'] for jid in c['job_ids'] if status['jobs'].get(jid, {}).get('execution_status') in ('completed', 'done'))
    total = task['total_jobs']
    pct = completed / total * 100 if total > 0 else 0
    
    task_state = linear_state.get(task_id, {})
    last_completed = task_state.get('last_completed', -1)
    if last_completed == completed:
        continue
    
    desc = f"""## FedProp Experiment Task

**Track:** {task['track']}
**Dataset:** {task['dataset']}
**Model:** {task['model']}

### Progress
- Jobs: {completed}/{total} completed ({pct:.1f}%)
- Status: {'In Progress' if pct < 100 else 'Done'}

_Last updated: {datetime.now().isoformat()}_"""
    
    req = urllib.request.Request(
        "https://api.linear.app/graphql",
        data=json.dumps({"query": f'mutation {{ issueUpdate(id: "{issue_id}", input: {{ description: {json.dumps(desc)} }}) {{ success }} }}'}).encode(),
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        method="POST"
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        task_state['last_completed'] = completed
        linear_state[task_id] = task_state
        updated += 1
    except Exception as e:
        print(f"Error updating {task_id}: {e}")

linear_state_path.write_text(json.dumps(linear_state, indent=2))
print(f"Updated {updated} Linear issues")
