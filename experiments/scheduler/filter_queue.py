#!/usr/bin/env python3
"""Filter queue to only include jobs that don't have existing results.

Hybrid approach: Keep old results, only run what's missing.
"""

import json
from pathlib import Path
from collections import defaultdict

FP_ROOT = Path("/home/bosho/FP")
RESULTS_BASE = FP_ROOT / "experiments" / "results"
QUEUE_PATH = FP_ROOT / "experiments" / "scheduler" / "queue.json"
FILTERED_QUEUE_PATH = FP_ROOT / "experiments" / "scheduler" / "queue_filtered.json"


def parse_result_dir_name(dir_name: str) -> dict:
    """Parse result directory name to extract parameters."""
    result = {}
    
    # Known propagation names that might contain underscores
    PROPAGATIONS = ['zero_hop', 'adjacency', 'diffusion', 'full']
    
    # Find which propagation is in the name
    propagation = None
    for prop in PROPAGATIONS:
        if f'_{prop}_' in dir_name:
            propagation = prop
            break
    
    if not propagation:
        return result
    
    result['propagation'] = propagation
    
    # Split by propagation to get dataset and model parts
    parts = dir_name.split(f'_{propagation}_')
    if len(parts) != 2:
        return result
    
    dataset_part = parts[0]
    rest = parts[1]
    
    # dataset_part is just the dataset name (might have underscores for ogbn-arxiv)
    result['dataset'] = dataset_part
    
    # rest is: Model_beta{beta}_clients{n}
    rest_parts = rest.split('_')
    if len(rest_parts) >= 1:
        result['model'] = rest_parts[0]
    
    # Find beta and clients
    for i, part in enumerate(rest_parts):
        if part.startswith('beta'):
            beta_str = part[4:]
            if '.0' in beta_str:
                beta_str = beta_str.replace('.0', '')
            result['beta'] = int(beta_str) if beta_str.isdigit() else beta_str
        elif part == 'clients' and i + 1 < len(rest_parts):
            result['n_clients'] = int(rest_parts[i + 1])
        elif part.startswith('clients'):
            # Handle "clients10" format (no underscore)
            client_str = part[7:]  # Remove 'clients' prefix
            if client_str.isdigit():
                result['n_clients'] = int(client_str)
    
    return result


def has_existing_result(track: str, dataset: str, model: str, 
                        propagation: str, beta: int, n_clients: int) -> bool:
    """Check if result directory exists for this config."""
    track_dir = RESULTS_BASE / track
    if not track_dir.exists():
        return False
    
    patterns = [
        f"{dataset}_{propagation}_{model}_beta{beta}_clients{n_clients}",
        f"{dataset}_{propagation}_{model}_beta{beta}.0_clients{n_clients}",
    ]
    
    for d in track_dir.iterdir():
        if d.is_dir() and d.name not in ('propagation_stats',):
            has_results = bool(list(d.glob('results_*.json')))
            if has_results:
                for pattern in patterns:
                    if pattern in d.name:
                        return True
    
    return False


def filter_queue():
    print("Loading queue...")
    queue = json.loads(QUEUE_PATH.read_text())
    
    print("Scanning existing results...")
    existing_configs = set()
    
    for track_dir in RESULTS_BASE.iterdir():
        if not track_dir.is_dir():
            continue
        track = track_dir.name
        for d in track_dir.iterdir():
            if d.is_dir() and d.name not in ('propagation_stats',):
                has_results = bool(list(d.glob('results_*.json')))
                if has_results:
                    parsed = parse_result_dir_name(d.name)
                    if all(k in parsed for k in ('dataset', 'model', 'propagation', 'beta', 'n_clients')):
                        key = (track, parsed['dataset'], parsed['model'], 
                               parsed['propagation'], parsed['beta'], parsed['n_clients'])
                        existing_configs.add(key)
    
    print(f"Found {len(existing_configs)} existing config combinations with results")
    
    # Filter jobs
    filtered_jobs = []
    skipped_jobs = []
    
    for job in queue['jobs']:
        key = (job['track'], job['dataset'], job['model'], 
               job['propagation'], job['beta'], job['n_clients'])
        
        if key in existing_configs:
            skipped_jobs.append(job)
        else:
            filtered_jobs.append(job)
    
    print(f"\nJobs to run: {len(filtered_jobs)}")
    print(f"Jobs skipped (already have results): {len(skipped_jobs)}")
    
    # Update tasks
    filtered_job_ids = {j['job_id'] for j in filtered_jobs}
    filtered_tasks = []
    
    for task in queue['tasks']:
        new_children = []
        task_total = 0
        
        for child in task['children']:
            kept_job_ids = [jid for jid in child['job_ids'] if jid in filtered_job_ids]
            if kept_job_ids:
                new_children.append({
                    **child,
                    'job_ids': kept_job_ids,
                    'job_count': len(kept_job_ids),
                })
                task_total += len(kept_job_ids)
        
        if new_children:  # Only keep tasks that have remaining jobs
            filtered_tasks.append({
                **task,
                'children': new_children,
                'total_jobs': task_total,
                'dataloadings': [c['propagation'] for c in new_children],
            })
    
    # Build filtered queue
    filtered_queue = {
        **queue,
        'jobs': filtered_jobs,
        'tasks': filtered_tasks,
        'filtered_from': len(queue['jobs']),
        'filtered_count': len(filtered_jobs),
        'skipped_count': len(skipped_jobs),
    }
    
    # Save
    FILTERED_QUEUE_PATH.write_text(json.dumps(filtered_queue, indent=2))
    
    # Report
    print(f"\n{'='*60}")
    print(f"FILTERED QUEUE SUMMARY")
    print(f"{'='*60}")
    print(f"Original jobs: {len(queue['jobs'])}")
    print(f"Jobs to run: {len(filtered_jobs)}")
    print(f"Jobs skipped: {len(skipped_jobs)}")
    print(f"Tasks with remaining jobs: {len(filtered_tasks)}")
    
    print(f"\nRemaining tasks:")
    for task in filtered_tasks:
        print(f"  {task['display_name']}: {task['total_jobs']} jobs")
        for child in task['children']:
            print(f"    └─ {child['display_name']}: {child['job_count']} jobs")
    
    print(f"\nSkipped configs (have old results):")
    for track, dataset, model, prop, beta, n_clients in sorted(existing_configs):
        print(f"  {track}/{dataset}/{model}/{prop}/beta={beta}/clients={n_clients}")
    
    print(f"\nFiltered queue saved to: {FILTERED_QUEUE_PATH}")
    
    return filtered_queue


if __name__ == '__main__':
    filter_queue()
