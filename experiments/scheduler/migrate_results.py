#!/usr/bin/env python3
"""Migrate old experiment results to new task_status.json format.

Scans result directories, matches to new job IDs, marks as completed.
"""

import json
from pathlib import Path
from collections import defaultdict

FP_ROOT = Path("/home/bosho/FP")
RESULTS_BASE = FP_ROOT / "experiments" / "results"
QUEUE_PATH = FP_ROOT / "experiments" / "scheduler" / "queue.json"
STATUS_PATH = FP_ROOT / "experiments" / "scheduler" / "task_status.json"


def parse_result_dir_name(dir_name: str) -> dict:
    """Parse result directory name to extract parameters."""
    # Format: {Dataset}_{propagation}_{Model}_beta{beta}_clients{n}
    # or: {Dataset}_{propagation}_{Model}_beta{beta}.0_clients{n}
    parts = dir_name.split('_')
    
    result = {}
    
    # Find 'beta' and 'clients' positions
    for i, part in enumerate(parts):
        if part.startswith('beta'):
            beta_str = part[4:]  # Remove 'beta' prefix
            # Handle beta1.0 vs beta1
            if '.0' in beta_str:
                beta_str = beta_str.replace('.0', '')
            result['beta'] = int(beta_str) if beta_str.isdigit() else beta_str
        elif part == 'clients' and i + 1 < len(parts):
            result['n_clients'] = int(parts[i + 1])
    
    # Extract dataset, propagation, model
    # These are the parts before 'beta'
    if 'beta' in dir_name:
        beta_idx = dir_name.index('_beta')
        prefix = dir_name[:beta_idx]
        prefix_parts = prefix.split('_')
        
        # Try to identify: dataset_propagation_model
        # e.g., Cora_zero_hop_GCN or Citeseer_full_GCN
        if len(prefix_parts) >= 3:
            result['model'] = prefix_parts[-1]
            result['propagation'] = prefix_parts[-2]
            result['dataset'] = '_'.join(prefix_parts[:-2])
    
    return result


def match_job_to_result(job: dict, result_dirs: dict) -> bool:
    """Check if a job has corresponding result directory."""
    track = job['track']
    dataset = job['dataset']
    model = job['model']
    propagation = job['propagation']
    beta = job['beta']
    n_clients = job['n_clients']
    
    track_dirs = result_dirs.get(track, [])
    
    for dir_name in track_dirs:
        parsed = parse_result_dir_name(dir_name)
        
        if (parsed.get('dataset') == dataset and
            parsed.get('model') == model and
            parsed.get('propagation') == propagation and
            parsed.get('beta') == beta and
            parsed.get('n_clients') == n_clients):
            return True
    
    return False


def migrate():
    print("Loading queue...")
    queue = json.loads(QUEUE_PATH.read_text())
    
    # Index all result directories
    print("Scanning result directories...")
    result_dirs = defaultdict(list)
    result_dir_details = {}
    
    for track_dir in RESULTS_BASE.iterdir():
        if not track_dir.is_dir():
            continue
        track = track_dir.name
        for d in track_dir.iterdir():
            if d.is_dir() and d.name not in ('propagation_stats',):
                has_results = bool(list(d.glob('results_*.json')))
                if has_results:
                    result_dirs[track].append(d.name)
                    result_dir_details[d.name] = {
                        'path': str(d),
                        'files': len(list(d.glob('*'))),
                    }
    
    print(f"Found {sum(len(v) for v in result_dirs.values())} result directories with results")
    
    # Build status
    status = {"jobs": {}}
    total_completed = 0
    total_jobs = len(queue['jobs'])
    
    print(f"\nMatching {total_jobs} jobs to results...")
    
    for job in queue['jobs']:
        job_id = job['job_id']
        found = match_job_to_result(job, result_dirs)
        
        status['jobs'][job_id] = {
            'job_id': job_id,
            'execution_status': 'completed' if found else 'pending',
            'result_status': 'valid' if found else 'unassessed',
        }
        
        if found:
            total_completed += 1
    
    # Save status
    STATUS_PATH.write_text(json.dumps(status, indent=2))
    
    # Report
    print(f"\n{'='*60}")
    print(f"MIGRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total jobs: {total_jobs}")
    print(f"Completed: {total_completed}")
    print(f"Pending: {total_jobs - total_completed}")
    print(f"Percentage: {total_completed/total_jobs*100:.1f}%")
    
    # Per-task breakdown
    print(f"\nPer-task completion:")
    for task in queue['tasks']:
        completed = 0
        for child in task['children']:
            for job_id in child['job_ids']:
                if status['jobs'].get(job_id, {}).get('execution_status') == 'completed':
                    completed += 1
        
        pct = completed / task['total_jobs'] * 100 if task['total_jobs'] > 0 else 0
        print(f"  {task['display_name']}: {completed}/{task['total_jobs']} ({pct:.1f}%)")
    
    print(f"\nStatus saved to: {STATUS_PATH}")
    
    return status


if __name__ == '__main__':
    migrate()
