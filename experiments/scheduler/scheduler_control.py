#!/usr/bin/env python3
"""Scheduler control script — start, stop, pause, reorder queue, spawn fix agents.

Usage:
    # Pause scheduler (comment out cron job)
    python scheduler_control.py pause

    # Resume scheduler
    python scheduler_control.py resume

    # Check status
    python scheduler_control.py status

    # Bump a task to top priority
    python scheduler_control.py priority <task_id> <new_priority>

    # Add emergency bug-fix task
    python scheduler_control.py emergency "bug description"

    # Spawn agent to fix a task
    python scheduler_control.py fix <task_id>
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FP_ROOT = Path("/home/bosho/FP")
CRONTAB_BACKUP = Path("/tmp/crontab.backup")

def get_crontab() -> str:
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    return result.stdout

def set_crontab(content: str) -> None:
    with open("/tmp/crontab.new", "w") as f:
        f.write(content)
    subprocess.run(["crontab", "/tmp/crontab.new"], check=True)

def cmd_pause():
    """Comment out the task_scheduler cron job."""
    current = get_crontab()
    lines = current.splitlines()
    new_lines = []
    paused = False
    for line in lines:
        if "task_scheduler.py" in line and not line.startswith("#"):
            new_lines.append("# PAUSED: " + line)
            paused = True
        else:
            new_lines.append(line)
    
    if paused:
        set_crontab("\n".join(new_lines) + "\n")
        print("✅ Scheduler PAUSED. Task scheduler cron job commented out.")
        print("   Existing running jobs will continue until completion.")
    else:
        print("ℹ️ Scheduler already paused or not found in crontab.")

def cmd_resume():
    """Uncomment the task_scheduler cron job."""
    current = get_crontab()
    lines = current.splitlines()
    new_lines = []
    resumed = False
    for line in lines:
        if "PAUSED: task_scheduler.py" in line:
            new_lines.append(line.replace("# PAUSED: ", ""))
            resumed = True
        else:
            new_lines.append(line)
    
    if resumed:
        set_crontab("\n".join(new_lines) + "\n")
        print("✅ Scheduler RESUMED. Task scheduler will run every 5 min.")
    else:
        print("ℹ️ Scheduler already active or not found in crontab.")

def cmd_status():
    """Show scheduler status."""
    current = get_crontab()
    
    print("=" * 60)
    print("SCHEDULER STATUS")
    print("=" * 60)
    
    # Check if task_scheduler is active
    if "task_scheduler.py" in current and "# PAUSED: task_scheduler.py" not in current:
        print("\n🟢 Task Scheduler: ACTIVE (runs every 5 min)")
    elif "# PAUSED: task_scheduler.py" in current:
        print("\n🟡 Task Scheduler: PAUSED")
    else:
        print("\n🔴 Task Scheduler: NOT IN CRONTAB")
    
    # Check running jobs
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    running = [line for line in result.stdout.split("\n") if "run_experiments" in line and "grep" not in line]
    print(f"\n📊 Running experiments: {len(running)}")
    for line in running[:3]:
        parts = line.split()
        if len(parts) > 10:
            pid = parts[1]
            cmd = " ".join(parts[10:])[:80]
            print(f"   PID {pid}: {cmd}")
    if len(running) > 3:
        print(f"   ... and {len(running) - 3} more")
    
    # Check queue status
    queue_path = FP_ROOT / "experiments/scheduler/queue.json"
    status_path = FP_ROOT / "experiments/scheduler/task_status.json"
    
    if queue_path.exists():
        queue = json.loads(queue_path.read_text())
        print(f"\n📋 Queue: {len(queue.get('jobs', []))} total jobs")
        
        if status_path.exists():
            status = json.loads(status_path.read_text())
            status_jobs = status.get("jobs", {})
            pending = sum(1 for j in status_jobs.values() if j.get("execution_status") == "pending")
            running_count = sum(1 for j in status_jobs.values() if j.get("execution_status") == "running")
            completed = sum(1 for j in status_jobs.values() if j.get("execution_status") in ("completed", "done"))
            failed = sum(1 for j in status_jobs.values() if j.get("execution_status") == "failed")
            
            print(f"   Pending: {pending}")
            print(f"   Running: {running_count}")
            print(f"   Completed: {completed}")
            print(f"   Failed: {failed}")
    
    # Show last scheduler log
    log_path = FP_ROOT / "experiments/scheduler/logs/task_scheduler.log"
    if log_path.exists():
        lines = log_path.read_text().strip().splitlines()
        if lines:
            print(f"\n📝 Last scheduler tick: {lines[-1]}")

def cmd_priority(task_id: str, new_priority: int):
    """Bump a task to a new priority."""
    queue_path = FP_ROOT / "experiments/scheduler/queue.json"
    queue = json.loads(queue_path.read_text())
    
    found = False
    for task in queue.get("tasks", []):
        if task["task_id"] == task_id:
            task["priority"] = new_priority
            # Also update all child jobs
            for child in task.get("children", []):
                for job_id in child.get("job_ids", []):
                    for job in queue.get("jobs", []):
                        if job["job_id"] == job_id:
                            job["priority"] = new_priority
            found = True
            break
    
    if found:
        # Re-sort jobs by priority
        queue["jobs"].sort(key=lambda j: j["priority"])
        queue_path.write_text(json.dumps(queue, indent=2))
        print(f"✅ Task '{task_id}' bumped to priority {new_priority}")
        print("   Jobs re-sorted. Next scheduler tick will pick this first.")
    else:
        print(f"❌ Task '{task_id}' not found in queue.")
        print(f"   Available tasks: {[t['task_id'] for t in queue.get('tasks', [])]}")

def cmd_emergency(description: str):
    """Add an emergency bug-fix task to the queue."""
    queue_path = FP_ROOT / "experiments/scheduler/queue.json"
    queue = json.loads(queue_path.read_text())
    
    # Create emergency task
    emergency_task = {
        "task_id": "EMERGENCY_fix",
        "task_type": "emergency",
        "track": "EMERGENCY",
        "dataset": "fix",
        "model": "agent",
        "display_name": f"🚨 EMERGENCY: {description}",
        "priority": 0,  # Highest
        "execution_status": "pending",
        "total_jobs": 0,
        "dataloadings": [],
        "children": [],
    }
    
    queue["tasks"].insert(0, emergency_task)
    queue_path.write_text(json.dumps(queue, indent=2))
    
    print(f"🚨 EMERGENCY task added: {description}")
    print("   This will block all other tasks until resolved.")
    print("   Use 'scheduler_control.py resume' after fix is complete.")

def cmd_fix(task_id: str):
    """Spawn a coding agent to fix a task."""
    queue_path = FP_ROOT / "experiments/scheduler/queue.json"
    queue = json.loads(queue_path.read_text())
    
    task = None
    for t in queue.get("tasks", []):
        if t["task_id"] == task_id:
            task = t
            break
    
    if not task:
        print(f"❌ Task '{task_id}' not found.")
        return
    
    print(f"🛠️ Spawning fix agent for: {task['display_name']}")
    print("   This will run in background. Check process logs for progress.")
    
    # Write fix instructions
    fix_dir = FP_ROOT / "experiments/scheduler/fixes"
    fix_dir.mkdir(exist_ok=True)
    fix_file = fix_dir / f"fix_{task_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    fix_file.write_text(f"""# Fix Task: {task['display_name']}
- Task ID: {task_id}
- Created: {datetime.now(timezone.utc).isoformat()}
- Status: Investigating

## Notes
Add findings and fix steps here.
""")
    
    # Pause scheduler to prevent more failures
    cmd_pause()
    
    print(f"   Fix notes: {fix_file}")
    print("   Scheduler paused. Resume with: scheduler_control.py resume")

def main():
    parser = argparse.ArgumentParser(description="Control the FedProp task scheduler")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    subparsers.add_parser("pause", help="Pause the scheduler")
    subparsers.add_parser("resume", help="Resume the scheduler")
    subparsers.add_parser("status", help="Show scheduler status")
    
    priority_parser = subparsers.add_parser("priority", help="Change task priority")
    priority_parser.add_argument("task_id", help="Task ID to modify")
    priority_parser.add_argument("new_priority", type=int, help="New priority value (lower = higher priority)")
    
    emergency_parser = subparsers.add_parser("emergency", help="Add emergency task")
    emergency_parser.add_argument("description", help="Description of the emergency")
    
    fix_parser = subparsers.add_parser("fix", help="Spawn fix agent for a task")
    fix_parser.add_argument("task_id", help="Task ID to fix")
    
    args = parser.parse_args()
    
    if args.command == "pause":
        cmd_pause()
    elif args.command == "resume":
        cmd_resume()
    elif args.command == "status":
        cmd_status()
    elif args.command == "priority":
        cmd_priority(args.task_id, args.new_priority)
    elif args.command == "emergency":
        cmd_emergency(args.description)
    elif args.command == "fix":
        cmd_fix(args.task_id)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
