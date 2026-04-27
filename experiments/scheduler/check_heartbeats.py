#!/usr/bin/env python3
"""Heartbeat checker — runs every 30s.

Detects: completions, OOM crashes, code errors, timeouts.
Handles: retry (reset to pending), blocked (max retries exhausted → Linear).

Protocol per failure type:
  oom        → kill Ray stragglers, reset to pending (retry up to max_retries)
  code_error → reset to pending for retry; if max_retries exhausted → blocked
  timeout    → SIGTERM job, reset to pending for retry; if exhausted → blocked
  unknown    → same as code_error
"""

import json
import os
import signal
import subprocess
from pathlib import Path
from datetime import datetime, timezone

FP_ROOT = Path("/home/bosho/FP")
runs_dir  = FP_ROOT / "experiments/scheduler/runs"
status_path = FP_ROOT / "experiments/scheduler/task_status.json"
queue_path  = FP_ROOT / "experiments/scheduler/queue.json"

TIMEOUT_MULTIPLIER  = 4    # kill if running > 4× estimated_minutes
DEFAULT_TIMEOUT_MIN = 120  # fallback when no estimate


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_job_index() -> dict:
    if not queue_path.exists():
        return {}
    try:
        queue = json.loads(queue_path.read_text())
        return {j["job_id"]: j for j in queue.get("jobs", [])}
    except Exception:
        return {}


def _result_exists(job: dict) -> bool:
    result_dir = FP_ROOT / job.get("result_dir", "")
    if not result_dir.exists():
        return False
    return any(result_dir.glob("results_*.json"))


def _classify_failure(stderr_text: str) -> str:
    if "OutOfMemoryError" in stderr_text or "CUDA out of memory" in stderr_text:
        return "oom"
    if ("Traceback" in stderr_text and
            ("Error" in stderr_text or "Exception" in stderr_text)):
        return "code_error"
    return "unknown"


def _kill_ray_stragglers():
    """Kill orphaned Ray worker and Raylet processes to free GPU memory."""
    for pattern in ["ray::IDLE", "ray::FLClient", "raylet", "gcs_server"]:
        subprocess.run(["pkill", "-9", "-f", pattern],
                       capture_output=True, timeout=5)


# ── Main ─────────────────────────────────────────────────────────────────────

def check_heartbeats():
    if not runs_dir.exists():
        return 0

    status = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            pass

    job_index = _load_job_index()
    now = datetime.now(timezone.utc)
    updated = 0

    for task_dir in runs_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for attempt_dir in task_dir.iterdir():
            hb_file = attempt_dir / "heartbeat.json"
            if not hb_file.exists():
                continue

            try:
                hb = json.loads(hb_file.read_text())
            except json.JSONDecodeError:
                continue

            if hb.get("status") != "running":
                continue

            job_id = attempt_dir.name.replace("attempt_", "")
            job    = job_index.get(job_id, {})
            pid    = hb.get("pid")
            pid_alive = bool(pid and os.path.exists(f"/proc/{pid}"))

            # ── Timeout check ────────────────────────────────────────────────
            if pid_alive and hb.get("started_at"):
                try:
                    started = datetime.fromisoformat(
                        hb["started_at"].replace("Z", "+00:00"))
                    elapsed_min = (now - started).total_seconds() / 60
                    timeout_min = (job.get("estimated_minutes", DEFAULT_TIMEOUT_MIN)
                                   * TIMEOUT_MULTIPLIER)
                    if elapsed_min > timeout_min:
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        hb["status"]        = "failed"
                        hb["failure_reason"] = "timeout"
                        hb["finished_at"]   = now.isoformat()
                        hb_file.write_text(json.dumps(hb))
                        pid_alive = False
                        print(f"[timeout] {job_id}: killed after {elapsed_min:.0f}min "
                              f"(limit {timeout_min:.0f}min)")
                except Exception:
                    pass

            if pid_alive:
                continue  # still running normally

            # ── Process is dead — classify and act ───────────────────────────
            stderr_path = attempt_dir / "stderr.log"
            stderr_text = (stderr_path.read_text(errors="replace")
                           if stderr_path.exists() else "")

            if _result_exists(job):
                # ── Successful completion ────────────────────────────────────
                hb.update(status="done", exit_code=0, finished_at=now.isoformat())
                hb_file.write_text(json.dumps(hb))
                if job_id in status.get("jobs", {}):
                    status["jobs"][job_id].update(
                        execution_status="completed",
                        completed_at=now.isoformat(),
                    )
                    updated += 1
                print(f"[done] {job_id}")

            else:
                # ── Failure ──────────────────────────────────────────────────
                failure_reason = (hb.get("failure_reason")
                                  or _classify_failure(stderr_text))

                hb.update(status="failed", failure_reason=failure_reason,
                           finished_at=now.isoformat())
                hb_file.write_text(json.dumps(hb))

                job_status_entry = status.get("jobs", {}).get(job_id, {})
                retry_count = job_status_entry.get("retry_count", 0)
                max_retries = job.get("max_retries", 2)

                if failure_reason == "oom":
                    print(f"[oom] {job_id}: killing Ray stragglers, freeing GPU")
                    _kill_ray_stragglers()

                if retry_count < max_retries:
                    status["jobs"][job_id] = {
                        **job_status_entry,
                        "execution_status":   "pending",
                        "retry_count":        retry_count + 1,
                        "last_failure_reason": failure_reason,
                    }
                    print(f"[retry {retry_count+1}/{max_retries}] {job_id}: {failure_reason}")
                else:
                    status["jobs"][job_id] = {
                        **job_status_entry,
                        "execution_status": "blocked",
                        "failure_reason":   failure_reason,
                        "retry_count":      retry_count,
                    }
                    print(f"[blocked] {job_id}: {failure_reason} — max retries reached")

                updated += 1

    if updated and status:
        status_path.write_text(json.dumps(status, indent=2))

    return updated


if __name__ == "__main__":
    n = check_heartbeats()
    if n:
        print(f"[{datetime.now(timezone.utc).isoformat()}] {n} jobs updated")
