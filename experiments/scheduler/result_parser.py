#!/usr/bin/env python3
"""Result parser: read FedProp result dirs -> ResultPacket dict."""

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

FP_ROOT = Path("/home/bosho/FP")


class ResultMissing(FileNotFoundError):
    pass


def _latest_results_json(result_dir: Path) -> Optional[Path]:
    """Find the most recent results_*.json file in the directory."""
    if not result_dir.exists():
        return None

    candidates = [
        p for p in result_dir.iterdir()
        if p.is_file() and p.name.startswith("results_") and p.suffix == ".json"
    ]
    if not candidates:
        return None

    # Prefer timestamp in filename (results_{name}_YYYYMMDD_HHMMSS.json)
    def _extract_ts(p: Path) -> datetime:
        m = re.search(r"_(\d{8}_\d{6})\.json$", p.name)
        if m:
            return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        return datetime.fromtimestamp(p.stat().st_mtime)

    return max(candidates, key=_extract_ts)


def parse(job: dict, attempt_dir: Path) -> dict:
    """Read result files from job['result_dir'] and return a ResultPacket."""
    result_dir = FP_ROOT / job["result_dir"]

    results_file = _latest_results_json(result_dir)
    if results_file is None:
        raise ResultMissing(
            f"results_*.json not found in {result_dir}. "
            f"Contents: {[p.name for p in result_dir.iterdir()] if result_dir.exists() else 'dir missing'}"
        )

    data = json.loads(results_file.read_text())

    # The JSON contains a 'summary' dict with averaged metrics across
    # all repetitions (experiment_seed + repetitions from config).
    summary = data.get("summary", {})
    primary_value = summary.get("average_client_result")

    # Anomaly detection
    anomaly = False
    if primary_value is None:
        anomaly = True
    elif isinstance(primary_value, float) and (math.isnan(primary_value) or math.isinf(primary_value)):
        anomaly = True
    elif isinstance(primary_value, (int, float)) and (primary_value < 0.0 or primary_value > 1.0):
        anomaly = True

    return {
        "job_id": job["job_id"],
        "primary_metric": "average_client_result",
        "primary_value": primary_value,
        "anomalies_detected": anomaly,
        "raw": data,
    }
