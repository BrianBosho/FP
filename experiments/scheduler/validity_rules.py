#!/usr/bin/env python3
"""Validity rules: define what a valid FedProp result means scientifically."""

import math
from pathlib import Path
from typing import Tuple

FP_ROOT = Path("/home/bosho/FP")


def is_valid(result: dict, job: dict) -> Tuple[bool, str]:
    """Check if a parsed result passes all scientific validity rules."""

    # Rule 1: No anomalies from parser
    if result.get("anomalies_detected"):
        return False, "anomaly detected by parser (NaN/inf/out-of-range)"

    # Rule 2: Primary value exists and is numeric
    val = result.get("primary_value")
    if val is None:
        return False, "primary_value is None — metric not found in result file"
    if not isinstance(val, (int, float)):
        return False, f"primary_value is not numeric: {val!r}"

    # Rule 3: NaN / inf check (redundant but explicit)
    val_f = float(val)
    if math.isnan(val_f) or math.isinf(val_f):
        return False, f"primary_value is {val_f} (NaN or inf)"

    # Rule 4: Within valid range
    lo, hi = job.get("valid_range", [0.0, 1.0])
    if not (lo <= val_f <= hi):
        return False, f"primary_value {val_f:.4f} outside valid_range [{lo}, {hi}]"

    # Rule 5: Above sanity floor
    expected_min = job.get("expected_min", 0.0)
    if val_f < expected_min:
        return False, (
            f"primary_value {val_f:.4f} below expected_min {expected_min} "
            f"for {job.get('dataset')} {job.get('propagation')} beta={job.get('beta')}"
        )

    # Rule 6: Diffusion soft floor (known NaN bug historically)
    if job.get("propagation") == "diffusion" and val_f < 0.1:
        return False, (
            f"diffusion result {val_f:.4f} suspiciously low — "
            "possible unresolved diffusion NaN bug"
        )

    # Rule 7: Result directory actually exists
    result_dir = FP_ROOT / job.get("result_dir", "")
    if not result_dir.exists():
        return False, f"result_dir does not exist: {result_dir}"

    return True, "ok"
