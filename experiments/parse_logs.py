#!/usr/bin/env python3
"""Extract structured snippets from raw FP experiment logs (partial-run friendly).

Parses:
  - FP CSV FORMAT RESULT blocks
  - Experiment configuration echoes (best-effort)
  - Traceback / exception lines for failure typing

Usage:
  python experiments/parse_logs.py path/to/run.log
  python experiments/parse_logs.py runs/**/*.log
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path


FP_CSV_BLOCK = re.compile(
    r"FP CSV FORMAT RESULT:\n(?P<header>[^\n]+)\n(?P<row>[^\n]+)",
    re.MULTILINE,
)


def classify_failure(text: str) -> str:
    t = text.lower()
    if "outofmemoryerror" in t or "cuda out of memory" in t:
        return "oom_gpu"
    if "runtimeerror: cuda" in t or "cuda driver" in t:
        return "cuda_driver_error"
    if "raylet died" in t or "actor died" in t or "ray::" in t:
        return "ray_actor_died"
    if "object store" in t and "full" in t:
        return "ray_object_store_full"
    if "sigterm" in t or "terminated" in t:
        return "sigterm"
    if "nan" in t and "loss" in t:
        return "nan_metric"
    if "timeout" in t:
        return "timeout"
    return "unknown_exception"


def parse_log_text(content: str, source: str) -> dict:
    out: dict = {"source": source, "fp_csv_rows": [], "failures": []}
    for m in FP_CSV_BLOCK.finditer(content):
        out["fp_csv_rows"].append(
            {"header": m.group("header").strip(), "row": m.group("row").strip()}
        )
    if "Traceback (most recent call last):" in content:
        tail = content.split("Traceback (most recent call last):")[-1].strip()
        out["failures"].append(
            {
                "kind": classify_failure(tail),
                "snippet": tail[-4000:],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse FP experiment logs for CSV blocks and failures.")
    parser.add_argument("paths", nargs="+", help="Log files or glob patterns")
    parser.add_argument("--json", action="store_true", help="Emit JSON lines to stdout")
    args = parser.parse_args()

    expanded: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if "*" in p or "?" in p or "[" in p:
            expanded.extend(Path(x) for x in glob.glob(p, recursive=True))
        elif path.is_file():
            expanded.append(path)

    if not expanded:
        print("No files matched.", file=sys.stderr)
        sys.exit(1)

    for fp in expanded:
        text = fp.read_text(errors="replace")
        record = parse_log_text(text, str(fp))
        if args.json:
            print(json.dumps(record))
        else:
            print(f"=== {fp} ===")
            print(f"CSV blocks found: {len(record['fp_csv_rows'])}")
            for row in record["fp_csv_rows"]:
                print(" ", row["row"][:200], ("..." if len(row["row"]) > 200 else ""))
            for f in record["failures"]:
                print(f" failure[{f['kind']}]: {f['snippet'][:300]}...")


if __name__ == "__main__":
    main()
