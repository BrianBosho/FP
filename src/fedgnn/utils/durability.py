"""Streaming JSONL writers for partial-result durability.

Protects long runs against SIGTERM / OOM before final JSON is written
by flushing per-round, per-repetition, telemetry, and event records
as they are produced.

Usage::

    bundle = DurabilityBundle(Path("runs/my_exp"))
    bundle.event({"kind": "start"})
    # inside round loop:
    bundle.round({"fl_round": i, "avg_val_acc": ...})
    # after each rep:
    bundle.repetition({"rep": 1, "global_result": ...})
    # at end:
    bundle.telemetry({"phases_sec": ...})
    bundle.close()
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def _to_json_line(record: Any) -> str:
    return json.dumps(record, separators=(",", ":"), default=str)


class JsonlWriter:
    """Thread-safe, append-only JSONL file sink. Opens on first write."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._fh = None

    def _open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered so each flush() is a real OS write
        self._fh = open(self._path, "a", buffering=1)  # noqa: WPS515

    def append(self, record: dict[str, Any]) -> None:
        line = _to_json_line(record)
        with self._lock:
            if self._fh is None:
                self._open()
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class DurabilityBundle:
    """Four JSONL streams for one experiment run directory.

    Files written:
      per_round.jsonl       — one record per FL round
      per_repetition.jsonl  — one record per completed repetition
      telemetry.jsonl       — telemetry snapshots (written at end / on demand)
      events.jsonl          — lifecycle events (start, fail, complete, etc.)
    """

    def __init__(self, run_dir: Path | str, run_id: str | None = None) -> None:
        self._dir = Path(run_dir)
        self._run_id = run_id
        self._rounds  = JsonlWriter(self._dir / "per_round.jsonl")
        self._reps    = JsonlWriter(self._dir / "per_repetition.jsonl")
        self._tel     = JsonlWriter(self._dir / "telemetry.jsonl")
        self._events  = JsonlWriter(self._dir / "events.jsonl")

    # ------------------------------------------------------------------
    def _stamp(self) -> dict[str, Any]:
        d: dict[str, Any] = {"ts": time.time()}
        if self._run_id:
            d["run_id"] = self._run_id
        return d

    def event(self, record: dict[str, Any]) -> None:
        """Write a lifecycle event (start / oom / sigterm / complete …)."""
        self._events.append({**self._stamp(), **record})

    def round(self, record: dict[str, Any]) -> None:
        """Write one FL-round record."""
        self._rounds.append({**self._stamp(), **record})

    def repetition(self, record: dict[str, Any]) -> None:
        """Write one repetition summary after it completes."""
        self._reps.append({**self._stamp(), **record})

    def telemetry(self, record: dict[str, Any]) -> None:
        """Write a telemetry snapshot."""
        self._tel.append({**self._stamp(), **record})

    # ------------------------------------------------------------------
    def close(self) -> None:
        for w in (self._rounds, self._reps, self._tel, self._events):
            try:
                w.close()
            except Exception:
                pass

    def __enter__(self) -> "DurabilityBundle":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
