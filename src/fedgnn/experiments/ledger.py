"""Run ledger: durable packet store for experiment runs.

Each experiment condition appends one packet when it starts and an update
record when it finishes. The scheduler reads the ledger instead of scanning
result directories, so ``resume_completed`` becomes reliable even after
partial failures.

File format: append-only JSONL at ``<ledger_dir>/run_ledger.jsonl``.
Concurrent writers are serialised via ``fcntl.LOCK_EX``.

Usage::

    ledger = RunLedger("runs/experiments")

    # At the start of a condition:
    packet = RunPacket(
        condition_key  = make_condition_key(...),
        requested_config = {...},
        status         = STATUS_RUNNING,
    )
    ledger.append(packet)

    # On success:
    ledger.update_status(packet.run_id, STATUS_SUCCESS,
                         result_path=filepath, end_time=time.time())

    # Query:
    done = ledger.completed_condition_keys()
"""

from __future__ import annotations

import fcntl
import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

_LEDGER_FILENAME = "run_ledger.jsonl"

STATUS_PENDING  = "pending"
STATUS_RUNNING  = "running"
STATUS_SUCCESS  = "success"
STATUS_FAILED   = "failed"
STATUS_PARTIAL  = "partial_failed"
STATUS_SKIPPED  = "skipped_cached"


def _dumps(v: Any) -> str:
    return json.dumps(v, separators=(",", ":"), default=str)


def make_condition_key(
    dataset: str,
    data_loading: str,
    model: str,
    beta: float,
    clients: int,
    hop: int,
    use_pe: bool,
    seed: Any = None,
) -> str:
    """Stable string key that uniquely identifies one sweep condition."""
    parts = [
        dataset, data_loading, model,
        f"b{beta}", f"c{clients}", f"h{hop}",
        "pe" if use_pe else "nope",
    ]
    if seed is not None:
        parts.append(f"s{seed}")
    return "_".join(str(p) for p in parts)


class RunPacket:
    """Immutable snapshot for one experiment attempt."""

    __slots__ = (
        "run_id", "attempt_id",
        "condition_key",
        "requested_config", "effective_config",
        "status",
        "start_time", "end_time",
        "result_path", "log_path",
        "failure_kind", "cache_state",
    )

    def __init__(
        self,
        *,
        run_id: str | None = None,
        attempt_id: str | None = None,
        condition_key: str = "",
        requested_config: dict[str, Any] | None = None,
        effective_config: dict[str, Any] | None = None,
        status: str = STATUS_PENDING,
        start_time: float | None = None,
        end_time: float | None = None,
        result_path: str = "",
        log_path: str = "",
        failure_kind: str = "",
        cache_state: str = "",
    ) -> None:
        self.run_id          = run_id or str(uuid.uuid4())
        self.attempt_id      = attempt_id or str(uuid.uuid4())
        self.condition_key   = condition_key
        self.requested_config = requested_config or {}
        self.effective_config = effective_config or {}
        self.status          = status
        self.start_time      = start_time if start_time is not None else time.time()
        self.end_time        = end_time
        self.result_path     = result_path
        self.log_path        = log_path
        self.failure_kind    = failure_kind
        self.cache_state     = cache_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id":           self.run_id,
            "attempt_id":       self.attempt_id,
            "condition_key":    self.condition_key,
            "requested_config": self.requested_config,
            "effective_config": self.effective_config,
            "status":           self.status,
            "start_time":       self.start_time,
            "end_time":         self.end_time,
            "result_path":      self.result_path,
            "log_path":         self.log_path,
            "failure_kind":     self.failure_kind,
            "cache_state":      self.cache_state,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunPacket":
        return cls(
            run_id=d.get("run_id"),
            attempt_id=d.get("attempt_id"),
            condition_key=d.get("condition_key", ""),
            requested_config=d.get("requested_config"),
            effective_config=d.get("effective_config"),
            status=d.get("status", STATUS_PENDING),
            start_time=d.get("start_time"),
            end_time=d.get("end_time"),
            result_path=d.get("result_path", ""),
            log_path=d.get("log_path", ""),
            failure_kind=d.get("failure_kind", ""),
            cache_state=d.get("cache_state", ""),
        )


class RunLedger:
    """Append-only, file-locked ledger stored as JSONL.

    Read semantics: for each run_id the *latest* record wins — update
    records (``_type: "update"``) patch the original packet fields.
    """

    def __init__(self, ledger_dir: str | Path) -> None:
        self._dir  = Path(ledger_dir)
        self._path = self._dir / _LEDGER_FILENAME
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, packet: RunPacket) -> None:
        """Atomically append a packet record."""
        self._write_line(_dumps(packet.to_dict()))

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        end_time: float | None = None,
        result_path: str = "",
        failure_kind: str = "",
        cache_state: str = "",
    ) -> None:
        """Append an update record (ledger is append-only; readers merge by run_id)."""
        record: dict[str, Any] = {
            "_type":    "update",
            "run_id":   run_id,
            "status":   status,
            "end_time": end_time if end_time is not None else time.time(),
        }
        if result_path:
            record["result_path"] = result_path
        if failure_kind:
            record["failure_kind"] = failure_kind
        if cache_state:
            record["cache_state"] = cache_state
        self._write_line(_dumps(record))

    def _write_line(self, line: str) -> None:
        with open(self._path, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line + "\n")
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _iter_raw(self) -> Iterator[dict[str, Any]]:
        if not self._path.exists():
            return
        with open(self._path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    continue

    def all_packets(self) -> dict[str, RunPacket]:
        """Return the merged latest state for every run_id."""
        base: dict[str, dict[str, Any]] = {}
        for rec in self._iter_raw():
            rid = rec.get("run_id")
            if not rid:
                continue
            if rec.get("_type") == "update":
                if rid in base:
                    base[rid].update({k: v for k, v in rec.items() if k != "_type"})
            else:
                base[rid] = dict(rec)
        return {rid: RunPacket.from_dict(d) for rid, d in base.items()}

    def completed_condition_keys(self) -> set[str]:
        """Return condition_keys whose latest status is ``success`` or ``skipped_cached``."""
        terminal = {STATUS_SUCCESS, STATUS_SKIPPED}
        return {
            p.condition_key
            for p in self.all_packets().values()
            if p.status in terminal and p.condition_key
        }

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.all_packets().values():
            counts[p.status] = counts.get(p.status, 0) + 1
        return counts
