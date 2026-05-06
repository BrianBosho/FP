"""Lightweight experiment telemetry: phase timings, memory peaks, comm estimates, CSV summaries."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import torch

FP_CSV_HEADER = (
    "Dataset,Model,DataLoading,Beta,Clients,Hop,UsePE,Seed,Status,TotalTime[s],"
    "LoadTime[s],ShardTime[s],PartitionTime[s],PropTime[s],ActorInitTime[s],TrainTime[s],EvalTime[s],"
    "CommTime[s],FinalAcc,BestAcc,PeakCPU[MB],PeakDriverGPU[MB],PeakActorGPU[MB],"
    "CommCost[MB],ModelSize[MB],TotalParams,ResultPath"
)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Mapping[str, Any] | dict | None) -> str | None:
    if config is None:
        return None
    try:
        payload = stable_json(dict(config))
    except TypeError:
        payload = stable_json(json.loads(json.dumps(config, default=str)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def model_param_stats(model: torch.nn.Module) -> dict[str, Any]:
    """Return parameter count and approximate model size (float weights only; buffers counted separately)."""
    total_params = 0
    trainable = 0
    bytes_fp = 0
    for p in model.parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable += n
        bytes_fp += n * p.element_size()
    for b in model.buffers():
        bytes_fp += b.numel() * b.element_size()
    mb = bytes_fp / (1024 * 1024)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable),
        "model_size_mb": round(mb, 6),
        "model_bytes": int(bytes_fp),
    }


def fedavg_comm_estimates(
    num_clients: int,
    num_rounds: int,
    model_size_mb: float,
) -> dict[str, float]:
    """Symmetric FedAvg-style parameter traffic (upload + download per round, all clients)."""
    nc = max(0, int(num_clients))
    nr = max(0, int(num_rounds))
    m = float(model_size_mb)
    upload_mb_round = nc * m
    broadcast_mb_round = nc * m
    total_mb = nr * (upload_mb_round + broadcast_mb_round)
    return {
        "param_upload_mb_per_round": round(upload_mb_round, 6),
        "param_broadcast_mb_per_round": round(broadcast_mb_round, 6),
        "total_theoretical_comm_mb": round(total_mb, 6),
    }


def peak_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        rss = psutil.Process().memory_info().rss
        return round(rss / (1024 * 1024), 3)
    except Exception:
        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux: kilobytes; macOS: bytes — heuristic
            if ru > 10**9:
                return round(ru / (1024 * 1024), 3)
            return round(ru / 1024.0, 3)
        except Exception:
            return None


def peak_cuda_allocated_mb(device_index: int = 0) -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        return round(torch.cuda.max_memory_allocated(device_index) / (1024 * 1024), 3)
    except Exception:
        return None


class TelemetryCollector:
    """Accumulates phase seconds (driver process), peaks, and model/comm stats."""

    def __init__(self) -> None:
        self.phases: dict[str, float] = {}
        self.peak_cpu_mb: float | None = None
        self.peak_gpu_mb: float | None = None          # driver-process peak
        self.actor_peak_gpu_mb: float | None = None   # max across Ray actor workers
        self.actor_peak_gpu_mb_per_client: list[float] | None = None
        self.loader_timings: dict[str, float] = {}
        self.model_stats: dict[str, Any] | None = None
        self.comm_estimates: dict[str, float] | None = None
        self.adaptive_changes: list[str] = []

    def add_phase(self, name: str, seconds: float) -> None:
        if seconds is None or math.isnan(seconds):
            return
        self.phases[name] = self.phases.get(name, 0.0) + float(seconds)

    def note_peaks(self) -> None:
        cpu = peak_rss_mb()
        if cpu is not None:
            self.peak_cpu_mb = cpu if self.peak_cpu_mb is None else max(self.peak_cpu_mb, cpu)
        gpu = peak_cuda_allocated_mb()
        if gpu is not None:
            self.peak_gpu_mb = gpu if self.peak_gpu_mb is None else max(self.peak_gpu_mb, gpu)

    def merge_loader_timings(self, sink: Mapping[str, float] | None) -> None:
        if not sink:
            return
        for k, v in sink.items():
            try:
                self.loader_timings[k] = self.loader_timings.get(k, 0.0) + float(v)
            except (TypeError, ValueError):
                continue
        load_ds = float(sink.get("dataset_load_s", 0.0) or 0.0)
        shard = float(sink.get("shard_cache_hit_s", 0.0) or 0.0) + float(
            sink.get("shard_cache_write_s", 0.0) or 0.0
        )
        part = float(sink.get("partition_s", 0.0) or 0.0)
        fp = float(sink.get("feature_propagation_s", 0.0) or 0.0)
        pe = float(sink.get("positional_encoding_global_s", 0.0) or 0.0)
        self.add_phase("dataset_load", load_ds)
        self.add_phase("shard_cache", shard)
        self.add_phase("partition", part)
        self.add_phase("feature_propagation", fp)
        self.add_phase("positional_encoding_global", pe)

    def set_model_and_comm(self, model: torch.nn.Module, num_clients: int, num_rounds: int) -> None:
        stats = model_param_stats(model)
        self.model_stats = stats
        self.comm_estimates = fedavg_comm_estimates(
            num_clients,
            num_rounds,
            stats["model_size_mb"],
        )

    def to_json_blob(self) -> dict[str, Any]:
        return {
            "phases_sec": dict(self.phases),
            "loader_timings_sec": dict(self.loader_timings),
            "peak_cpu_mb": self.peak_cpu_mb,
            "peak_driver_gpu_mb": self.peak_gpu_mb,
            "peak_actor_gpu_mb": self.actor_peak_gpu_mb,
            "peak_actor_gpu_mb_per_client": self.actor_peak_gpu_mb_per_client,
            "model": self.model_stats,
            "communication_estimates_mb": self.comm_estimates,
            "adaptive_changes": list(self.adaptive_changes),
        }


def _fmt_num(x: Any, digits: int = 6) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return ""
        return str(round(float(x), digits))
    except (TypeError, ValueError):
        return ""


def format_fp_csv_result_row(fields: Mapping[str, Any]) -> str:
    """Single CSV data line (no header)."""
    keys = [
        "Dataset",
        "Model",
        "DataLoading",
        "Beta",
        "Clients",
        "Hop",
        "UsePE",
        "Seed",
        "Status",
        "TotalTime[s]",
        "LoadTime[s]",
        "ShardTime[s]",
        "PartitionTime[s]",
        "PropTime[s]",
        "ActorInitTime[s]",
        "TrainTime[s]",
        "EvalTime[s]",
        "CommTime[s]",
        "FinalAcc",
        "BestAcc",
        "PeakCPU[MB]",
        "PeakDriverGPU[MB]",
        "PeakActorGPU[MB]",
        "CommCost[MB]",
        "ModelSize[MB]",
        "TotalParams",
        "ResultPath",
    ]
    parts = []
    for k in keys:
        v = fields.get(k, "")
        if v is None:
            v = ""
        s = str(v)
        if any(c in s for c in ',"\n'):
            s = '"' + s.replace('"', '""') + '"'
        parts.append(s)
    return ",".join(parts)


def print_fp_csv_result_block(fields: Mapping[str, Any]) -> None:
    print("\nFP CSV FORMAT RESULT:")
    print(FP_CSV_HEADER)
    print(format_fp_csv_result_row(fields))


def build_fp_csv_fields(
    result: Mapping[str, Any],
    wall_clock_s: float | None,
    result_path: str = "",
    *,
    dataset: str = "",
    model: str = "",
    data_loading: str = "",
    beta: Any = "",
    clients: Any = "",
    hop: Any = "",
    use_pe: Any = "",
    seed: Any = "",
    status: str = "success",
) -> dict[str, Any]:
    """Build the canonical FP CSV row dict from a ``results_data`` blob plus sweep metadata."""
    tel = result.get("telemetry") or {}
    phases = tel.get("phases_sec") or {}
    lt = tel.get("loader_timings_sec") or {}
    comm = tel.get("communication_estimates_mb") or {}
    mod = tel.get("model") or {}

    load_t = lt.get("dataset_load_s", phases.get("dataset_load"))
    shard_t = (lt.get("shard_cache_hit_s", 0.0) or 0.0) + (lt.get("shard_cache_write_s", 0.0) or 0.0)
    if not shard_t:
        shard_t = phases.get("shard_cache")
    part_t = lt.get("partition_s", phases.get("partition"))
    # PropTime: actual feature-propagation/PE work inside partition_data + optional global PE
    _fp_s = lt.get("feature_propagation_s") or 0.0
    _pe_s = lt.get("positional_encoding_global_s") or 0.0
    prop_t = (_fp_s + _pe_s) or phases.get("positional_encoding_global") or None
    actor_t = phases.get("actor_init")
    train_t = phases.get("train")
    eval_t = phases.get("eval")

    ecfg = result.get("experiment_config") or {}
    summary = result.get("summary") or {}

    final_acc = summary.get("average_global_result")
    best_acc = result.get("best_eval_acc_across_reps")
    if best_acc is None:
        best_acc = summary.get("average_global_result")

    total_comm = comm.get("total_theoretical_comm_mb")

    def _cli():
        if clients != "":
            return clients
        return ecfg.get("effective_num_clients", ecfg.get("num_clients", ""))

    return {
        "Dataset": dataset or ecfg.get("dataset", ""),
        "Model": model or ecfg.get("model_type", ""),
        "DataLoading": data_loading or ecfg.get("data_loading_option", ""),
        "Beta": beta if beta != "" else ecfg.get("beta", ""),
        "Clients": _cli(),
        "Hop": hop if hop != "" else ecfg.get("hop", ""),
        "UsePE": use_pe if use_pe != "" else ecfg.get("use_pe", ""),
        "Seed": seed if seed != "" else ecfg.get("experiment_seed", ""),
        # Explicit caller-supplied status (e.g. "skipped_cached", "failed") takes priority;
        # fall back to the value recorded in result["summary"]["status"] only when the
        # default "success" placeholder is still in place.
        "Status": status if status != "success" else (result.get("summary") or {}).get("status", status),
        "TotalTime[s]": _fmt_num(wall_clock_s, 3) if wall_clock_s is not None else "",
        "LoadTime[s]": _fmt_num(load_t, 3),
        "ShardTime[s]": _fmt_num(shard_t, 3),
        "PartitionTime[s]": _fmt_num(part_t, 3),
        "PropTime[s]": _fmt_num(prop_t, 3),
        "ActorInitTime[s]": _fmt_num(actor_t, 3),
        "TrainTime[s]": _fmt_num(train_t, 3),
        "EvalTime[s]": _fmt_num(eval_t, 3),
        "CommTime[s]": "",
        "FinalAcc": _fmt_num(final_acc, 6),
        "BestAcc": _fmt_num(best_acc, 6),
        "PeakCPU[MB]": _fmt_num(tel.get("peak_cpu_mb"), 3),
        "PeakDriverGPU[MB]": _fmt_num(tel.get("peak_driver_gpu_mb"), 3),
        "PeakActorGPU[MB]": _fmt_num(tel.get("peak_actor_gpu_mb"), 3),
        "CommCost[MB]": _fmt_num(total_comm, 6),
        "ModelSize[MB]": _fmt_num(mod.get("model_size_mb"), 6),
        "TotalParams": mod.get("total_params", ""),
        "ResultPath": result_path or "",
    }
