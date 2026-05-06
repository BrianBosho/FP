"""Local on-disk cache for preprocessed federated client graph shards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ClientShardRef:
    """Lightweight pointer to a preprocessed client shard on local disk."""

    cache_dir: str
    client_id: int
    path: str
    num_nodes: int
    num_edges: int
    num_features: int
    train_count: int
    val_count: int
    test_count: int
    remote_count: int

    def load(self) -> Data:
        return torch.load(self.path, map_location="cpu", weights_only=False)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:20]


def shard_cache_enabled(config: dict[str, Any] | None) -> bool:
    return bool(config and config.get("use_shard_cache", False))


def build_cache_payload(
    dataset_name: str,
    data_loading_option: str,
    num_clients: int,
    beta: float,
    hop: int,
    fulltraining_flag: bool,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = config or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "data_loading": data_loading_option,
        "num_clients": int(num_clients),
        "beta": float(beta),
        "hop": int(hop),
        "fulltraining_flag": bool(fulltraining_flag),
        "experiment_seed": cfg.get("experiment_seed"),
        "use_pe": cfg.get("use_pe"),
        "num_iterations": cfg.get("num_iterations"),
        "diffusion_t": cfg.get("diffusion_t"),
        "alpha": cfg.get("alpha"),
        "feature_prop_init_strategy": cfg.get("feature_prop_init_strategy"),
        "feature_prop_tolerance": cfg.get("feature_prop_tolerance"),
        "prop_dtype": cfg.get("prop_dtype"),
        "model_input_affecting_schema": "pyg-data-v1",
    }


def get_cache_dir(
    dataset_name: str,
    data_loading_option: str,
    num_clients: int,
    beta: float,
    hop: int,
    fulltraining_flag: bool,
    config: dict[str, Any] | None,
) -> Path:
    cfg = config or {}
    root = Path(cfg.get("shard_cache_dir", "artifacts/shard_cache"))
    payload = build_cache_payload(
        dataset_name,
        data_loading_option,
        num_clients,
        beta,
        hop,
        fulltraining_flag,
        config,
    )
    slug = f"{dataset_name}_{data_loading_option}_c{num_clients}_b{beta}_h{hop}_{_fingerprint(payload)}"
    return root / slug


def _client_stats(client_id: int, shard_path: Path, data: Data) -> dict[str, Any]:
    def _count(name: str) -> int:
        value = getattr(data, name, None)
        return int(value.sum().item()) if value is not None and hasattr(value, "sum") else 0

    train_labels = []
    try:
        train_ids = data.train_mask.nonzero(as_tuple=True)[0]
        train_labels = data.y[train_ids].cpu().tolist()
    except Exception:
        train_labels = []
    class_hist = {}
    for label in train_labels:
        key = str(int(label))
        class_hist[key] = class_hist.get(key, 0) + 1

    remote_count = 0
    try:
        remote_count = int(getattr(data, "remote_local_ids").numel())
    except Exception:
        remote_count = 0

    return {
        "client_id": int(client_id),
        "path": str(shard_path),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)) if hasattr(data, "edge_index") else 0,
        "num_features": int(data.x.size(1)) if hasattr(data, "x") else 0,
        "train_count": _count("train_mask"),
        "val_count": _count("val_mask"),
        "test_count": _count("test_mask"),
        "remote_count": remote_count,
        "train_class_hist": class_hist,
    }


def write_shard_cache(
    cache_dir: Path,
    client_data: list[Data],
    metadata: dict[str, Any],
) -> list[ClientShardRef]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    clients = []
    for client_id, data in enumerate(client_data):
        shard_path = cache_dir / f"client_{client_id:04d}.pt"
        torch.save(data.to("cpu"), shard_path)
        clients.append(_client_stats(client_id, shard_path, data))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "clients": clients,
    }
    tmp_manifest = cache_dir / "manifest.json.tmp"
    manifest_path = cache_dir / "manifest.json"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp_manifest.replace(manifest_path)
    return refs_from_manifest(cache_dir, manifest)


def load_shard_cache(cache_dir: Path, expected_metadata: dict[str, Any]) -> list[ClientShardRef] | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return None
    if manifest.get("metadata") != expected_metadata:
        return None
    clients = manifest.get("clients")
    if not isinstance(clients, list):
        return None
    for client in clients:
        if not Path(client.get("path", "")).exists():
            return None
    return refs_from_manifest(cache_dir, manifest)


def refs_from_manifest(cache_dir: Path, manifest: dict[str, Any]) -> list[ClientShardRef]:
    refs = []
    for client in manifest["clients"]:
        refs.append(
            ClientShardRef(
                cache_dir=str(cache_dir),
                client_id=int(client["client_id"]),
                path=str(client["path"]),
                num_nodes=int(client["num_nodes"]),
                num_edges=int(client["num_edges"]),
                num_features=int(client["num_features"]),
                train_count=int(client["train_count"]),
                val_count=int(client["val_count"]),
                test_count=int(client["test_count"]),
                remote_count=int(client.get("remote_count", 0)),
            )
        )
    return refs


def is_shard_ref(value: Any) -> bool:
    return isinstance(value, ClientShardRef)
