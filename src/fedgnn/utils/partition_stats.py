"""Partition statistics: per-client graph sizes, memory, and cross-client edges."""

import torch


def compute_partition_stats(data, clients_data):
    """Compute per-client graph statistics and cross-client edge count.

    Uses owned_global_ids / communicate_global_ids set by the partitioner on
    every subgraph.  Falls back gracefully when those attributes are absent
    (e.g. legacy loaders that don't call _attach_client_index_bookkeeping).

    Returns a dict with keys:
      client_stats        — list of per-client dicts (num_nodes, num_edges, …)
      cross_client_edges  — int or None if not computable
      cross_client_pct    — float % or None
      overlap_nodes       — nodes appearing in >1 subgraph communicate set
      total_full_nodes    — int
      total_full_edges    — int or None
    """
    from src.fedgnn.data.shard_cache import is_shard_ref

    n_clients = len(clients_data)
    client_stats = []
    node_to_owner = {}      # global_node_id -> client_id
    subgraph_count = {}     # global_node_id -> how many subgraphs include it

    for i, cd_ref in enumerate(clients_data):
        cd = cd_ref.load() if is_shard_ref(cd_ref) else cd_ref

        n_nodes = cd.num_nodes
        n_edges = int(cd.edge_index.shape[1]) if hasattr(cd, 'edge_index') and cd.edge_index is not None else 0
        n_train = int(cd.train_mask.sum()) if hasattr(cd, 'train_mask') else 0
        n_val   = int(cd.val_mask.sum())   if hasattr(cd, 'val_mask')   else 0
        n_test  = int(cd.test_mask.sum())  if hasattr(cd, 'test_mask')  else 0

        owned = (
            cd.owned_global_ids.tolist()
            if hasattr(cd, 'owned_global_ids') and cd.owned_global_ids is not None
            else None
        )
        comm_ids = (
            cd.communicate_global_ids.tolist()
            if hasattr(cd, 'communicate_global_ids') and cd.communicate_global_ids is not None
            else None
        )

        client_stats.append({
            'client_id':   i,
            'num_nodes':   n_nodes,
            'num_edges':   n_edges,
            'owned_nodes': len(owned) if owned is not None else n_train,
            'train_nodes': n_train,
            'val_nodes':   n_val,
            'test_nodes':  n_test,
        })

        if owned is not None:
            for gid in owned:
                node_to_owner[gid] = i

        if comm_ids is not None:
            for gid in comm_ids:
                subgraph_count[gid] = subgraph_count.get(gid, 0) + 1

    # Cross-client edges (vectorized, O(E) but on GPU-free CPU tensors)
    cross_client_edges = cross_client_pct = None
    if node_to_owner and hasattr(data, 'edge_index') and data.edge_index is not None:
        n_full = data.num_nodes
        ownership = torch.full((n_full,), -1, dtype=torch.long)
        for gid, cid in node_to_owner.items():
            if gid < n_full:
                ownership[gid] = cid

        ei = data.edge_index.cpu()
        src_cl = ownership[ei[0]]
        dst_cl = ownership[ei[1]]
        cross_mask = (src_cl != dst_cl) & (src_cl >= 0) & (dst_cl >= 0)
        cross_client_edges = int(cross_mask.sum())
        total_edges = int(ei.shape[1])
        cross_client_pct = 100.0 * cross_client_edges / total_edges if total_edges > 0 else 0.0

    overlap_nodes = sum(1 for cnt in subgraph_count.values() if cnt > 1)

    return {
        'client_stats':       client_stats,
        'cross_client_edges': cross_client_edges,
        'cross_client_pct':   cross_client_pct,
        'overlap_nodes':      overlap_nodes,
        'total_full_nodes':   data.num_nodes,
        'total_full_edges':   int(data.edge_index.shape[1]) if hasattr(data, 'edge_index') else None,
    }


def print_partition_stats(stats, data, mem_stats=None):
    """Print partition statistics in FedGCN-style tabular format."""

    client_stats = stats['client_stats']
    n_clients    = len(client_stats)
    W = 100
    SEP = "=" * W
    sep = "-" * W

    print(f"\n{SEP}")
    print("PARTITION STATISTICS")
    print(SEP)

    # Per-client table
    if mem_stats:
        print(f"{'Client':<8} {'Nodes':<8} {'Edges':<10} {'Owned':<8} "
              f"{'Train':<7} {'Val':<6} {'Test':<6} "
              f"{'CPU_RSS(MB)':<13} {'Peak_GPU(MB)':<13} {'MB/Node':<9}")
        print(sep)
        for cs, ms in zip(client_stats, mem_stats):
            cpu_mb = ms.get('cpu_rss_mb', float('nan'))
            gpu_mb = ms.get('peak_gpu_mb') or float('nan')
            mb_per_node = cpu_mb / cs['num_nodes'] if cs['num_nodes'] > 0 else float('nan')
            print(f"{cs['client_id']:<8} {cs['num_nodes']:<8} {cs['num_edges']:<10} "
                  f"{cs['owned_nodes']:<8} {cs['train_nodes']:<7} {cs['val_nodes']:<6} {cs['test_nodes']:<6} "
                  f"{cpu_mb:<13.1f} {gpu_mb:<13.3f} {mb_per_node:<9.3f}")
    else:
        print(f"{'Client':<8} {'Nodes':<8} {'Edges':<10} {'Owned':<8} "
              f"{'Train':<7} {'Val':<6} {'Test':<6}")
        print(sep)
        for cs in client_stats:
            print(f"{cs['client_id']:<8} {cs['num_nodes']:<8} {cs['num_edges']:<10} "
                  f"{cs['owned_nodes']:<8} {cs['train_nodes']:<7} {cs['val_nodes']:<6} {cs['test_nodes']:<6}")

    # Aggregate summary
    print(sep)
    total_sub_nodes = sum(cs['num_nodes'] for cs in client_stats)
    total_sub_edges = sum(cs['num_edges'] for cs in client_stats)
    avg_nodes = total_sub_nodes / n_clients
    avg_edges = total_sub_edges / n_clients
    max_nodes = max(cs['num_nodes'] for cs in client_stats)
    min_nodes = min(cs['num_nodes'] for cs in client_stats)
    max_cl = next(cs['client_id'] for cs in client_stats if cs['num_nodes'] == max_nodes)
    min_cl = next(cs['client_id'] for cs in client_stats if cs['num_nodes'] == min_nodes)
    full_nodes = stats['total_full_nodes']
    full_edges = stats['total_full_edges']

    print(f"Subgraph totals : nodes={total_sub_nodes:,}  edges={total_sub_edges:,}")
    print(f"Full graph      : nodes={full_nodes:,}  edges={full_edges:,}")
    print(f"Avg per client  : nodes={avg_nodes:.1f}  edges={avg_edges:.1f}")
    print(f"Node range      : max={max_nodes} (client {max_cl})  min={min_nodes} (client {min_cl})")
    print(f"Overlap nodes   : {stats['overlap_nodes']:,} nodes appear in >1 subgraph "
          f"({100*stats['overlap_nodes']/full_nodes:.1f}% of full graph)")

    if mem_stats:
        cpu_mbs = [ms.get('cpu_rss_mb', 0) for ms in mem_stats]
        total_cpu = sum(cpu_mbs)
        max_cpu   = max(cpu_mbs)
        min_cpu   = min(cpu_mbs)
        max_cpu_cl = cpu_mbs.index(max_cpu)
        min_cpu_cl = cpu_mbs.index(min_cpu)
        print(f"Total CPU RSS   : {total_cpu:.1f} MB ({total_cpu/1024:.2f} GB)")
        print(f"Avg/client      : {total_cpu/n_clients:.1f} MB  "
              f"Max: {max_cpu:.1f} MB (client {max_cpu_cl})  "
              f"Min: {min_cpu:.1f} MB (client {min_cpu_cl})")

    # Cross-client edges
    print(sep)
    if stats['cross_client_edges'] is not None:
        cc  = stats['cross_client_edges']
        pct = stats['cross_client_pct']
        print(f"Cross-client edges : {cc:,} / {full_edges:,} ({pct:.1f}%)")
        if hasattr(data, 'x') and data.x is not None:
            feat_dim = data.x.shape[1]
            comm_mb  = cc * feat_dim * 4 / (1024 ** 2)
            print(f"Theoretical cross-client feature comm : {comm_mb:.2f} MB  (feat_dim={feat_dim})")
    else:
        print("Cross-client edges : n/a (owned_global_ids not present on subgraphs)")

    print(SEP)
    print()
