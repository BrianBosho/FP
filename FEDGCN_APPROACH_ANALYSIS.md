# FedGCN Memory-Efficient Approach - Complete Analysis

## **Why FedGCN Doesn't Run Out of Memory**

### **Key Insight: Subgraph-Based Training (No Full Graph Per Client)**

FedGCN's approach is fundamentally different from naive federated learning:

1. ✅ **Load FULL graph on CPU once** (main process)
2. ✅ **Create k-hop SUBGRAPHS** for each client  
3. ✅ **Each client receives ONLY their small subgraph**
4. ✅ **Full-batch training on SMALL subgraph** (no mini-batching needed)

---

## **Step-by-Step: How FedGCN Handles ogbn-arxiv**

### **Step 1: Load Full Data on CPU** (fedgcn_run.py:75)
```python
features, adj, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
# Features: (169,343, 128) - all nodes on CPU
# Adj: Sparse adjacency matrix - on CPU  
# Labels: (169,343,) - all labels on CPU
```

**Memory**: ~500MB on CPU (not GPU!)

---

### **Step 2: Partition Node Indices** (fedgcn_run.py:115-122)
```python
split_data_indexes = label_dirichlet_partition(
    labels, len(labels), class_num, args.n_trainer, beta=args.iid_beta
)
# Result: List of node indices for each client
# Example with 10 clients: 
#   split_data_indexes[0] = [234, 567, 1234, ...]  # ~17K indices
#   split_data_indexes[1] = [12, 89, 345, ...]     # ~17K indices
```

**Memory**: Negligible (just indices)

---

### **Step 3: Get K-Hop Communication Indices** (fedgcn_run.py:136-148)
```python
communicate_indexes, in_com_train_data_indexes, edge_indexes_clients = get_in_comm_indexes(
    edge_index, split_data_indexes, args.n_trainer, args.num_hops, idx_train, idx_test
)

# For each client i:
# - communicate_indexes[i]: ALL nodes in client i's k-hop neighborhood
#   Example: 17K original + 10K k-hop neighbors = 27K total nodes
# - in_com_train_data_indexes[i]: Training indices WITHIN that subgraph
# - edge_indexes_clients[i]: Edges for ONLY that subgraph (relabeled to 0-27K)
```

**Key Point**: Each client gets 20-30K nodes, NOT 169K!

---

### **Step 4: Create Trainers with SUBGRAPH Data** (fedgcn_run.py:201-215)
```python
trainers = [
    Trainer.remote(
        i,
        edge_indexes_clients[i],           # Subgraph edges (e.g., 50K edges)
        labels[communicate_indexes[i]],     # Subgraph labels (e.g., 27K labels)
        features[communicate_indexes[i]],   # Subgraph features (e.g., 27K × 128)
        in_com_train_data_indexes[i],      # Training indices in subgraph
        in_com_test_data_indexes[i],       # Test indices in subgraph
        args_hidden, class_num, device, args
    )
    for i in range(args.n_trainer)
]
```

**Critical**: Each trainer receives:
- NOT the full 169K graph
- ONLY their ~27K node subgraph
- All data CPU-side initially

---

### **Step 5: Trainer Moves SMALL Subgraph to GPU** (trainer_class.py:68-72)
```python
def __init__(self, ...):
    self.adj = adj.to(device)          # ~27K×27K sparse matrix → GPU
    self.labels = labels.to(device)    # ~27K labels → GPU
    self.features = features.to(device) # ~27K×128 features → GPU
    self.idx_train = idx_train.to(device)
    self.idx_test = idx_test.to(device)
```

**Memory per client**: 
- Features: 27K × 128 × 4 bytes = ~14 MB
- Adj (sparse): ~50K edges × 8 bytes = ~0.4 MB
- Model: ~50 MB (GCN with 256 hidden)
- **Total: ~100-150 MB per client**

With 10 clients: 1-1.5 GB total ✅

---

### **Step 6: Full-Batch Training on SMALL Subgraph** (train_func.py:83)
```python
def train(...):
    output = model(features, adj)  # Forward pass on 27K nodes
    loss = F.nll_loss(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
```

**Key Points**:
- ❌ NO mini-batching
- ❌ NO NeighborLoader
- ❌ NO sampling
- ✅ Full-batch on SMALL subgraph (27K nodes fits easily)

---

## **FedProp's Current Approach vs FedGCN**

### **FedProp (CORRECT - Already Implemented)**:
```python
# loaders.py:104
clients_data, test_data, _ = partition_data(data, num_clients, beta, ...)

# partitioning.py:187
for i in range(num_clients):
    subgraph, node_map, mapping = create_k_hop_subgraph(
        data, split_data_indexes[i], hop, device, ...
    )
    clients_data.append(subgraph)  # Each client gets SUBGRAPH

# run.py:135
clients = [FLClient.remote(data, ...) for data in clients_data]
# CORRECT: Each client gets their SUBGRAPH, not full graph
```

✅ **FedProp ALREADY creates subgraphs correctly!**

---

## **The Memory Issue - What Went WRONG**

### **❌ WRONG Approach (My Bad Fix)**:
```python
# BAD: Using -1 neighbors in NeighborLoader
self.num_neighbors_eval = [-1, -1, -1]  # Tries to load ALL neighbors

# Result in NeighborLoader:
NeighborLoader(data, num_neighbors=[-1, -1, -1], ...)
# This tries to access the FULL neighborhood
# Defeats the subgraph purpose!
# Memory explosion! 💥
```

### **✅ CORRECT Approach**:
```python
# OPTION 1: Full-batch on subgraph (like FedGCN)
if subgraph.num_nodes < 50000:  # Small subgraph
    output = model(subgraph.x, subgraph.edge_index)  # No mini-batching

# OPTION 2: Mini-batching with REASONABLE neighbors (if subgraph is large)
if subgraph.num_nodes >= 50000:  # Large subgraph
    self.num_neighbors = [10, 10, 10]  # MODERATE sampling
    # NOT [-1, -1, -1] which defeats subgraph!
```

---

## **Why Mini-Batching in FedProp is Different**

### **FedGCN**: 
- Each subgraph: ~20-30K nodes
- **Full-batch works** → No mini-batching needed

### **FedProp** (with k-hop expansion):
- Each subgraph: Can be 40-60K nodes (larger k-hop)
- **Full-batch may not fit** → Mini-batching needed

But the key: **Mini-batch WITHIN the subgraph**, not the full graph!

---

## **Correct Mini-Batching Strategy for FedProp**

```python
# client.py - CURRENT (CORRECT)
self.batch_size = cfg.get("batch_size", DEFAULT_BATCH_SIZE)
self.num_neighbors = cfg.get("num_neighbors", DEFAULT_NUM_NEIGHBORS)  # [10, 10, 10]

# For ogbn-arxiv:
if self.dataset_name == "ogbn-arxiv":
    self.batch_size = OGBN_ARXIV_BATCH_SIZE  # 2048
    self.num_neighbors = OGBN_ARXIV_NUM_NEIGHBORS  # [10, 10, 10]

# train.py - Mini-batch on SUBGRAPH
train_loader = NeighborLoader(
    data,  # This is the SUBGRAPH (e.g., 40K nodes), not full graph!
    num_neighbors=[10, 10, 10],  # Sample 10 neighbors per layer
    batch_size=2048,
    input_nodes=train_idx,
)
```

**This works because**:
- `data` is already a subgraph (40K nodes)
- `NeighborLoader` samples WITHIN the 40K node subgraph
- Memory stays reasonable

---

## **The Critical Mistake to Avoid**

### **❌ NEVER DO THIS**:
```python
num_neighbors = [-1, -1, -1]  # ALL neighbors
```

**Why it's bad**:
- Defeats the subgraph optimization
- Tries to load entire neighborhood
- For k-hop subgraph with high connectivity → Memory explosion
- Example: 40K node subgraph with avg degree 10 → 400K neighbors!

### **✅ INSTEAD DO THIS**:
```python
# For training: Use moderate sampling
num_neighbors = [10, 10, 10]  # Sample 10 neighbors per layer

# For eval/test on SUBGRAPH: 
# Option 1: Same as training (consistent)
num_neighbors = [10, 10, 10]

# Option 2: Slightly more (better accuracy, more memory)
num_neighbors = [15, 15, 15]

# Option 3: If subgraph is small enough, full-batch
if subgraph.num_nodes < 50000:
    # Don't use NeighborLoader, just do full forward pass
    output = model(data.x, data.edge_index)
```

---

## **Summary: Correct Memory-Efficient Approach**

### **FedGCN Approach (for small subgraphs)**:
1. ✅ Create k-hop subgraphs (~20-30K nodes)
2. ✅ Each client gets ONLY their subgraph
3. ✅ Full-batch training on subgraph
4. ✅ No mini-batching needed

### **FedProp Approach (for larger subgraphs)**:
1. ✅ Create k-hop subgraphs (~40-60K nodes) - **ALREADY DONE**
2. ✅ Each client gets ONLY their subgraph - **ALREADY DONE**
3. ✅ Mini-batch training WITHIN subgraph
4. ✅ Use MODERATE neighbor sampling (e.g., [10, 10, 10]) - **CURRENT**
5. ❌ **NEVER use [-1, -1, -1]** - defeats subgraph purpose

---

## **Current FedProp Status**

✅ **What's Already Correct**:
- Subgraph creation (partitioning.py)
- Client initialization with subgraphs (run.py)
- Mini-batching with moderate neighbors (client.py)
- Data stays on CPU until needed (train.py - fixed)

❌ **What Was Wrong** (now reverted):
- Using `[-1, -1, -1]` for evaluation/testing
- This defeated the subgraph optimization

✅ **Current Configuration (CORRECT)**:
```python
# client.py
self.num_neighbors = [10, 10, 10]  # For ALL operations
self.batch_size = 2048  # For ogbn-arxiv
```

This matches the spirit of FedGCN (subgraph-based) while using mini-batching for larger subgraphs.

---

## **Recommendations**

### **For Memory Optimization**:
1. ✅ Keep current approach (subgraphs + moderate sampling)
2. ✅ Use `data_loading: zero` to reduce k-hop feature memory
3. ✅ Reduce `hop` from 2 to 1 if needed (smaller subgraphs)
4. ❌ NEVER use `num_neighbors: [-1, -1, -1]`

### **For Result Accuracy**:
- Current sampling ([10, 10, 10]) gives reasonable results
- Trade-off: More neighbors = better accuracy but more memory
- Suggested range: [10, 10, 10] to [20, 15, 10]
- The subgraph structure already provides the graph context

---

## **Final Takeaway**

**FedGCN's secret**: Not mini-batching, but **small subgraphs**!

**FedProp's approach**: Correct subgraphing + mini-batching for larger subgraphs

**The mistake**: Using [-1, -1, -1] which defeats the subgraph optimization

**The solution**: Keep moderate neighbor sampling ([10, 10, 10]) for all operations

