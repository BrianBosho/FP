# FedGCN Params / Hyperparams

This document is a **FedGCN-paper-anchored** reference for configuring the GCN-side experiments in FedProp.

It is designed to answer one practical question:

> **Which hyperparameters should we adopt if we want our GCN experiments to be aligned as closely as possible to the FedGCN paper, while still running our own unified FedProp protocol?**

---

## 1. Scope and philosophy

This markdown distinguishes between two things:

1. **Directly inherited from the FedGCN paper**  
   These are settings explicitly stated in the FedGCN paper.
2. **FedGCN-anchored extensions**  
   These are settings for datasets **not directly covered by FedGCN's GCN experiments**, chosen to be consistent with the same design philosophy.

This means:

- For **Cora** and **Citeseer**, we can follow FedGCN almost directly.
- For **OGBN-Arxiv**, we can also follow FedGCN directly.
- For **Pubmed**, **Amazon-Computers**, **Amazon-Photo**, **Texas**, and **Wisconsin**, we must extend the FedGCN philosophy using dataset-standard conventions, because FedGCN does not provide a directly matching GCN configuration for all of them.

---

## 2. What the FedGCN paper explicitly uses

### 2.1 Datasets covered by FedGCN GCN experiments

FedGCN reports GCN results on:

- **Cora**
- **Citeseer**
- **OGBN-Arxiv**

It also reports on **OGBN-Products**, but there the paper uses **GraphSAGE**, not GCN.

### 2.2 FedGCN paper settings for Cora and Citeseer

The FedGCN paper explicitly states the following for **Cora** and **Citeseer**:

- **Backbone:** 2-layer GCN
- **Activation:** ReLU on the first layer, Softmax on the second layer
- **Hidden units:** 16
- **Dropout:** 0.5 between the two GCN layers
- **Optimizer:** SGD
- **Learning rate:** 0.5
- **L2 regularization / weight decay:** 5e-4
- **Training rounds:** 300
- **Local steps per round:** 3

### 2.3 FedGCN paper settings for OGBN-Arxiv

The FedGCN paper explicitly states the following for **OGBN-Arxiv**:

- **Backbone:** 3-layer GCN
- **Hidden units:** 256
- **Training rounds:** 600
- **Hyperparameter source:** OGB / Hu et al. style setup
- **BatchNorm:** used between GCN layers in their OGBN-Arxiv experiments

### 2.4 FedGCN protocol settings stated in the paper

FedGCN's experiment section also makes the following protocol choices clear:

- **Partition family:** Dirichlet label distribution
- **Beta values used in the paper:** 10000, 100, 1
- **Number of clients:**
  - 10 clients for Cora, Citeseer, OGBN-Arxiv
  - 5 clients for OGBN-Products
- **Reported metric:** test accuracy
- **Repetitions:** average over 10 runs

---

## 3. The main decision for FedProp

If the goal is:

> **"Make GCN experiments as defensible as possible relative to FedGCN"**

then the cleanest move is:

- Use **FedGCN paper settings directly** where available.
- Keep our own **FedProp unified protocol** where we intentionally differ.
- Be explicit in the paper about which parts are inherited and which parts are adapted.

That yields the following policy:

### 3.1 Use FedGCN directly for

- **Cora**
- **Citeseer**
- **OGBN-Arxiv**

### 3.2 Extend in the same spirit for

- **Pubmed**
- **Amazon-Computers**
- **Amazon-Photo**
- **Texas**
- **Wisconsin**

---

## 4. Recommended locked GCN configuration

## 4.1 Table: direct FedGCN-paper anchor + extensions

| Dataset | Coverage status | Layers | Hidden | Dropout | Normalization | Optimizer | LR | Weight decay | FL rounds | Local steps / round | Early stop patience | Notes |
|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| Cora | **Direct from FedGCN paper** | 2 | 16 | 0.5 | symmetric GCN norm | SGD | 0.5 | 5e-4 | 300 | 3 | 10 | Closest possible match to FedGCN |
| Citeseer | **Direct from FedGCN paper** | 2 | 16 | 0.5 | symmetric GCN norm | SGD | 0.5 | 5e-4 | 300 | 3 | 10 | Closest possible match to FedGCN |
| Pubmed | **Extension** | 2 | 16 | 0.5 | symmetric GCN norm | Adam | 0.01 | 5e-4 | 300 or 200* | 1 or 3* | 10 | Not directly specified by FedGCN; use citation-scale GCN with larger-graph optimizer |
| OGBN-Arxiv | **Direct from FedGCN paper** | 3 | 256 | 0.5 | symmetric GCN norm + BatchNorm | Adam | 0.01 | 0 | 600 | 3** | 20 | OGB-style setting adopted by FedGCN |
| Amazon-Computers | **Extension** | 2 | 64 | 0.5 | symmetric GCN norm | Adam | 0.01 | 5e-4 | 300 or 200* | 1 or 3* | 10 | Medium-scale dense graph extension |
| Amazon-Photo | **Extension** | 2 | 64 | 0.5 | symmetric GCN norm | Adam | 0.01 | 5e-4 | 300 or 200* | 1 or 3* | 10 | Same rationale as Amazon-Computers |
| Texas | **Extension** | 2 | 32 | 0.5 | symmetric GCN norm | Adam | 0.01 | 5e-4 | 300 or 200* | 1 or 3* | 20 | Heterophily limitation dataset |
| Wisconsin | **Extension** | 2 | 32 | 0.5 | symmetric GCN norm | Adam | 0.01 | 5e-4 | 300 or 200* | 1 or 3* | 20 | Heterophily limitation dataset |

### Important notes

- `*` For datasets not directly covered by FedGCN, there are two defensible choices:
  - **strict FedGCN-style FL shell:** use **300 rounds, 3 local steps**
  - **FedProp unified shell:** use **200 rounds, 1 local step**
- `**` FedGCN explicitly states 600 rounds for OGBN-Arxiv, but does not separately restate local steps in the appendix sentence. If you want maximum consistency, keep the same federated shell unless implementation evidence suggests otherwise.

---

## 5. My recommended final lock

If the goal is **scientific defensibility against FedGCN**, I recommend this exact split.

### 5.1 Strict FedGCN-aligned GCN lock

Use this if you want to say:

> **"Our GCN-side experiments are FedGCN-paper-aligned."**

| Dataset | Layers | Hidden | Dropout | Optimizer | LR | WD | FL rounds | Local steps |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Cora | 2 | 16 | 0.5 | SGD | 0.5 | 5e-4 | 300 | 3 |
| Citeseer | 2 | 16 | 0.5 | SGD | 0.5 | 5e-4 | 300 | 3 |
| OGBN-Arxiv | 3 | 256 | 0.5 | Adam | 0.01 | 0 | 600 | 3 |

### 5.2 Practical FedProp-wide lock (recommended if one shell is preferred)

Use this if you want to keep the entire project internally uniform while still being **FedGCN-anchored**.

| Dataset family | Layers | Hidden | Dropout | Optimizer | LR | WD | FL rounds | Local steps |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Cora / Citeseer | 2 | 16 | 0.5 | SGD | 0.5 | 5e-4 | 200 | 1 |
| Pubmed | 2 | 16 | 0.5 | Adam | 0.01 | 5e-4 | 200 | 1 |
| Amazon CS / Photo | 2 | 64 | 0.5 | Adam | 0.01 | 5e-4 | 200 | 1 |
| OGBN-Arxiv | 3 | 256 | 0.5 | Adam | 0.01 | 0 | 200 or 500*** | 1 |
| Texas / Wisconsin | 2 | 32 | 0.5 | Adam | 0.01 | 5e-4 | 200 | 1 |

`***` If OGBN-Arxiv is a headline dataset, it is better to allow **500-600 rounds** rather than force it into 200.

---

## 6. What should be inherited from FedGCN vs. what can stay ours

## 6.1 Inherit from FedGCN

These are the parts most worth inheriting:

- **Cora/Citeseer GCN optimizer family**: SGD, lr 0.5, wd 5e-4
- **Cora/Citeseer architecture**: 2-layer, hidden 16, dropout 0.5
- **OGBN-Arxiv architecture scale**: 3-layer, hidden 256, BatchNorm, Adam
- **Dirichlet family of partitioning**
- **10-run reporting**
- **0/1/2-hop communication framing** when comparing against FedGCN baselines

## 6.2 Keep as ours

These are the parts you can keep as part of the FedProp project protocol:

- **beta = 10000 and 10** instead of FedGCN's 10000, 100, 1
- **200 global rounds / 1 local epoch** if you want one unified shell
- **Pubmed / Amazon / Texas / Wisconsin additions**
- **FedProp-specific preprocessing settings** such as propagation iterations, diffusion step, and PE toggles

---

## 7. Recommended wording for the paper

You can use something like this in the experimental setup section:

```text
For GCN backbones, we use the FedGCN paper as our primary experimental anchor. In particular, for Cora and Citeseer we adopt the same citation-scale GCN architecture and optimizer family used in FedGCN, while for OGBN-Arxiv we use the same OGB-style deeper and wider GCN regime. For datasets not directly covered by FedGCN, we extend this setup using dataset-standard GCN conventions while keeping the federated protocol fixed across methods.
```

And, if needed, a more explicit variant:

```text
Our GCN-side comparison is FedGCN-anchored rather than identical to FedGCN in every detail: we inherit the paper's backbone and optimizer choices where directly available, while adapting the federated shell and dataset coverage to our unified evaluation protocol.
```

---

## 8. Practical update instructions for the experiment repo

If you want the experiments to reflect this markdown, update the GCN configs in this order.

### 8.1 Cora / Citeseer

Set to:

```yaml
model: GCN
num_layers: 2
hidden_dim: 16
dropout: 0.5
optimizer: sgd
learning_rate: 0.5
weight_decay: 0.0005
num_rounds: 300        # or 200 if keeping unified FedProp shell
local_step: 3          # or 1 if keeping unified FedProp shell
```

### 8.2 OGBN-Arxiv

Set to:

```yaml
model: GCN_arxiv
num_layers: 3
hidden_dim: 256
dropout: 0.5
batch_norm: true
optimizer: adam
learning_rate: 0.01
weight_decay: 0.0
num_rounds: 600        # preferred if mirroring FedGCN paper closely
local_step: 3          # use same shell unless codebase requires otherwise
```

### 8.3 Pubmed

Set to:

```yaml
model: GCN
num_layers: 2
hidden_dim: 16
dropout: 0.5
optimizer: adam
learning_rate: 0.01
weight_decay: 0.0005
num_rounds: 200 or 300
local_step: 1 or 3
```

### 8.4 Amazon-Computers / Amazon-Photo

Set to:

```yaml
model: GCN
num_layers: 2
hidden_dim: 64
dropout: 0.5
optimizer: adam
learning_rate: 0.01
weight_decay: 0.0005
num_rounds: 200 or 300
local_step: 1 or 3
```

### 8.5 Texas / Wisconsin

Set to:

```yaml
model: GCN
num_layers: 2
hidden_dim: 32
dropout: 0.5
optimizer: adam
learning_rate: 0.01
weight_decay: 0.0005
num_rounds: 200 or 300
local_step: 1 or 3
```

---

## 9. Final recommendation

If you want the cleanest and most defensible GCN story, do this:

### Main GCN comparison datasets
- **Cora**: direct FedGCN settings
- **Citeseer**: direct FedGCN settings
- **OGBN-Arxiv**: direct FedGCN settings

### Additional GCN datasets
- **Pubmed**: FedGCN-anchored extension
- **Amazon-Computers / Amazon-Photo**: FedGCN-anchored extension
- **Texas / Wisconsin**: limitation-focused extension

### Best single-sentence policy

> **Use FedGCN as the anchor for GCN-side settings wherever the paper directly covers the dataset; otherwise extend in the same architectural and optimization spirit while keeping the federated protocol unified.**

---

## 10. Checklist before rerunning experiments

- [ ] Decide whether to use **strict FedGCN shell** (`300 rounds, 3 local steps`) or **unified FedProp shell** (`200 rounds, 1 local step`).
- [ ] Update Cora and Citeseer GCN configs to match FedGCN-paper settings.
- [ ] Update OGBN-Arxiv to the OGB/FedGCN setting.
- [ ] Mark Pubmed/Amazon/Texas/Wisconsin as **extensions**, not direct FedGCN copies.
- [ ] Update experimental setup text to say **FedGCN-anchored** rather than **Kipf & Welling optimizer defaults**.
- [ ] Rerun any affected experiments whose optimizer, rounds, or local steps changed.

