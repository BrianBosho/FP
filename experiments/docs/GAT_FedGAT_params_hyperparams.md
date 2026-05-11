# GAT / FedGAT Params & Hyperparams

This note consolidates a **paper-anchored** hyperparameter reference for GAT-based experiments in FedProp.
It separates:

1. **Original GAT paper settings** (Veličković et al., ICLR 2018), which are the canonical source for citation-network GAT training.
2. **FedGAT paper settings** (Ambekar et al., 2024/2025 arXiv), which are the direct federated GAT comparison source.
3. **Recommended lock for FedProp experiments**, which uses the papers as anchors but keeps the federated protocol unified across methods.

---

## 1) What the original GAT paper gives us

The original GAT paper evaluates **Cora, Citeseer, and Pubmed** for the transductive citation-network setting.
It does **not** define an OGBN-Arxiv setup.

### Citation-network training settings from the original GAT paper

- **2 GAT layers**
- **8 attention heads** in the hidden layer
- **8 features per hidden head** (total hidden width = 64 before concatenation)
- **Final layer:** 1 head for transductive citation-network tasks
- **Dropout = 0.6** on both layer inputs and normalized attention coefficients
- **L2 regularization = 0.0005**
- **Learning rate = 0.005 for Cora/Citeseer**, **0.01 for Pubmed**
- **Early stopping patience = 100 epochs**

Source basis:
- Dropout 0.6 and L2 regularization 0.0005 are stated in the original GAT paper.
- LR 0.005 is used for all datasets except Pubmed, where LR 0.01 is used.

### Practical interpretation for FedProp

For **Cora / Citeseer / Pubmed**, the original GAT paper is the best direct anchor for the **backbone architecture** and the **core optimizer/LR regime**.

---

## 2) What the FedGAT paper gives us

The FedGAT paper performs federated node-classification experiments on:

- **Cora**
- **Citeseer**
- **Pubmed**

It does **not** report an experimental setup for **OGBN-Arxiv**.
The only mention of OGBN-Arxiv is illustrative/background, not as an evaluated dataset.

### Experimental setup reported by FedGAT

The FedGAT appendix states:

- For **Cora**:
  - **2 GAT layers**
  - **Hidden dimensions = 8**
  - **Attention heads = 8**
  - **Optimizer = Adam**
  - **Regularization = 0.001**
  - **Learning rate = 0.1**
- For **Citeseer**:
  - **exact same architecture as Cora**
- For **Pubmed**:
  - same base architecture, **but with 8 attention heads in the output layer too**
- Approximation scheme:
  - **Chebyshev approximation, degree 16**

### Important caution on FedGAT settings

These FedGAT hyperparameters are the paper's settings for **their federated approximation algorithm**, not canonical general-purpose GAT defaults.
In particular:

- **Adam lr = 0.1** is much more aggressive than the original GAT paper.
- **Regularization = 0.001** differs from the original GAT paper's 0.0005.
- The setup is tuned for **FedGAT's approximation-based federated training**, not necessarily for a plain GAT backbone in every setting.

So FedGAT is best treated as:

- the **primary comparator source** for the **federated GAT baseline**, and
- **not** the universal source for all GAT hyperparameters across all datasets.

---

## 3) Dataset coverage: which paper anchors which dataset?

| Dataset | Original GAT paper? | FedGAT paper? | Best anchor |
|---|---|---|---|
| Cora | Yes | Yes | Original GAT for backbone, FedGAT for federated comparator |
| Citeseer | Yes | Yes | Original GAT for backbone, FedGAT for federated comparator |
| Pubmed | Yes | Yes | Original GAT for backbone, FedGAT for federated comparator |
| OGBN-Arxiv | No | No | OGB-style / large-scale GAT literature, not FedGAT |
| Amazon-Computers | No | No | community-standard extension |
| Amazon-Photo | No | No | community-standard extension |
| Texas | No | No | heterophily / Geom-GCN-style extension |
| Wisconsin | No | No | heterophily / Geom-GCN-style extension |

---

## 4) Recommended GAT lock for FedProp experiments

This is the recommended **paper-consistent lock** for your experiments.
It distinguishes:

- **FedProp backbone settings**: should be stable, literature-consistent, and not overly tailored to FedGAT.
- **FedGAT comparison settings**: should follow the FedGAT paper as closely as feasible when running FedGAT itself.

### 4.1 FedProp GAT backbone lock (recommended)

| Dataset | Layers | Hidden per head | Heads (hidden / output) | Dropout | Optimizer | LR | Weight decay | Notes |
|---|---:|---:|---|---:|---|---:|---:|---|
| Cora | 2 | 8 | 8 / 1 | 0.6 | Adam | 0.005 | 0.0005 | Directly aligned with original GAT paper |
| Citeseer | 2 | 8 | 8 / 1 | 0.6 | Adam | 0.005 | 0.0005 | Directly aligned with original GAT paper |
| Pubmed | 2 | 8 | 8 / 1 | 0.6 | Adam | 0.01 | 0.0005 | Aligned with original GAT LR choice for Pubmed |
| OGBN-Arxiv | 3 | ~85 | 3 / 1 or ~256 total | 0.5 | Adam | 0.005 | 0 or 0.0005 | OGB-style extension; not from original GAT or FedGAT |
| Amazon-Computers | 2 | 8 | 8 / 1 | 0.6 | Adam | 0.005 | 0.0005 | Community-style extension from citation-scale GAT |
| Amazon-Photo | 2 | 8 | 8 / 1 | 0.6 | Adam | 0.005 | 0.0005 | Community-style extension from citation-scale GAT |
| Texas | 2 | 8 | 8 / 1 | 0.6 | Adam | 0.005 or 0.01 | 0.0005 | Small heterophily extension; tune lightly if needed |
| Wisconsin | 2 | 8 | 8 / 1 | 0.6 | Adam | 0.005 or 0.01 | 0.0005 | Small heterophily extension; tune lightly if needed |

### Why this is the recommended FedProp GAT lock

- It stays **faithful to the original GAT paper** where that paper actually provides settings.
- It avoids importing the unusually aggressive **FedGAT lr = 0.1** into all your own GAT experiments.
- It gives you a stable, easy-to-defend backbone setting for **FedProp + GAT**.

---

## 5) Recommended FedGAT comparison lock

When you run **FedGAT itself** as a baseline, use the **FedGAT paper settings**, not your FedProp GAT backbone lock.

| Dataset | Layers | Hidden dims | Heads | Dropout | Optimizer | LR | Regularization | Other |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Cora | 2 | 8 | 8 | not explicitly redefined in appendix snippet | Adam | 0.1 | 0.001 | Chebyshev degree 16 |
| Citeseer | 2 | 8 | 8 | same as Cora architecture | Adam | 0.1 | 0.001 | Chebyshev degree 16 |
| Pubmed | 2 | 8 | 8 hidden + 8 output | modified output layer | Adam | 0.1 | 0.001 | Chebyshev degree 16 |

### Why keep FedGAT separate

FedGAT is not just a standard GAT training recipe. It is a federated approximation algorithm for GATs.
So its settings should be used for:

- **running FedGAT as a baseline**,
- **reproducing FedGAT-style results**,

but not automatically for:

- **FedProp + GAT backbone settings**, or
- all extended datasets outside the FedGAT paper.

---

## 6) OGBN-Arxiv: what to do?

### Answer
**FedGAT does not report OGBN-Arxiv experiments.**
So neither the original GAT paper nor the FedGAT paper gives you a direct OGBN-Arxiv GAT lock.

### Recommended policy

For **OGBN-Arxiv with GAT**, use an **OGB-style large-scale GAT extension**, for example:

- **3 layers**
- **~256 total hidden width**
- **dropout 0.5**
- **Adam**
- **lr 0.005**
- **BatchNorm on hidden layers if your implementation supports it cleanly**
- early stopping / patience matched to your OGB protocol

This should be presented as a **dataset-standard extension**, not as “from the FedGAT paper.”

---

## 7) Clean policy for the paper

Use this wording in the experiment section:

> For GAT backbones on Cora, Citeseer, and Pubmed, we follow the original Graph Attention Networks paper for the backbone architecture and core optimization regime. For the federated FedGAT baseline, we use the settings reported in the FedGAT paper. Since FedGAT does not report OGBN-Arxiv experiments, our OGBN-Arxiv GAT configuration follows a standard large-scale OGB-style extension rather than the FedGAT paper.

---

## 8) Final recommendation summary

### Use for **FedProp + GAT backbone**
- **Original GAT paper** for Cora/Citeseer/Pubmed
- **OGB-style extension** for OGBN-Arxiv
- cautious community-style extensions for Amazon / Texas / Wisconsin

### Use for **FedGAT baseline**
- **FedGAT paper settings** for Cora/Citeseer/Pubmed
- **No OGBN-Arxiv FedGAT row unless you explicitly design and justify an extension**

### Bottom line
- **Yes**, we can do the same for GAT.
- **Yes**, the original GAT paper + FedGAT paper are the right pair of anchors.
- **Correct:** **FedGAT does not experimentally use OGBN-Arxiv**.
