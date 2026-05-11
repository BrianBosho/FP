# Advanced Feature Propagation Strategies: Refined Research Plan

This document outlines the tightened strategy for investigating and improving Feature Propagation (FP) in Federated GNNs, incorporating feedback on structural diagnostics, multi-scale fusion, and regularized learning.

---

## 1. Core Strategic Framing: The Bias-Variance Trade-off
We treat Feature Propagation not merely as a convergence problem, but as a **Bias-Variance problem**:
*   **Bias Reduction:** Recovering signal for missing neighbors to improve local model gradients.
*   **Variance Loss (Over-smoothing):** Excessive propagation leading to a non-discriminative "flat" signal where all nodes converge to a global mean.
*   **Goal:** Identify the "Goldilocks zone" where structural recovery is maximized before feature collapse occurs.

---

## 2. Advanced Techniques

### 2.1 Multi-Scale Weighted Fusion (High Priority)
Instead of a single propagation depth, we capture information at multiple scales and fuse them.
*   **Mechanism:** Run parallel passes at $T \in \{5, 20, 50\}$.
*   **Fusion Strategy:** Use **Weighted Averaging** ($X_{final} = \sum w_i X_{Ti}$) rather than concatenation. This preserves input dimensionality and ensures controlled model capacity.
*   **Reference:** Grounded in *SIGN (Simple Graph Convolutional Networks)* architectures but adapted for local federated subgraphs.

### 2.2 Topology-Aware Scale ($t$)
Shift from density-aware $\alpha$ to topology-aware diffusion time $t$.
*   **Metric:** Subgraph diameter approximation or Average Path Length.
*   **Logic:** The heat kernel $e^{-tL}$ has a natural scale parameter $t$. Subgraphs with larger diameters require higher $t$ to allow features to traverse the partition boundaries.

### 2.3 Structural Consistency Regularization (Refined)
Reframed from "distillation" to avoid dimensional/semantic mismatch.
*   **Mechanism:** Add a Laplacian-style regularizer $\lambda \| Z - X_{fp} \|^2$ where $Z$ is the hidden representation of the first GNN layer.
*   **Safety:** Implement $\lambda$ scheduling (e.g., warm-up or decay) to prevent the structural prior from suppressing discriminative learning in later rounds.

---

## 3. Evaluation & Heuristics ("The FP Conscience")

We will implement a rich diagnostic suite to monitor the "conscience" of the propagation. These are **diagnostics first**, stopping-criteria second.

### 3.1 Convergence Diagnostics
*   **Stationary Residual:** $\| P X^{(k)} - X^{(k)} \|$ on unknown nodes. Measures how close we are to the harmonic manifold.
*   **Dirichlet Energy Plateau:** $\Delta E(X)$. Identifies the point where structural smoothing yields diminishing returns.

### 3.2 Signal Integrity Diagnostics
*   **Feature Norm Drift:** Monitor $\|X^{(k)}\| / \|X^{(0)}\|$. Detects vanishing or exploding feature scales during deep propagation.
*   **Pairwise Feature Variance:** Average variance across nodes. A sharp drop indicates the onset of over-smoothing.

---

## 4. Tightened Ablation Matrix

| ID | Factor | Values | Priority |
|---|---|---|---|
| **A1** | **Diagnostic Baseline** | Standard Adj/Diff + New Logging | **CRITICAL** |
| **A2** | **Multi-Scale Fusion** | Single vs. Weighted Average ($T=\{5, 20, 50\}$) | **HIGH** |
| **A3** | **Initialization** | $\{Zero, Mean, Neighbor\}$ | **HIGH** |
| **A4** | **Step Size ($\alpha$)** | $\{0.1, 0.5, 1.0\}$ | **MEDIUM** |
| **A5** | **Consistency $\lambda$** | $\{0, 1e-4, 1e-2\}$ with scheduling | **LOW** |

---

## 5. Execution Roadmap

1.  **Phase 1: Instrumentation.** Add logging for Residual, Energy, Norm Drift, and Variance to `propagate_features`.
2.  **Phase 2: Diagnostic Sweep.** Run Cora/Citeseer with a grid of `iterations` × `diffusion_t`. Identify if diagnostic plateaus correlate with test accuracy.
3.  **Phase 3: Multi-Scale.** Implement weighted fusion and compare against the best single-scale baseline.
4.  **Phase 4: Regularization.** Test structural consistency on the hardest (most non-IID) partitions.
