# Feature Propagation Ablation & Research Plan

This document outlines the systematic investigation into Feature Propagation (FP) parameters, heuristics, and theoretical grounding within the FedProp framework.

## 1. Core Objectives
- **Empirical Validation:** Reproduce baseline results and establish a stable benchmark.
- **Sensitivity Analysis:** Explore the impact of FP hyperparameters (iterations, step size, propagators).
- **Theoretical Refinement:** Develop grounded heuristics for evaluating FP quality and convergence.
- **Advanced Techniques:** Evaluate adaptive FP and distillation regularizers.

## 2. Theoretical Grounding of Heuristics

### 2.1. Convergence "Conscience"
How do we know if FP has done its job?
- **Stationary Distribution Gap:** For Markovian propagators (Adjacency, RWR), the target is the stationary state. We can monitor the distance to the manifold of harmonic functions: $\|(I - P)X_{unknown}\|$.
- **Dirichlet Energy Smoothing:** Monitor the rate of change in Dirichlet Energy $E(X)$. A stable $E(X)$ suggests the features have reached a natural "smoothness" plateau.
- **Spectral Filtering Analysis:** FP acts as a low-pass filter $H(L)X$. We can quantify the high-frequency "leakage" to determine if the propagation was deep enough.

### 2.2. Step Size ($\alpha$) Dynamics
The current update rule is: $X^{(k+1)} = (1-\alpha) X^{(k)} + \alpha P X^{(k)}$.
- **Theoretical Step Size:** If $P$ is symmetrically normalized, $\alpha=1.0$ is the standard Power Method. However, dampening with $\alpha < 1.0$ can prevent oscillations in bipartite subgraphs (where $\lambda \approx -1$).
- **Ablation Target:** Test $\alpha \in \{0.1, 0.25, 0.5, 0.75, 1.0\}$.
- **Momentum FP:** Introduce a momentum term $\beta(X^{(k)} - X^{(k-1)})$ to accelerate convergence in large-diameter subgraphs.

## 3. Implementation Ablation Matrix

| Ablation ID | Parameter | Values | Goal |
|---|---|---|---|
| **FP-1** | Iterations | $\{5, 10, 20, 50, 100, 200\}$ | Determine the "diminishing returns" point. |
| **FP-2** | Step Size ($\alpha$) | $\{0.1, 0.3, 0.5, 0.8, 1.0\}$ | Optimize convergence speed vs stability. |
| **FP-3** | Propagator | $\{Adj, RWR, Diff, Chebyshev\}$ | Compare structural recovery capabilities. |
| **FP-4** | Initialization | $\{Zero, Mean, Neighbor, Spectral\}$ | Test sensitivity to "warm start". |
| **FP-5** | Adaptive $\alpha, t$ | $\{Static, Structural\}$ | Test if local topology awareness helps. |

## 4. Constraint Checklist
- [x] **Zero Communication:** All FP must be strictly local to the client subgraph.
- [x] **Reproducibility:** Use `experiment_seed` to ensure deterministic propagation.
- [x] **Resource Efficiency:** Monitor GPU memory, especially for Diffusion modes.

## 5. Execution Steps

### Phase 1: Baseline Synchronization
1. Run a full experiment using `experiments/configs/R1/R1_cora.yaml`.
2. Compare results with targets in `experiments/docs/FedProp_Experimental_Design_Locked.md`.

### Phase 2: Heuristic Development
1. Instrument `propagate_features` to log detailed per-step metrics (residual, energy, delta).
2. Visualize these metrics to identify "convergence signatures".

### Phase 3: Systematic Ablations
1. Use the experiment runner to sweep the parameters defined in Section 3.
2. Focus on the interplay between **Step Size** and **Iterations**.

### Phase 4: Refinement
1. Implement the adaptive FP logic based on findings.
2. Integrate the Distillation Regularizer if FP-derived features prove high-quality.
