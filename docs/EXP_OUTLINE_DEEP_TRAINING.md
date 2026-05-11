# Experimental Outline: Deep Training Calibration
**Objective:** Determine if the static accuracy results in Phase 1 were an artifact of premature training termination. We will expand the training budget to 200 rounds to allow the GNN to fully exploit the structural information recovered by Feature Propagation.

---

## 1. Core Changes

### 1.1 Extended Training Budget
*   **Max Rounds:** Increased from 50 to **200**.
*   **Early Stopping Patience:** Increased to **20 rounds** to ensure we capture late-stage convergence plateaus.
*   **Why:** Current logs show early stopping at round 41-44. With a 50-round cap, the model had no "breathing room" to stabilize.

### 1.2 Fine-Grained Propagation Sweep
Since Phase 1 showed that Diffusion is "slower" but more signal-rich, we will focus on high-depth diffusion:
*   **Diffusion Iterations:** $\{50, 100, 200\}$.
*   **Diffusion Time ($t$):** $\{0.1, 0.5, 1.0\}$.

---

## 2. Experimental Parameters

| Factor | Value |
|---|---|
| **Datasets** | Cora, Citeseer |
| **Data Loading** | `diffusion`, `adjacency` |
| **Max rounds** | 200 |
| **Local Epochs** | 1 |
| **Learning Rate** | 0.5 (SGD) |
| **Patience** | 20 |
| **Repetitions** | 3 (To ensure accuracy deltas are statistically significant) |

---

## 3. Heuristic Focus
We will specifically look for the **"Convergence Crossing"**:
*   Does a deep 200-iteration propagation (low residual) combined with deep training (200 rounds) finally break the 0.745 accuracy barrier?
*   Or does deep training actually favor *shallower* propagation because it allows the GNN to "learn" the missing features itself rather than relying on a potentially over-smoothed prior?

---

## 4. Execution Plan

1.  **Config Creation:** `conf/ablations/A1b_deep_training.yaml`.
2.  **Instrumentation:** Keep the existing structural diagnostic logging (Residual, Variance).
3.  **Run:** Execute via the background sweep runner.
4.  **Analysis:** Compare the 200-round final accuracies against the 50-round Phase 1 results.
