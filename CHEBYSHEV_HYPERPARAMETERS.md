# Chebyshev Diffusion Hyperparameter Tuning Guide

## 🔍 ROOT CAUSE ANALYSIS

### Why Chebyshev is Underperforming

**Critical Issue**: The Chebyshev and Taylor methods are NOT being applied the same way:

1. **Taylor-based `diffusion` method** (lines 156-162 in data_utils.py):
   - Builds exp(-tL) operator with t=1.0
   - Applies it **50 times iteratively** with alpha=0.5 blending
   - Each iteration: `out = 0.5 * (operator @ out) + 0.5 * out`
   - Resets known features each iteration
   - Result: **Gradual, controlled diffusion**

2. **Chebyshev `chebyshev_diffusion` method** (lines 166-170):
   - Applies exp(-tL) **ONCE** with t=1.0
   - Restores known features
   - Returns immediately - **NO iterations**
   - Result: **Aggressive single-shot diffusion**

3. **Chebyshev `chebyshev_diffusion_operator` method** (lines 171-175):
   - Builds exp(-tL) operator with t=1.0
   - Applies it **ONCE** (num_iterations=1)
   - Result: **Still single-shot**

### The Problem

Applying exp(-tL) with t=1.0 in a single shot is TOO AGGRESSIVE. It over-smooths features, losing important information. The Taylor method applies it 50 times with blending, creating a gentler diffusion.

---

## 🎯 HYPERPARAMETERS TO TUNE

### 1. **Diffusion Time `t`** ⭐ MOST IMPORTANT

**Current**: `t = 1.0`  
**Location**: `src/dataprocessing/data_utils.py` lines 170, 175

**Impact**: Controls strength of single diffusion step
- **Larger t**: More smoothing, features become more similar
- **Smaller t**: Less smoothing, preserves local structure

**Recommended values to test**:
```python
t_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
```

**Why this matters**: For single-shot application, smaller t (0.01-0.1) might work better than t=1.0

**Expected behavior**:
- `t=0.01-0.1`: Subtle diffusion, may need more iterations
- `t=0.5-1.0`: Medium diffusion
- `t=2.0+`: Strong diffusion, risk of over-smoothing

---

### 2. **Chebyshev Order `K`** 

**Current**: `K = 5`  
**Location**: `src/dataprocessing/data_utils.py` lines 170, 175

**Impact**: Approximation accuracy of exp(-tL)
- **Lower K** (3-5): Faster, less accurate
- **Higher K** (10-20): Slower, more accurate

**Recommended values to test**:
```python
K_values = [3, 5, 8, 10, 15, 20]
```

**Trade-offs**:
- K=3: Fastest, may be inaccurate for large t
- K=5-8: Good balance (current default)
- K=10-15: Better accuracy, especially for larger t
- K=20+: Overkill for most graphs

**Expected behavior**:
- Accuracy improves with K, but diminishing returns after K=10
- For large t (1.0+), need higher K for accuracy
- For small t (0.01-0.1), K=3-5 sufficient

---

### 3. **Number of Iterations** ⭐ CRITICAL FIX

**Current**: 
- Matrix-free: 0 iterations (returns immediately)
- Operator: 1 iteration

**Location**: 
- `propagate_features_efficient()` line 487-492 (matrix-free)
- `data_utils.py` line 175 (operator)

**Recommended FIX**: Make Chebyshev iterative like Taylor method

**Option A - Use smaller t with iterations**:
```python
# Apply gentle diffusion multiple times
t = 0.05  # Smaller for iterative application
K = 5
num_iterations = 50  # Match Taylor method
```

**Option B - Use very small t, many iterations**:
```python
# Mimic traditional label propagation
t = 0.01
K = 3  # Can use lower K for small t
num_iterations = 50
```

**Option C - Moderate t, fewer iterations**:
```python
# Balance between single-shot and iterative
t = 0.2
K = 8
num_iterations = 10
```

---

### 4. **Alpha Blending Weight**

**Current**: Not used for Chebyshev (matrix-free returns immediately)  
**Location**: `propagate_features_efficient()` line 524

**Impact**: Controls how much new features mix with old
- `alpha=0.5`: Equal weight to old and new (default for Taylor)
- `alpha=0.7`: Favor new features (more aggressive)
- `alpha=0.3`: Favor old features (more conservative)

**Recommended**: Test if making Chebyshev iterative
```python
alpha_values = [0.3, 0.5, 0.7]
```

---

### 5. **Convergence Tolerance**

**Current**: `tol = 1e-3` (from config)  
**Location**: Set in partitioning.py, used in propagate_features

**Impact**: When to stop iterations
- Smaller: More iterations, better convergence
- Larger: Fewer iterations, faster

**Recommended values**:
```python
tolerance_values = [1e-4, 1e-3, 1e-2]
```

---

## 🔧 IMPLEMENTATION FIXES

### Fix 1: Make Matrix-Free Chebyshev Iterative

**File**: `src/dataprocessing/data_utils.py`

**Change line 168-170 from**:
```python
elif mode == "chebyshev_diffusion":
    return propagate_features_efficient(x, edge_index, mask, device, 
                                       propagation_type="chebyshev_diffusion",
                                       chebyshev_k=5, diffusion_t=1.0)
```

**To**:
```python
elif mode == "chebyshev_diffusion":
    return propagate_features_efficient(x, edge_index, mask, device, 
                                       propagation_type="chebyshev_diffusion",
                                       chebyshev_k=5, 
                                       diffusion_t=0.05,  # SMALLER for iterative
                                       num_iterations=50)  # ADD iterations
```

**File**: `src/dataprocessing/propagation_functions.py`

**Modify `propagate_features_efficient()` lines 486-492**:

Change from returning immediately to allowing iterations:
```python
# Special case: matrix-free Chebyshev diffusion
if propagation_type == "chebyshev_diffusion":
    # For matrix-free, we need to apply iteratively with small t
    # Track previous iteration for convergence
    prev_out = None
    out = torch.zeros_like(x)
    out[mask] = x[mask]
    
    for i in range(num_iterations):
        # Apply one step of Chebyshev diffusion
        new_out = chebyshev_expmL_apply(edge_index, num_nodes, out, 
                                        t=diffusion_t, K=chebyshev_k, device=device)
        
        # Weighted combination
        beta = 0.5
        out = beta * new_out + (1 - beta) * out
        
        # Restore known features
        out[mask] = x[mask]
        
        # Check convergence
        if prev_out is not None and torch.allclose(out, prev_out, rtol=1e-5):
            print(f"Chebyshev converged after {i+1} iterations")
            break
        prev_out = out.clone()
    
    return out
```

---

### Fix 2: Update Operator Method Iterations

**File**: `src/dataprocessing/data_utils.py`

**Change line 173-175**:
```python
elif mode == "chebyshev_diffusion_operator":
    return propagate_features_efficient(x, edge_index, mask, device, 
                                       propagation_type="chebyshev_diffusion_operator",
                                       chebyshev_k=5, 
                                       diffusion_t=0.05,  # Smaller
                                       num_iterations=50)  # More iterations
```

---

## 📊 EXPERIMENTAL GRID

### Recommended Testing Grid

**Phase 1 - Fix the iteration issue** (Test these first):
```yaml
experiments:
  - name: "cheb_iterative_t0.05_k5"
    t: 0.05
    K: 5
    iterations: 50
    
  - name: "cheb_iterative_t0.1_k5"
    t: 0.1
    K: 5
    iterations: 50
    
  - name: "cheb_iterative_t0.01_k5"
    t: 0.01
    K: 5
    iterations: 50
```

**Phase 2 - Optimize K**:
```yaml
# Once you find good t, test K values
experiments:
  - {t: 0.05, K: 3, iterations: 50}
  - {t: 0.05, K: 5, iterations: 50}
  - {t: 0.05, K: 8, iterations: 50}
  - {t: 0.05, K: 10, iterations: 50}
```

**Phase 3 - Fine-tune iterations**:
```yaml
# Test if fewer iterations work with optimized t and K
experiments:
  - {t: 0.05, K: 8, iterations: 20}
  - {t: 0.05, K: 8, iterations: 30}
  - {t: 0.05, K: 8, iterations: 50}
```

---

## 🎛️ CONFIG FILE MODIFICATIONS

### Option 1: Add to YAML config

**File**: `conf/ablation/ogbn-arxiv_config.yaml`

```yaml
# Add these new parameters
chebyshev_k: 5
chebyshev_t: 0.05
chebyshev_iterations: 50
```

Then modify code to read from config:
```python
# In data_utils.py
chebyshev_k = config.get('chebyshev_k', 5)
chebyshev_t = config.get('chebyshev_t', 0.05)
chebyshev_iterations = config.get('chebyshev_iterations', 50)
```

### Option 2: Direct code modification

Edit `src/dataprocessing/data_utils.py` lines 168-175 directly with your chosen values.

---

## 🔍 DEBUGGING / VALIDATION

### Check approximation quality

Add this test function:
```python
def test_chebyshev_accuracy(edge_index, num_nodes, device):
    """Compare Chebyshev vs Taylor for same t, K"""
    import torch
    from src.dataprocessing.propagation_functions import (
        chebyshev_expmL_operator, diffusion_kernel
    )
    
    t = 1.0
    K = 5
    
    # Build both operators
    cheb_op = chebyshev_expmL_operator(edge_index, num_nodes, t=t, K=K, device=device)
    taylor_op = diffusion_kernel(edge_index, num_nodes, device, t=t)
    
    # Convert to dense for comparison
    cheb_dense = cheb_op.to_dense()
    taylor_dense = taylor_op.to_dense()
    
    # Compute difference
    diff = torch.abs(cheb_dense - taylor_dense).max().item()
    print(f"Max difference between Chebyshev and Taylor: {diff:.6f}")
    
    return diff
```

---

## 🎯 QUICK START - MOST LIKELY FIX

**Based on analysis, try this first**:

1. **File**: `src/dataprocessing/data_utils.py`

2. **Change lines 166-175** to:
```python
elif mode == "chebyshev_diffusion":
    return propagate_features_efficient(x, edge_index, mask, device, 
                                       propagation_type="chebyshev_diffusion",
                                       chebyshev_k=5, 
                                       diffusion_t=0.05,  # Much smaller!
                                       num_iterations=50)  # Add iterations!

elif mode == "chebyshev_diffusion_operator":
    return propagate_features_efficient(x, edge_index, mask, device, 
                                       propagation_type="chebyshev_diffusion_operator",
                                       chebyshev_k=5, 
                                       diffusion_t=0.05,  # Much smaller!
                                       num_iterations=50)  # More iterations!
```

3. **File**: `src/dataprocessing/propagation_functions.py`

4. **Modify lines 486-492** to implement iterative Chebyshev (see Fix 1 above)

This should bring performance much closer to the Taylor method!

---

## 📈 EXPECTED IMPROVEMENTS

After implementing iterative Chebyshev with t=0.05-0.1:

| Method | Current Global | Expected Global | Current Client | Expected Client |
|--------|---------------|-----------------|----------------|-----------------|
| Taylor diffusion | 0.794 | 0.794 | 0.7817 | 0.7817 |
| Chebyshev (current) | 0.770 | - | 0.6742 | - |
| **Chebyshev (fixed)** | **0.770** | **~0.79** | **0.6742** | **~0.78** |

The fix should bring Chebyshev performance within 1-2% of Taylor, with potential speed advantages.

