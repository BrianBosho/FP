"""
Memory management utilities for federated learning with large graphs.
Provides comprehensive memory clearing functions to handle CUDA out of memory issues.
"""

import torch
import gc
import logging
from typing import Optional, Union

def clear_cuda_cache():
    """Basic CUDA cache clearing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def clear_memory_basic():
    """Basic memory clearing with garbage collection"""
    clear_cuda_cache()
    gc.collect()
    clear_cuda_cache()

def clear_memory_aggressive():
    """Aggressive memory clearing for memory-intensive operations"""
    # Multiple rounds of clearing
    for _ in range(3):
        clear_cuda_cache()
        gc.collect()
    
    # Force garbage collection of all generations
    gc.collect()
    
    # Final synchronization and cache clearing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def clear_memory_with_model(model: Optional[torch.nn.Module] = None, 
                           device: Union[str, torch.device] = "cuda"):
    """Clear memory with optional model movement to CPU"""
    if model is not None:
        # Temporarily move model to CPU to free GPU memory
        original_device = next(model.parameters()).device
        model.cpu()
        clear_memory_aggressive()
        model.to(device)
        clear_cuda_cache()
    else:
        clear_memory_aggressive()

def clear_memory_for_diffusion():
    """Specialized memory clearing for diffusion operations"""
    logging.info("Clearing memory for diffusion operation")
    
    # Multiple aggressive clearing rounds
    for i in range(3):
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Log memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        logging.info(f"Memory after clearing - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

def clear_memory_for_adjacency():
    """Specialized memory clearing for adjacency operations"""
    logging.info("Clearing memory for adjacency operation")
    
    # Clear cache and force garbage collection
    clear_memory_aggressive()
    
    # Additional clearing for sparse matrix operations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def log_memory_usage(stage: str = ""):
    """Log current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        logging.info(f"Memory {stage} - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Max: {max_allocated:.2f}GB")

def clear_memory_between_batches(batch_idx: int, clear_frequency: int = 5):
    """Clear memory between batches at specified frequency"""
    if batch_idx % clear_frequency == 0:
        clear_memory_basic()

def clear_memory_on_error():
    """Emergency memory clearing when errors occur"""
    logging.warning("Emergency memory clearing due to error")
    try:
        clear_memory_aggressive()
    except Exception as e:
        logging.error(f"Failed to clear memory: {e}")

def memory_guard(func):
    """Decorator to automatically clear memory before and after function execution"""
    def wrapper(*args, **kwargs):
        clear_memory_basic()
        try:
            result = func(*args, **kwargs)
            clear_memory_basic()
            return result
        except Exception as e:
            clear_memory_on_error()
            raise e
    return wrapper

def get_memory_info():
    """Get detailed memory information"""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    return {
        "cuda_available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "cached_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(),
    }
