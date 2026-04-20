"""
Memory helpers (Phase C compatibility layer).
"""

from src.utils.memory_utils import (  # noqa: F401
    clear_cuda_cache,
    clear_memory_basic,
    clear_memory_aggressive,
    clear_memory_with_model,
    clear_memory_for_diffusion,
    clear_memory_for_adjacency,
    log_memory_usage,
    clear_memory_between_batches,
    clear_memory_on_error,
    memory_guard,
    get_memory_info,
)

__all__ = [
    "clear_cuda_cache",
    "clear_memory_basic",
    "clear_memory_aggressive",
    "clear_memory_with_model",
    "clear_memory_for_diffusion",
    "clear_memory_for_adjacency",
    "log_memory_usage",
    "clear_memory_between_batches",
    "clear_memory_on_error",
    "memory_guard",
    "get_memory_info",
]

