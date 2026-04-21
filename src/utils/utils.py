"""Legacy compatibility wrapper — real implementation is in `src.fedgnn.utils.config`."""

from src.fedgnn.utils.config import load_config  # noqa: F401

__all__ = ["load_config"]
