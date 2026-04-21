"""Legacy compatibility wrapper — real implementation is in `src.fedgnn.models`."""

from src.fedgnn.models import (  # noqa: F401
    get_model_config,
    GCN,
    GAT_Arxiv,
    GCN_arxiv,
    GraphSAGEProducts,
    GAT,
    PubmedGAT,
    VanillaGNN,
    MLP,
    SparseVanillaGNN,
)

__all__ = [
    "get_model_config",
    "GCN",
    "GAT_Arxiv",
    "GCN_arxiv",
    "GraphSAGEProducts",
    "GAT",
    "PubmedGAT",
    "VanillaGNN",
    "MLP",
    "SparseVanillaGNN",
]
