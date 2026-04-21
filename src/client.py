"""Legacy wrapper — real implementation is in `src.fedgnn.fl.client`."""

from src.fedgnn.fl.client import (  # noqa: F401
    FLClient,
    LARGE_DATASET_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_NEIGHBORS,
    OGBN_ARXIV_BATCH_SIZE,
    OGBN_ARXIV_NUM_NEIGHBORS,
)
