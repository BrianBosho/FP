"""Federated learning module — real implementations live here."""

from .client import (
    FLClient,
    LARGE_DATASET_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_NEIGHBORS,
    OGBN_ARXIV_BATCH_SIZE,
    OGBN_ARXIV_NUM_NEIGHBORS,
)

from .server import (
    Server,
    LARGE_DATASET_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_NEIGHBORS,
)

from .train import (
    train,
    evaluate,
    test,
    train_with_minibatch,
    evaluate_with_minibatch,
    test_with_minibatch,
    set_seed,
)

from .run import (
    load_configuration,
    instantiate_model,
    initialize_clients,
    load_data,
    run_with_server,
    main_experiment,
    verify_test_masks,
)

__all__ = [
    "FLClient",
    "Server",
    "train",
    "evaluate",
    "test",
    "train_with_minibatch",
    "evaluate_with_minibatch",
    "test_with_minibatch",
    "set_seed",
    "load_configuration",
    "instantiate_model",
    "initialize_clients",
    "load_data",
    "run_with_server",
    "main_experiment",
    "verify_test_masks",
    "LARGE_DATASET_THRESHOLD",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_NUM_NEIGHBORS",
    "OGBN_ARXIV_BATCH_SIZE",
    "OGBN_ARXIV_NUM_NEIGHBORS",
]