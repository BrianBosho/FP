"""Legacy wrapper — real implementation is in `src.fedgnn.fl.train`."""

from src.fedgnn.fl.train import (  # noqa: F401
    train,
    evaluate,
    test,
    train_with_minibatch,
    evaluate_with_minibatch,
    test_with_minibatch,
    set_seed,
)
