"""Legacy wrapper — real implementation is in `src.fedgnn.fl.run`."""

from src.fedgnn.fl.run import (  # noqa: F401
    load_configuration,
    instantiate_model,
    initialize_clients,
    load_data,
    run_with_server,
    main_experiment,
    verify_test_masks,
)
