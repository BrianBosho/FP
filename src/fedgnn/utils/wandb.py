import wandb
import numpy as np


def to_cpu_scalar(x):
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().item()
    return x


def initialize_wandb(
    project="FGL",
    entity=None,
    config=None,
    name=None,
    dir=None,
    mode="online",
    resume=None,
    group=None,
    use_wandb=True,
):
    # Check if wandb should be used
    if not use_wandb:
        # Silently skip wandb initialization when disabled
        return

    # Check if there's already an active run and finish it
    if wandb.run is not None:
        print(
            f"[WANDB DEBUG] Found existing run, finishing it before starting new one"
        )
        wandb.finish()

    try:
        wandb.init(
            project=project,
            entity=entity,
            config=config,
            name=name,
            dir=dir,
            mode=mode,
            resume=resume,
            group=group,
        )
        print(
            f"[WANDB DEBUG] Successfully initialized wandb run: {wandb.run.name if wandb.run else 'Unknown'}"
        )
    except Exception as e:
        print(f"[WANDB ERROR] Failed to initialize wandb: {e}")
        import traceback

        traceback.print_exc()


def log_client_training_metrics(
    train_results: list, current_global_epoch: int
) -> None:
    """Log client training metrics"""
    # Check if wandb is initialized and enabled
    if wandb.run is None:
        return

    client_losses = [to_cpu_scalar(result[0]) for result in train_results]
    client_accuracies = [to_cpu_scalar(result[1]) for result in train_results]

    # Calculate mean and standard deviation
    mean_loss = np.mean(client_losses)
    std_loss = np.std(client_losses)
    mean_acc = np.mean(client_accuracies)
    std_acc = np.std(client_accuracies)

    # Log to wandb
    wandb.log(
        {
            "round": current_global_epoch,
            "federation/avg_client_train_loss": mean_loss,
            "federation/avg_client_train_acc": mean_acc,
        },
        step=current_global_epoch,
    )


def log_client_validation_metrics(
    val_results: list, current_global_epoch: int
) -> None:
    # Check if wandb is initialized and enabled
    if wandb.run is None:
        return

    client_val_losses = [to_cpu_scalar(result[0]) for result in val_results]
    client_val_accuracies = [to_cpu_scalar(result[1]) for result in val_results]
    mean_val_loss = np.mean(client_val_losses)
    std_val_loss = np.std(client_val_losses)
    mean_val_acc = np.mean(client_val_accuracies)
    std_val_acc = np.std(client_val_accuracies)
    wandb.log(
        {
            "round": current_global_epoch,
            "federation/avg_client_val_loss": mean_val_loss,
            "federation/avg_client_val_acc": mean_val_acc,
        },
        step=current_global_epoch,
    )


def log_final_validation_metrics(
    val_results: list, current_global_epoch: int
) -> None:
    # Check if wandb is initialized and enabled
    if wandb.run is None:
        return

    client_val_losses = [to_cpu_scalar(result[0]) for result in val_results]
    client_val_accuracies = [to_cpu_scalar(result[1]) for result in val_results]
    mean_val_loss = np.mean(client_val_losses)
    std_val_loss = np.std(client_val_losses)
    mean_val_acc = np.mean(client_val_accuracies)
    std_val_acc = np.std(client_val_accuracies)
    wandb.log(
        {
            "round": current_global_epoch,
            "federation/final_val_loss": mean_val_loss,
            "federation/final_val_acc": mean_val_acc,
        },
        step=current_global_epoch,
    )


def log_test_metrics(
    global_test_acc: float,
    client_test_accs: list,
    current_global_epoch: int = -1,
) -> None:
    """
    Log test metrics for both global model and individual clients

    Args:
        global_test_acc (float): Global model test accuracy
        client_test_accs (list): List of client test accuracies
        current_global_epoch (int): Current global epoch (-1 for final test results)
    """
    # Convert to CPU scalars if needed
    global_test_acc = to_cpu_scalar(global_test_acc)
    client_test_accs = [to_cpu_scalar(acc) for acc in client_test_accs]

    # Validate inputs
    if not isinstance(client_test_accs, list) or len(client_test_accs) == 0:
        print(f"[WANDB ERROR] Invalid client_test_accs: {client_test_accs}")
        return

    # Calculate client test statistics
    mean_client_test_acc = np.mean(client_test_accs)
    std_client_test_acc = np.std(client_test_accs)

    # Use a valid step value (ensure it's non-negative)
    step_value = max(0, current_global_epoch) if current_global_epoch >= 0 else 0

    # Log to wandb
    metrics = {
        "round": current_global_epoch,
        "test/global_test_acc": global_test_acc,
        "test/avg_client_test_acc": mean_client_test_acc,
        "test/client_test_acc_std": std_client_test_acc,
        "test/num_clients": len(client_test_accs),
        "test/global_vs_avg_client_diff": global_test_acc - mean_client_test_acc,
    }

    # Check if wandb is initialized.  When use_wandb=false, callers still reach
    # this helper in some paths; silently skip just like the other log helpers.
    if wandb.run is None:
        return

    try:
        wandb.log(metrics, step=step_value)
    except Exception as e:
        print(f"[WANDB ERROR] Failed to log test metrics: {e}")
        import traceback

        traceback.print_exc()
