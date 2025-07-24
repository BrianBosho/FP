import wandb
import numpy as np

def to_cpu_scalar(x):
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().item()
    return x

def initialize_wandb(project= "FGL", entity=None, config=None, name=None, dir=None, mode="online", resume=None, group=None):
    import wandb    
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

def log_client_training_metrics(train_results: list, current_global_epoch: int) -> None:
    """Log client training metrics"""
    client_losses = [to_cpu_scalar(result[0]) for result in train_results]
    client_accuracies = [to_cpu_scalar(result[1]) for result in train_results]
        
    # Calculate mean and standard deviation
    mean_loss = np.mean(client_losses)
    std_loss = np.std(client_losses)
    mean_acc = np.mean(client_accuracies)
    std_acc = np.std(client_accuracies)

    # Log to wandb
    wandb.log({
        "round": current_global_epoch,
        "federation/avg_client_train_loss": mean_loss,
        "federation/avg_client_train_acc": mean_acc,
        # "federation/client_train_loss_std": std_loss,
        # "federation/client_train_acc_std": std_acc,
        # "federation/num_participating_clients": len(client_losses),
        # Client diversity metrics (normalized std)
        # "federation/client_diversity_loss": std_loss / (mean_loss + 1e-8),
        # "federation/client_diversity_acc": std_acc / (mean_acc + 1e-8),
    }, step=current_global_epoch)

def log_client_validation_metrics(val_results: list, current_global_epoch: int) -> None:
    client_val_losses = [to_cpu_scalar(result[0]) for result in val_results]
    client_val_accuracies = [to_cpu_scalar(result[1]) for result in val_results]
    mean_val_loss = np.mean(client_val_losses)
    std_val_loss = np.std(client_val_losses)
    mean_val_acc = np.mean(client_val_accuracies)
    std_val_acc = np.std(client_val_accuracies)
    wandb.log({
        "round": current_global_epoch,
        "federation/avg_client_val_loss": mean_val_loss,
        "federation/avg_client_val_acc": mean_val_acc,
        # "federation/client_val_loss_std": std_val_loss,
        # "federation/client_val_acc_std": std_val_acc,
    }, step=current_global_epoch)

def log_final_validation_metrics(val_results: list, current_global_epoch: int) -> None:
    client_val_losses = [to_cpu_scalar(result[0]) for result in val_results]
    client_val_accuracies = [to_cpu_scalar(result[1]) for result in val_results]
    mean_val_loss = np.mean(client_val_losses)
    std_val_loss = np.std(client_val_losses)
    mean_val_acc = np.mean(client_val_accuracies)
    std_val_acc = np.std(client_val_accuracies)
    wandb.log({
        "round": current_global_epoch,
        "federation/final_val_loss": mean_val_loss,
        "federation/final_val_acc": mean_val_acc,
        # "federation/client_val_loss_std": std_val_loss,
        # "federation/client_val_acc_std": std_val_acc,
    }, step=current_global_epoch)
    
    
