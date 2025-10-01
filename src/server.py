import ray
import torch
from src.train import evaluate, test, evaluate_with_minibatch, test_with_minibatch
from src.utils.wandb_utils import initialize_wandb, log_client_training_metrics, log_client_validation_metrics, log_final_validation_metrics, log_test_metrics
import gc
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import wandb


LARGE_DATASET_THRESHOLD = 100000  # Number of nodes threshold for large datasets
DEFAULT_BATCH_SIZE = 2048  # Batch size for large datasets
DEFAULT_NUM_NEIGHBORS = [25, 25]  # Increased from 10 to reduce variance while avoiding OOM 


class Server():
    def __init__(self, clients, model, device) -> None:
        self.DEVICE = device
        self.device = self.DEVICE
        self.clients = clients
        self.model = model.to(self.device)
        self.num_of_trainers = len(clients)

        # print input dim of the model
        print(f"Input dim of the model in server: {self.model.dim_in}")

        # update the client params
        self.broadcast_params(-1)
        
    @torch.no_grad()
    def train_clients(self, current_global_epoch: int) -> list:
        clients = self.clients
        train_futures = [client.train_client.remote() for client in clients]
        train_results = ray.get(train_futures)

        params = [client.get_params.remote() for client in clients]
        self.zero_params()
        while True:
            ready, left = ray.wait(params, num_returns=1, timeout=None)
            if ready:
                for t in ready:
                    for p, mp in zip(ray.get(t), self.model.parameters()):
                        mp.data += p.to(self.device)
            params = left
            if not params:
                break

        for p in self.model.parameters():
             p.data /= self.num_of_trainers
        self.broadcast_params(current_global_epoch)

        log_client_training_metrics(train_results, current_global_epoch)
        # lets run evaluation after training
        eval_results = self.evaluate_clients()
        log_client_validation_metrics(eval_results, current_global_epoch)

        def to_cpu_scalar(x):
            if hasattr(x, "detach") and hasattr(x, "cpu"):
                return x.detach().cpu().item()
            return x
        client_val_losses = [to_cpu_scalar(result[0]) for result in eval_results]
        client_val_accuracies = [to_cpu_scalar(result[1]) for result in eval_results]
        avg_eval_loss = np.mean(client_val_losses)
        avg_eval_acc = np.mean(client_val_accuracies)

        return train_results, avg_eval_acc, avg_eval_loss

    def evaluate_clients(self):
        clients = self.clients
        criterion = torch.nn.NLLLoss()
        eval_futures = [client.evaluate.remote(criterion) for client in clients]
        results = ray.get(eval_futures)
        # log_final_validation_metrics(results, -1)  # -1 for global evaluation
        return results
    
    def broadcast_params(self, current_global_epoch: int) -> None:
        for trainer in self.clients:
            trainer.update_params.remote(
                tuple(self.model.parameters()), current_global_epoch
            )  # run in submit order

    @torch.no_grad()
    def zero_params(self) -> None:
        for p in self.model.parameters():
            p.zero_()

    def evaluate_global_model(self, data, criterion):
        self.model.to(self.device)
        data = data.to(self.device)
        
        # Check if this is a large dataset that requires mini-batching
        use_minibatch = False
        if hasattr(data, 'x') and data.x.shape[0] > 100000:  # LARGE_DATASET_THRESHOLD from client.py
            use_minibatch = True
        
        # Always use mini-batching for ogbn datasets
        dataset_name = data.name if hasattr(data, 'name') else "unknown"
        if dataset_name == "ogbn-arxiv" or dataset_name == "ogbn-products":
            use_minibatch = True
        
        if use_minibatch:
            return evaluate_with_minibatch(self.model, data, criterion, batch_size=DEFAULT_BATCH_SIZE, num_neighbors=DEFAULT_NUM_NEIGHBORS)
        else:
            return evaluate(self.model, data, criterion)
    
    def test_global_model(self, data):
        # Clear memory before testing
        torch.cuda.empty_cache()
        gc.collect()

        # check the input dim of the model being tested
        print(f"Input dim of the model in server: {self.model.dim_in}")
        # check the input dim of the data
        print(f"Input dim of the test data in server: {data.x.shape[1]}")
        
        self.model.to(self.device)
        
        # Don't move the entire data to device at once
        # data = data.to(self.device)
        # return 0
        
        # Check if this is a large dataset that requires mini-batching
        use_minibatch = False
        if hasattr(data, 'x') and data.x.shape[0] > 100000:  # LARGE_DATASET_THRESHOLD from client.py
            use_minibatch = True
            print(f"Server: Using mini-batch testing for large dataset with {data.x.shape[0]} nodes")
        
        # Always use mini-batching for ogbn datasets
        dataset_name = data.name if hasattr(data, 'name') else "unknown"
        if dataset_name == "ogbn-arxiv" or dataset_name == "ogbn-products":
            use_minibatch = True
            print(f"Server: Using mini-batch testing for {dataset_name} dataset")
        
        if use_minibatch:
            return test_with_minibatch(self.model, data, batch_size=DEFAULT_BATCH_SIZE, num_neighbors=DEFAULT_NUM_NEIGHBORS)
        else:
            # Only for small datasets, move to device
            data = data.to(self.device)
            return test(self.model, data)
