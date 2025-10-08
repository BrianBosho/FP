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
    def __init__(self, clients, model, device, cfg=None) -> None:
        self.DEVICE = device
        self.device = self.DEVICE
        self.clients = clients
        self.model = model.to(self.device)
        self.num_of_trainers = len(clients)
        self.cfg = cfg or {}

        # Config-driven minibatch settings
        self.use_minibatch = bool(self.cfg.get("use_minibatch", False))
        self.auto_minibatch_if_large = bool(self.cfg.get("auto_minibatch_if_large", False))
        self.batch_size = self.cfg.get("batch_size", DEFAULT_BATCH_SIZE)
        self.num_neighbors = self.cfg.get("num_neighbors", DEFAULT_NUM_NEIGHBORS)
        
        # Bounded concurrency: train K clients at a time to limit peak GPU memory
        # If None or 0, train all clients in parallel (original behavior)
        self.max_concurrent_clients = self.cfg.get("max_concurrent_clients", None)
        if self.max_concurrent_clients == 0:
            self.max_concurrent_clients = None

        # print input dim of the model
        if hasattr(self, 'cfg') and self.cfg.get("debug", False):
            print(f"Input dim of the model in server: {self.model.dim_in}")

        # update the client params
        self.broadcast_params(-1)
    
    def _train_clients_batched(self, clients, batch_size):
        """Train clients in batches to limit peak GPU memory usage"""
        import math
        all_results = []
        num_batches = math.ceil(len(clients) / batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(clients))
            batch_clients = clients[start_idx:end_idx]
            
            print(f"  Batch {batch_idx + 1}/{num_batches}: Training clients {start_idx} to {end_idx - 1}")
            
            # Train this batch of clients
            batch_futures = [client.train_client.remote() for client in batch_clients]
            batch_results = ray.get(batch_futures)
            all_results.extend(batch_results)
            
            # Clear CUDA cache between batches
            torch.cuda.empty_cache()
            gc.collect()
        
        return all_results
        
    @torch.no_grad()
    def train_clients(self, current_global_epoch: int) -> list:
        clients = self.clients
        
        # Bounded concurrency: train K clients at a time
        if self.max_concurrent_clients and self.max_concurrent_clients < len(clients):
            print(f"Training clients in batches of {self.max_concurrent_clients} (total: {len(clients)})")
            # CRITICAL: Ensure all clients have received broadcast params before batched training
            # In parallel mode, Ray handles this automatically, but in batched mode we need explicit sync
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            train_results = self._train_clients_batched(clients, self.max_concurrent_clients)
        else:
            # Original behavior: train all clients in parallel
            train_futures = [client.train_client.remote() for client in clients]
            train_results = ray.get(train_futures)

        params = [client.get_params.remote() for client in clients]
        self.zero_params()
        self.zero_buffers()
        
        while True:
            ready, left = ray.wait(params, num_returns=1, timeout=None)
            if ready:
                for t in ready:
                    params_dict = ray.get(t)
                    # Aggregate parameters
                    for p, mp in zip(params_dict['params'], self.model.parameters()):
                        mp.data += p.to(self.device)
                    # Aggregate buffers (e.g., BatchNorm running stats)
                    # Only aggregate floating point buffers (skip num_batches_tracked, etc.)
                    for b, mb in zip(params_dict['buffers'], self.model.buffers()):
                        if b.dtype.is_floating_point:
                            mb.data += b.to(self.device)
                        else:
                            # For non-float buffers (like num_batches_tracked), just copy from first client
                            if self.num_of_trainers == 1 or mb.sum() == 0:
                                mb.data = b.to(self.device)
            params = left
            if not params:
                break

        # Average parameters
        for p in self.model.parameters():
             p.data /= self.num_of_trainers
        
        # Average buffers (only floating point ones)
        for b in self.model.buffers():
            if b.dtype.is_floating_point:
                b.data /= self.num_of_trainers
        
        # Broadcast params with sync=True when using batched training to ensure consistency
        use_sync = self.max_concurrent_clients and self.max_concurrent_clients < len(clients)
        self.broadcast_params(current_global_epoch, sync=use_sync)

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
    
    def broadcast_params(self, current_global_epoch: int, sync=False) -> None:
        params_dict = {
            'params': tuple(self.model.parameters()),
            'buffers': tuple(self.model.buffers())
        }
        futures = []
        for trainer in self.clients:
            future = trainer.update_params.remote(params_dict, current_global_epoch)
            if sync:
                futures.append(future)
        
        # If sync=True, wait for all updates to complete before returning
        if sync:
            ray.get(futures)

    @torch.no_grad()
    def zero_params(self) -> None:
        for p in self.model.parameters():
            p.zero_()
    
    @torch.no_grad()
    def zero_buffers(self) -> None:
        for b in self.model.buffers():
            if b.dtype.is_floating_point:
                b.zero_()

    def evaluate_global_model(self, data, criterion):
        self.model.to(self.device)
        data = data.to(self.device)
        
        # Determine minibatch usage from config with optional auto fallback
        use_minibatch = self.use_minibatch
        if not use_minibatch and self.auto_minibatch_if_large:
            if hasattr(data, 'x') and data.x.shape[0] > LARGE_DATASET_THRESHOLD:
                use_minibatch = True
        
        if use_minibatch:
            return evaluate_with_minibatch(self.model, data, criterion, batch_size=self.batch_size, num_neighbors=self.num_neighbors)
        else:
            return evaluate(self.model, data, criterion)
    
    def test_global_model(self, data):
        # Clear memory before testing
        torch.cuda.empty_cache()
        gc.collect()

        # check the input dim of the model being tested
        if hasattr(self, 'cfg') and self.cfg.get("debug", False):
            print(f"Input dim of the model in server: {self.model.dim_in}")
            # check the input dim of the data
            print(f"Input dim of the test data in server: {data.x.shape[1]}")
        
        self.model.to(self.device)
        
        # Don't move the entire data to device at once
        # data = data.to(self.device)
        # return 0
        
        # Determine minibatch usage from config with optional auto fallback
        use_minibatch = self.use_minibatch
        if not use_minibatch and self.auto_minibatch_if_large:
            if hasattr(data, 'x') and data.x.shape[0] > LARGE_DATASET_THRESHOLD:
                use_minibatch = True
                print(f"Server: Using mini-batch testing for large dataset with {data.x.shape[0]} nodes")
        
        if use_minibatch:
            return test_with_minibatch(self.model, data, batch_size=self.batch_size, num_neighbors=self.num_neighbors)
        else:
            # Only for small datasets, move to device
            data = data.to(self.device)
            return test(self.model, data)
