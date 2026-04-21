import ray
import torch
from src.train import evaluate, test, evaluate_with_minibatch, test_with_minibatch
from src.utils.wandb_utils import initialize_wandb, log_client_training_metrics, log_client_validation_metrics, log_final_validation_metrics, log_test_metrics
import gc
import numpy as np
from src.utils.memory_utils import (
    clear_memory_basic, clear_memory_aggressive, 
    log_memory_usage
)
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

        # B1: aggregation strategy.  Default "mean" preserves the previous
        # simple-average behavior exactly; "fedavg_weighted" weights by the
        # number of training samples per client, which is the FedAvg rule
        # from McMahan et al.  Any unrecognized value falls back to "mean".
        self.aggregation = str(self.cfg.get("aggregation", "mean")).lower()
        if self.aggregation not in ("mean", "fedavg_weighted"):
            print(f"[Server] Unknown aggregation={self.aggregation!r}; falling back to 'mean'.")
            self.aggregation = "mean"
        print(f"[Server] Aggregation strategy: {self.aggregation}")

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
            
            print(f"  Batch {batch_idx + 1}/{num_batches}: Training clients {start_idx} to {end_idx - 1} (parallel)")
            
            # Train this batch of clients IN PARALLEL
            # The key is that we only submit batch_size tasks at once
            batch_futures = [client.train_client.remote() for client in batch_clients]
            batch_results = ray.get(batch_futures)
            all_results.extend(batch_results)
            
            # Memory clearing between batches removed for performance
        
        return all_results
        
    @torch.no_grad()
    def _aggregate_mean(self, clients) -> None:
        """Legacy unweighted-mean aggregation.  Preserved byte-for-byte so runs
        without ``aggregation: fedavg_weighted`` behave exactly as before."""
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

                    # Memory optimization: Explicitly delete params_dict after use
                    # to free GPU memory immediately after aggregation
                    del params_dict
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

    @torch.no_grad()
    def _aggregate_fedavg_weighted(self, clients) -> None:
        """FedAvg-style weighted aggregation.

        Each client's parameters are scaled by ``|D_k| / sum_k |D_k|`` where
        ``|D_k|`` is the number of training samples that client owns.  This is
        the aggregation rule from McMahan et al. (2017) -- the existing
        ``_aggregate_mean`` above collapses to this only when every client
        has the same sample count.

        If every client reports 0 training samples (pathological, shouldn't
        happen in practice), we fall back to uniform weights to avoid
        division-by-zero, matching the legacy behavior in that degenerate case.

        Buffers (e.g. BatchNorm running stats) are weighted with the same
        client weights for floating-point buffers; non-float buffers (e.g.
        ``num_batches_tracked``) retain the legacy "copy-from-first" rule.
        """
        # Fetch sample counts in deterministic client order.
        sample_counts = ray.get([c.get_num_train_samples.remote() for c in clients])
        total = int(sum(sample_counts))
        if total <= 0:
            weights = [1.0 / len(clients)] * len(clients)
            print("[Server] fedavg_weighted: all clients report 0 samples; using uniform weights.")
        else:
            weights = [float(n) / float(total) for n in sample_counts]

        if bool(self.cfg.get("debug", False)):
            print(f"[Server] fedavg_weighted: sample_counts={sample_counts}, "
                  f"weights={[round(w, 4) for w in weights]}")

        # Map each param-future back to its client weight so we can keep the
        # streaming ray.wait loop (memory-efficient for large models / many
        # clients) while still applying the correct per-client weight.
        param_futures = [c.get_params.remote() for c in clients]
        weight_by_future = {fut: w for fut, w in zip(param_futures, weights)}

        self.zero_params()
        self.zero_buffers()

        remaining = list(param_futures)
        while remaining:
            ready, remaining = ray.wait(remaining, num_returns=1, timeout=None)
            for t in ready:
                w = weight_by_future[t]
                params_dict = ray.get(t)

                for p, mp in zip(params_dict['params'], self.model.parameters()):
                    mp.data += w * p.to(self.device)

                for b, mb in zip(params_dict['buffers'], self.model.buffers()):
                    if b.dtype.is_floating_point:
                        mb.data += w * b.to(self.device)
                    else:
                        if self.num_of_trainers == 1 or mb.sum() == 0:
                            mb.data = b.to(self.device)

                del params_dict
        # No post-division: weights already sum to 1.

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
            
            # Memory clearing removed for performance (happens naturally with Python GC)

        if self.aggregation == "fedavg_weighted":
            self._aggregate_fedavg_weighted(clients)
        else:
            self._aggregate_mean(clients)
        
        # Broadcast params with sync=True when using batched training to ensure consistency
        use_sync = self.max_concurrent_clients and self.max_concurrent_clients < len(clients)
        self.broadcast_params(current_global_epoch, sync=use_sync)
        
        # Memory clearing removed for performance (happens naturally with Python GC)

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
        
        # Use batched evaluation based on max_concurrent_clients setting
        if self.max_concurrent_clients and self.max_concurrent_clients < len(clients):
            results = self._evaluate_clients_batched(clients, criterion, self.max_concurrent_clients)
        else:
            # Original parallel evaluation for small number of clients
            eval_futures = [client.evaluate.remote(criterion) for client in clients]
            results = ray.get(eval_futures)
        
        # log_final_validation_metrics(results, -1)  # -1 for global evaluation
        return results
    
    def _evaluate_clients_batched(self, clients, criterion, batch_size):
        """Evaluate clients in batches to limit peak GPU memory usage"""
        import math
        all_results = []
        num_batches = math.ceil(len(clients) / batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(clients))
            batch_clients = clients[start_idx:end_idx]
            
            print(f"  Evaluation Batch {batch_idx + 1}/{num_batches}: Evaluating clients {start_idx} to {end_idx - 1} (parallel)")
            
            # Evaluate this batch of clients IN PARALLEL
            batch_futures = [client.evaluate.remote(criterion) for client in batch_clients]
            batch_results = ray.get(batch_futures)
            all_results.extend(batch_results)
            
            # Memory clearing between batches removed for performance
        
        print(f"Completed batched evaluation of {len(clients)} clients in {num_batches} batches")
        return all_results
    
    def test_clients_batched(self, test_data_list, batch_size):
        """Test clients in batches to limit GPU memory usage"""
        import math
        results = []
        num_batches = math.ceil(len(self.clients) / batch_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.clients))
            
            print(f"  Test Batch {batch_idx + 1}/{num_batches}: Testing clients {start_idx} to {end_idx - 1} (parallel)")
            
            # Test this batch IN PARALLEL
            batch_futures = [
                self.clients[i].test.remote(test_data_list[i]) 
                for i in range(start_idx, end_idx)
            ]
            batch_results = ray.get(batch_futures)
            results.extend(batch_results)
            
            # Clear memory between batches
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
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
