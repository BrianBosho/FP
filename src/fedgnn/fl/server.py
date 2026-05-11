import ray
import torch
from src.fedgnn.fl.train import evaluate, test, evaluate_with_minibatch, test_with_minibatch
from src.fedgnn.utils.wandb import initialize_wandb, log_client_training_metrics, log_client_validation_metrics, log_final_validation_metrics, log_test_metrics
import gc
import numpy as np
from src.fedgnn.utils.memory import (
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
        self.broadcast_params(-1, sync=True)

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

    def _is_bn_running_stat(self, name: str) -> bool:
        """Check if a buffer name belongs to a BatchNorm running-mean or running-var."""
        return "running_mean" in name or "running_var" in name

    def _apply_params_list(self, params_list, weights=None) -> None:
        """Accumulate a list of params dicts into the server model.

        weights: per-client float weights summing to 1.0 (fedavg_weighted),
                 or None for uniform 1/N mean (caller must divide afterwards).
        Shared by both aggregation paths to avoid code duplication.
        """
        fedbn = self.cfg.get("bn_fl_strategy", "average") == "fedbn"
        first_client = True
        for idx, params_dict in enumerate(params_list):
            w = weights[idx] if weights is not None else 1.0
            for p, mp in zip(params_dict['params'], self.model.parameters()):
                mp.data += w * p.to(self.device)
            buf_names = params_dict.get('buffer_names', [])
            for b, mb, name in zip(params_dict['buffers'], self.model.buffers(), buf_names):
                if b.dtype.is_floating_point:
                    if fedbn and self._is_bn_running_stat(name):
                        pass
                    else:
                        mb.data += w * b.to(self.device)
                elif first_client:
                    mb.data = b.to(self.device)
            first_client = False

    def _fetch_params_streaming(self, clients):
        """Fetch params from clients via streaming ray.wait (memory-safe for large models)."""
        futures = [c.get_params.remote() for c in clients]
        collected = {}
        remaining = list(futures)
        while remaining:
            ready, remaining = ray.wait(remaining, num_returns=1, timeout=None)
            for t in ready:
                collected[t] = ray.get(t)
        # Return in submission order so caller can zip with clients/weights.
        return futures, collected

    @torch.no_grad()
    def _aggregate_mean(self, clients, prefetched_params=None) -> None:
        """Unweighted-mean aggregation with optional FedBN support.

        prefetched_params: ordered list of params dicts already in memory
        (fuse_train_get_params path). None falls back to streaming ray.wait.
        """
        active_count = len(clients)
        self.zero_params()
        self.zero_buffers()

        if prefetched_params is not None:
            self._apply_params_list(prefetched_params, weights=None)
        else:
            futures, collected = self._fetch_params_streaming(clients)
            self._apply_params_list([collected[f] for f in futures], weights=None)

        for p in self.model.parameters():
            p.data /= active_count

        fedbn = self.cfg.get("bn_fl_strategy", "average") == "fedbn"
        buf_names = tuple(name for name, _ in self.model.named_buffers())
        for b, name in zip(self.model.buffers(), buf_names):
            if b.dtype.is_floating_point and not (fedbn and self._is_bn_running_stat(name)):
                b.data /= active_count

    @torch.no_grad()
    def _aggregate_fedavg_weighted(self, clients, prefetched_params=None) -> None:
        """FedAvg-style weighted aggregation (McMahan et al. 2017).

        Each client's parameters are scaled by |D_k| / sum_k |D_k|.
        num_train_samples is piggybacked in the params dict to avoid a
        separate get_num_train_samples.remote() pre-pass.

        prefetched_params: ordered list of params dicts already in memory
        (fuse_train_get_params path). None falls back to streaming ray.wait.
        """
        if prefetched_params is not None:
            params_list = prefetched_params
        else:
            futures, collected = self._fetch_params_streaming(clients)
            params_list = [collected[f] for f in futures]

        sample_counts = [p.get('num_train_samples', 0) for p in params_list]
        total = int(sum(sample_counts))
        if total <= 0:
            weights = [1.0 / len(clients)] * len(clients)
            print("[Server] fedavg_weighted: all clients report 0 samples; using uniform weights.")
        else:
            weights = [float(n) / float(total) for n in sample_counts]

        if bool(self.cfg.get("debug", False)):
            print(f"[Server] fedavg_weighted: sample_counts={sample_counts}, "
                  f"weights={[round(w, 4) for w in weights]}")

        self.zero_params()
        self.zero_buffers()
        self._apply_params_list(params_list, weights=weights)
        # No post-division: weights already sum to 1.

    @torch.no_grad()
    def train_clients(self, current_global_epoch: int) -> list:
        clients = self.clients

        # fuse_train_get_params: combine train + param fetch into one Ray call.
        # Disable for large datasets (big param tensors alongside training results
        # prevent streaming aggregation and stress the object store).
        # Ignored when using batched concurrency (batched path needs separate calls).
        fuse = bool(self.cfg.get("fuse_train_get_params", True))
        use_batched = bool(self.max_concurrent_clients and self.max_concurrent_clients < len(clients))

        if use_batched:
            try:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            except Exception:
                pass
            print(f"Training clients in batches of {self.max_concurrent_clients} (total: {len(clients)})")
            train_results = self._train_clients_batched(clients, self.max_concurrent_clients)
            prefetched_params = None
        elif fuse:
            combined_futures = [c.train_and_get_params.remote() for c in clients]
            combined = ray.get(combined_futures)
            # combined[i] = (val_loss, val_acc, success, params_dict | None)
            train_results = [(r[0], r[1], r[2]) for r in combined]
            prefetched_params = [r[3] for r in combined]
        else:
            train_futures = [c.train_client.remote() for c in clients]
            train_results = ray.get(train_futures)
            prefetched_params = None

        # Filter failed clients from aggregation
        active_clients, active_results, active_prefetched = [], [], []
        for idx, (client, result) in enumerate(zip(clients, train_results)):
            success = result[2] if len(result) > 2 else True
            if success:
                active_clients.append(client)
                active_results.append(result)
                if prefetched_params is not None:
                    active_prefetched.append(prefetched_params[idx])
            else:
                print(f"[Server] Client {idx} failed; excluding from aggregation")

        if len(active_clients) < len(clients):
            print(f"[Server] Aggregating {len(active_clients)}/{len(clients)} clients")

        if not active_clients:
            print("[Server] All clients failed this round; skipping aggregation")
            return train_results, 0.0, float('inf')

        pre = active_prefetched if active_prefetched else None
        if self.aggregation == "fedavg_weighted":
            self._aggregate_fedavg_weighted(active_clients, prefetched_params=pre)
        else:
            self._aggregate_mean(active_clients, prefetched_params=pre)

        use_sync = use_batched
        self.broadcast_params(current_global_epoch, sync=use_sync)

        log_client_training_metrics(active_results, current_global_epoch)

        def to_cpu_scalar(x):
            if hasattr(x, "detach") and hasattr(x, "cpu"):
                return x.detach().cpu().item()
            return x
        client_val_losses = [to_cpu_scalar(r[0]) for r in active_results]
        client_val_accs   = [to_cpu_scalar(r[1]) for r in active_results]
        avg_eval_loss = np.mean(client_val_losses)
        avg_eval_acc  = np.mean(client_val_accs)

        torch.cuda.empty_cache()
        gc.collect()

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
            'params': tuple(p.detach().cpu() for p in self.model.parameters()),
            'buffers': tuple(b.detach().cpu() for b in self.model.buffers()),
            'buffer_names': tuple(name for name, _ in self.model.named_buffers()),
        }
        # Single serialisation into the plasma store; all actors get a
        # zero-copy memory-mapped reference instead of N separate pickles.
        params_ref = ray.put(params_dict)
        futures = []
        for trainer in self.clients:
            future = trainer.update_params.remote(params_ref, current_global_epoch)
            if sync:
                futures.append(future)

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