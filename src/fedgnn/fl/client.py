import ray
from src.fedgnn.fl.train import train, evaluate, test, train_with_minibatch, evaluate_with_minibatch, test_with_minibatch
from src.fedgnn.models import GCN, GAT, VanillaGNN, MLP, GCN_arxiv, GraphSAGEProducts, PubmedGAT, GAT_Arxiv
import torch
import sys
import gc
from src.fedgnn.data.shard_cache import is_shard_ref
from src.fedgnn.utils.memory import (
    clear_memory_basic, clear_memory_aggressive,
    clear_memory_for_diffusion, clear_memory_for_adjacency,
    log_memory_usage, memory_guard
)

# Configure batch sizes based on dataset size
LARGE_DATASET_THRESHOLD = 100000  # Number of nodes threshold for large datasets
DEFAULT_BATCH_SIZE = 1024  # Default batch size for most datasets
DEFAULT_NUM_NEIGHBORS = [10, 10, 10]

# Special configurations for specific datasets that need more careful handling
OGBN_ARXIV_BATCH_SIZE = 2048 # Smaller batch size for ogbn-arxiv
OGBN_ARXIV_NUM_NEIGHBORS = [25, 25]  # Increased from 10 to reduce variance while avoiding OOM

# Default to no GPU reservation; src.run.initialize_clients applies a
# config-driven num_gpus override via FLClient.options(...).
@ray.remote(num_gpus=0)
class FLClient:
    def __init__(self, data, dataset, client_id, cfg, device, model_type="GCN"):
        # Clear any existing CUDA cache and perform garbage collection
        torch.cuda.empty_cache()
        gc.collect()

        self.cfg = cfg
        if is_shard_ref(data):
            print(f"[Client {client_id}] Loading shard from {data.path}")
            data = data.load()

        # Resolve device string lazily: Ray workers may not see CUDA at
        # deserialization time even if the driver did.  Check again here.
        _device_str = str(device)
        import os
        print(f"[Client {client_id}] CVD={os.environ.get('CUDA_VISIBLE_DEVICES','?')} "
              f"torch.cuda.is_available={torch.cuda.is_available()} "
              f"torch.cuda.device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
              f"requested_device={_device_str}")
        if _device_str in ("cuda", "cuda:0") and not torch.cuda.is_available():
            print(f"[Client {client_id}] WARNING: CUDA requested but not available in worker, falling back to CPU")
            _device_str = "cpu"
        self.DEVICE = _device_str
        self.target_device = torch.device(_device_str)
        self.cpu_device = torch.device("cpu")
        self.current_device = self.cpu_device
        self.data_gpu = None

        # Get GPU memory optimization flag from config
        self.keep_data_on_gpu = cfg.get("keep_data_on_gpu", True)

        # Large-dataset guard: ogbn-products stays on CPU regardless
        self._large_dataset = False

        # Get debug flag from config
        debug = cfg.get("debug", False)

        # LOG: What data did this client receive?
        if debug:
            print(f"\n[Client {client_id}] Initializing with:")
            print(f"  - Input data nodes: {data.num_nodes}")
            print(f"  - Input data edges: {data.edge_index.shape[1] if hasattr(data, 'edge_index') else 'N/A'}")
            print(f"  - Feature shape: {data.x.shape}")
            print(f"  - Data device (before moving): {data.x.device}")

        # Keep data on its incoming device; CPU copy only used for device-swap path
        self.data_cpu = data.to(self.cpu_device)
        self.data = self.data_cpu
        self.dataset = dataset
        self.dataset_name = dataset.name if hasattr(dataset, 'name') else "unknown"

        # get input dim from data
        self.input_dim = data.x.shape[1]
        if debug:
            print(f"  - Input dim: {self.input_dim}")
            print(f"  - Data device (after moving): {self.data.x.device}")

        # Determine training mode from config (preferred) or fallback
        if "use_minibatch" in cfg:
            self.use_minibatch = bool(cfg.get("use_minibatch"))
        else:
            self.use_minibatch = False

        # Optional: auto-enable mini-batch for very large graphs if requested
        if not self.use_minibatch and bool(cfg.get("auto_minibatch_if_large", False)):
            if hasattr(self.data, 'x') and self.data.x.shape[0] > LARGE_DATASET_THRESHOLD:
                self.use_minibatch = True

        print(f"Client {client_id}: Using {'mini-batch' if self.use_minibatch else 'full-batch'} training for {self.dataset_name} dataset")

        # Import get_model_config for consistent model configuration
        from src.fedgnn.models import get_model_config

        # Get model configuration from config file
        model_config = get_model_config(cfg, model_type, self.dataset_name)

        # Always init model on CPU; _move_for_compute() moves it to target_device
        # just before training. This avoids concurrent CUDA context initialisation
        # across 10 Ray workers hitting the vGPU driver simultaneously.
        _init_device = self.cpu_device
        if self.dataset_name == "ogbn-products":
            self._large_dataset = True

        if model_type == "GCN":
            if self.dataset_name == "ogbn-arxiv":
                use_unified_gcn = model_config.get('use_unified_model', True)
                if use_unified_gcn:
                    self.model = GCN(
                        self.input_dim,
                        model_config.get('hidden_dim', 256),
                        dataset.num_classes,
                        num_layers=model_config.get('num_layers', 3),
                        dropout=model_config.get('dropout', 0.5),
                        normalization=model_config.get('normalization', 'batch')
                    ).to(_init_device)
                else:
                    self.model = GCN_arxiv(
                        input_dim=self.input_dim,
                        hidden_dim=model_config.get('hidden_dim', 256),
                        output_dim=dataset.num_classes,
                        dropout=model_config.get('dropout', 0.5),
                        num_layers=model_config.get('num_layers', 3),
                        normalization=model_config.get('normalization', 'batch')
                    ).to(_init_device)
            elif self.dataset_name == "ogbn-products":
                products_input_dim = getattr(dataset, "num_features", None) or self.input_dim
                self.model = GraphSAGEProducts(
                    input_dim=products_input_dim,
                    hidden_dim=model_config.get('hidden_dim', 256),
                    output_dim=dataset.num_classes,
                    dropout=model_config.get('dropout', 0.5),
                    num_layers=model_config.get('num_layers', 3)
                ).to(self.cpu_device)
            else:
                self.model = GCN(
                    self.input_dim,
                    model_config.get('hidden_dim', 16),
                    dataset.num_classes,
                    num_layers=model_config.get('num_layers', 2),
                    dropout=model_config.get('dropout', 0.5),
                    normalization=model_config.get('normalization', 'none')
                ).to(_init_device)

        elif model_type == "GCN_arxiv":
            self.model = GCN_arxiv(
                input_dim=self.input_dim,
                hidden_dim=model_config.get('hidden_dim', 256),
                output_dim=dataset.num_classes,
                dropout=model_config.get('dropout', 0.5),
                num_layers=model_config.get('num_layers', 3),
                normalization=model_config.get('normalization', 'batch')
            ).to(_init_device)

        elif model_type == "GAT":
            if self.dataset_name == "ogbn-arxiv":
                use_unified_gat = model_config.get('use_unified_model', False)
                if use_unified_gat:
                    self.model = GAT(
                        self.input_dim,
                        model_config.get('hidden_dim', 256),
                        dataset.num_classes,
                        heads=model_config.get('num_heads', 8),
                        dropout=model_config.get('dropout', 0.5),
                        num_layers=model_config.get('num_layers', 3),
                        normalization=model_config.get('normalization', 'batch')
                    ).to(_init_device)
                else:
                    self.model = GAT_Arxiv(
                        input_dim=self.input_dim,
                        hidden_dim=model_config.get('hidden_dim', 256),
                        output_dim=dataset.num_classes,
                        dropout=model_config.get('dropout', 0.5),
                        num_layers=model_config.get('num_layers', 3),
                        normalization=model_config.get('normalization', 'batch'),
                        heads_hidden=model_config.get('heads_hidden', 4),
                        heads_out=model_config.get('heads_out', 6)
                    ).to(_init_device)
            elif self.dataset_name == "Pubmed":
                self.model = PubmedGAT(
                    self.input_dim,
                    model_config.get('hidden_dim', 8),
                    dataset.num_classes,
                    heads=model_config.get('num_heads', 8),
                    dropout=model_config.get('dropout', 0.6),
                    num_layers=model_config.get('num_layers', 2),
                    normalization=model_config.get('normalization', 'none')
                ).to(_init_device)
            else:
                self.model = GAT(
                    self.input_dim,
                    model_config.get('hidden_dim', 8),
                    dataset.num_classes,
                    heads=model_config.get('num_heads', 8),
                    dropout=model_config.get('dropout', 0.6),
                    num_layers=model_config.get('num_layers', 2),
                    normalization=model_config.get('normalization', 'none')
                ).to(_init_device)

        self.epochs = cfg["epochs"]

        # Setup batch configuration from config (used only if use_minibatch=True)
        self.batch_size = cfg.get("batch_size", DEFAULT_BATCH_SIZE)
        self.num_neighbors = cfg.get("num_neighbors", DEFAULT_NUM_NEIGHBORS)

        # Setup mixed precision training
        self.use_amp = bool(cfg.get("use_amp", False) and self.target_device.type == "cuda")
        self.grad_clip_norm = cfg.get("grad_clip_norm", 1.0)
        # A3: Structural consistency regularization
        self.struct_reg_lambda = cfg.get("struct_reg_lambda", 0.0)
        self.struct_reg_warmup_rounds = cfg.get("struct_reg_warmup_rounds", 0)

        optimizer_type = cfg["optimizer"]
        lr = cfg["lr"]
        weight_decay = cfg["decay"]

        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )

        if optimizer_type == "SGD":
            momentum = cfg.get("momentum", 0)
            if momentum > 0:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=lr, weight_decay=weight_decay,
                    momentum=momentum
                )
            else:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=lr, weight_decay=weight_decay
                )


        self.criterion = torch.nn.NLLLoss()
        self.writer = None

        # list of training losses & accuracies
        self.training_losses = []
        self.training_accuracies = []
        self.validation_losses = []
        self.client_id = client_id

        # C5: experiment-level seed (opt-in).  None -> preserve legacy
        # unseeded/42-hardcoded training behavior.  Duck-typed .get so this
        # works for both plain dicts and OmegaConf DictConfigs.
        try:
            _es = cfg.get("experiment_seed", None) if hasattr(cfg, "get") else None
            self.experiment_seed = None if _es is None else int(_es)
        except (TypeError, ValueError):
            self.experiment_seed = None

    def _move_optimizer_state(self, device):
        """Move optimizer state tensors to the requested device."""
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _move_to_device(self, device):
        """Move model/data between CPU and the target compute device on demand."""
        device = torch.device(device)
        if device == self.current_device:
            return

        if device == self.cpu_device:
            # Return everything to CPU and release GPU memory
            self.model.to(self.cpu_device)
            self._move_optimizer_state(self.cpu_device)
            self.data = self.data_cpu
            self.data_gpu = None
            self.current_device = self.cpu_device
            torch.cuda.empty_cache()
            gc.collect()
            return

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")

        # Move to GPU and keep a transient copy of the data there
        self.model.to(device)
        self._move_optimizer_state(device)
        self.data_gpu = self.data_cpu.to(device)
        self.data = self.data_gpu
        self.current_device = device

    def _move_for_compute(self, device):
        """Move state for a train/eval call.

        Full-batch training needs the whole client graph on the compute device.
        NeighborLoader training keeps graph data on CPU and moves sampled
        batches inside the train/eval loop.
        """
        if not self.use_minibatch:
            self._move_to_device(device)
            return

        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")

        self.model.to(device)
        self._move_optimizer_state(device)
        self.data = self.data_cpu
        self.data_gpu = None
        self.current_device = device

    def _clear_memory(self):
        """Helper method to clear CUDA memory and perform garbage collection"""
        clear_memory_basic()

    def _clear_memory_aggressive(self):
        """More aggressive memory clearing for memory-intensive operations"""
        clear_memory_aggressive()

    def _clear_memory_for_data_loading(self, data_loading_option):
        """Clear memory based on data loading method"""
        if data_loading_option == "diffusion":
            clear_memory_for_diffusion()
        elif data_loading_option == "adjacency":
            clear_memory_for_adjacency()
        else:
            clear_memory_basic()

    def train_client(self):
        # Move model/state to device. In mini-batch mode, keep graph data on CPU.
        self._move_for_compute(self.target_device)

        # C5: derive a per-client training seed if experiment_seed is set.
        # None preserves legacy behavior (train() does nothing;
        # train_with_minibatch() seeds 42 as it always did).
        client_seed = (
            None
            if self.experiment_seed is None
            else int(self.experiment_seed) + int(self.client_id)
        )

        try:
            if self.use_minibatch:
                loss, acc, loss_list, acc_list = train_with_minibatch(
                    self.model,
                    self.data,
                    self.epochs,
                    self.optimizer,
                    self.criterion,
                    self.writer,
                    batch_size=self.batch_size,
                    num_neighbors=self.num_neighbors,
                    use_amp=self.use_amp,
                    seed=client_seed,
                    grad_clip_norm=self.grad_clip_norm,
                )
            else:
                loss, acc, loss_list, acc_list, val_acc = train(
                    self.model,
                    self.data,
                    self.epochs,
                    self.optimizer,
                    self.criterion,
                    self.writer,
                    use_amp=self.use_amp,
                    seed=client_seed,
                    grad_clip_norm=self.grad_clip_norm,
                    struct_reg_lambda=self.struct_reg_lambda,
                    struct_reg_warmup_rounds=self.struct_reg_warmup_rounds,
                )

            for loss_item in loss_list:
                self.training_losses.append(loss_item)

            for acc_item in acc_list:
                self.training_accuracies.append(acc_item)

            # Only move back to CPU if keep_data_on_gpu is disabled
            if not self.keep_data_on_gpu:
                self._move_to_device(self.cpu_device)
                self._clear_memory()

            return loss, val_acc, True
        except Exception as e:
            import traceback
            print(f"Error in client {self.client_id} training: {str(e)}")
            print(traceback.format_exc())
            # Attempt to free memory and return default values
            self._move_to_device(self.cpu_device)
            self._clear_memory()
            return 0.0, 0.0, False

    def evaluate(self, criterion):
        # Move model/state to device. In mini-batch mode, keep graph data on CPU.
        self._move_for_compute(self.target_device)

        try:
            if self.use_minibatch:
                result = evaluate_with_minibatch(
                    self.model,
                    self.data,
                    criterion,
                    batch_size=self.batch_size,
                    num_neighbors=self.num_neighbors
                )
            else:
                result = evaluate(self.model, self.data, criterion)

            # Only move back to CPU if keep_data_on_gpu is disabled
            if not self.keep_data_on_gpu:
                self._move_to_device(self.cpu_device)
                self._clear_memory()

            return result
        except Exception as e:
            print(f"Error in client {self.client_id} evaluation: {str(e)}")
            # Attempt to free memory
            self._move_to_device(self.cpu_device)
            self._clear_memory()
            return float('inf'), 0.0

    def test(self, data=None):
        # Move model/state to device. In mini-batch mode, keep graph data on CPU.
        self._move_for_compute(self.target_device)

        # check input dimension of the model itself
        if hasattr(self, 'cfg') and self.cfg.get("debug", False):
            print(f"Input dim of the model: {self.model.dim_in}")
            # print input dim of the data
            print(f"Input dim of the data: {self.data.x.shape[1]}")
        if data is None or is_shard_ref(data):
            data = self.data
        else:
            data = data if self.use_minibatch else data.to(self.target_device)

        try:
            if self.use_minibatch:
                result = test_with_minibatch(
                    self.model,
                    data,
                    batch_size=self.batch_size,
                    num_neighbors=self.num_neighbors
                )
            else:
                result = test(self.model, data)

            # Only move back to CPU if keep_data_on_gpu is disabled
            if not self.keep_data_on_gpu:
                self._move_to_device(self.cpu_device)
                self._clear_memory()

            return result
        except Exception as e:
            print(f"Error in client {self.client_id} testing: {str(e)}")
            # Attempt to free memory
            self._move_to_device(self.cpu_device)
            self._clear_memory()
            return 0.0  # or appropriate default value

    def get_num_train_samples(self) -> int:
        """B1: number of training samples this client owns.

        Used by the server when ``aggregation == "fedavg_weighted"`` to weight
        each client's parameter contribution by ``|D_k| / sum_k |D_k|``.
        Falls back to 0 if the data has no ``train_mask`` so the server can
        gracefully revert to uniform averaging in pathological cases.
        """
        try:
            tm = getattr(self.data, "train_mask", None)
            if tm is None:
                return 0
            # train_mask is typically a bool tensor; handle range/list just in case.
            if hasattr(tm, "sum"):
                return int(tm.sum().item())
            return int(sum(1 for _ in tm))
        except Exception:
            return 0

    def get_params(self) -> dict:
        if not self.keep_data_on_gpu:
            self._move_to_device(self.cpu_device)
        self.optimizer.zero_grad(set_to_none=True)

        params_cpu = tuple(p.detach().cpu() for p in self.model.parameters())
        buffers_cpu = tuple(b.detach().cpu() for b in self.model.buffers())
        buffer_names = tuple(name for name, _ in self.model.named_buffers())

        # Piggyback train sample count so the server avoids a separate
        # get_num_train_samples.remote() round-trip for fedavg_weighted.
        try:
            tm = getattr(self.data, "train_mask", None)
            n_train = int(tm.sum().item()) if tm is not None and hasattr(tm, "sum") else 0
        except Exception:
            n_train = 0

        return {
            'params': params_cpu,
            'buffers': buffers_cpu,
            'buffer_names': buffer_names,
            'num_train_samples': n_train,
        }

    @torch.no_grad()
    def update_params(self, params_dict: dict, current_global_epoch: int) -> None:
        # Update parameters on target device (GPU by default)
        target = self.target_device if self.keep_data_on_gpu else self.cpu_device
        
        if not self.keep_data_on_gpu:
            self._clear_memory()
            self._move_to_device(self.cpu_device)

        # Update parameters (weights and biases)
        for (p, mp) in zip(params_dict['params'], self.model.parameters()):
            mp.data.copy_(p.to(target))

        fedbn = self.cfg.get("bn_fl_strategy", "average") == "fedbn"
        buffer_names = params_dict.get(
            'buffer_names',
            tuple(name for name, _ in self.model.named_buffers())
        )

        # Update buffers. FedBN keeps BatchNorm running stats local to each client.
        for (b, mb, name) in zip(params_dict['buffers'], self.model.buffers(), buffer_names):
            if fedbn and ("running_mean" in name or "running_var" in name):
                continue
            mb.data.copy_(b.to(target))

        if not self.keep_data_on_gpu:
            self._clear_memory()

    def get_loss_acc(self):
        # create a dictionary of training losses and accuracies
        results_dict = {
            "client_id": self.client_id,
            "training_losses": self.training_losses,
            "training_accuracies": self.training_accuracies,
            "validation_losses": self.validation_losses
        }

        # Also include epoch-by-epoch data for finer-grained analysis
        results_dict["epochs_data"] = []
        for i in range(len(self.training_losses)):
            epoch_dict = {
                "epoch": i % self.epochs,  # Local epoch number
                "round": i // self.epochs,  # Global round number
                "loss": self.training_losses[i],
                "accuracy": self.training_accuracies[i]
            }
            results_dict["epochs_data"].append(epoch_dict)

        return results_dict

    def get_peak_gpu_mb(self) -> float | None:
        """Return peak CUDA memory allocated in this worker process (MB), or None."""
        if not torch.cuda.is_available():
            return None
        try:
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
        except Exception:
            return None
