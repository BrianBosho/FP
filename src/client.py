import ray
from src.train import train, evaluate, test, train_with_minibatch, evaluate_with_minibatch, test_with_minibatch
from src.models import GCN, GAT, VanillaGNN, MLP, GCN_arxiv, GraphSAGEProducts, PubmedGAT
import torch
import sys
import gc
from src.utils.memory_utils import (
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

# Use minimal GPU allocation (0.01) to give workers GPU visibility without reserving memory
# This allows multiple actors to share the same GPU while keeping data on CPU when idle
@ray.remote(num_gpus=1/10)
class FLClient:
    def __init__(self, data, dataset, client_id, cfg, device, model_type="GCN"):
        # Clear any existing CUDA cache and perform garbage collection
        torch.cuda.empty_cache()
        gc.collect()

        self.cfg = cfg
        self.DEVICE = device
        self.target_device = torch.device(device)
        self.cpu_device = torch.device("cpu")
        self.current_device = self.cpu_device
        self.data_gpu = None
        
        # Get debug flag from config
        debug = cfg.get("debug", False)
        
        # LOG: What data did this client receive?
        if debug:
            print(f"\n[Client {client_id}] Initializing with:")
            print(f"  - Input data nodes: {data.num_nodes}")
            print(f"  - Input data edges: {data.edge_index.shape[1] if hasattr(data, 'edge_index') else 'N/A'}")
            print(f"  - Feature shape: {data.x.shape}")
            print(f"  - Data device (before moving): {data.x.device}")
        
        # Keep a CPU copy always resident; swap to GPU only when actively computing
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

        if model_type == "GCN":
            if self.dataset_name == "ogbn-arxiv":
                self.model = GCN_arxiv(input_dim=self.input_dim, hidden_dim=256, output_dim=40, dropout=0.5).to(self.cpu_device)
            elif self.dataset_name == "ogbn-products":
                # Some processed splits use fixed dimensionality; fall back to self.input_dim if config overrides exist
                products_input_dim = getattr(dataset, "num_features", None) or self.input_dim
                self.model = GraphSAGEProducts(input_dim=products_input_dim, hidden_dim=256, output_dim=47, dropout=0.5, num_layers=3).to(self.cpu_device)
            else:
                self.model = GCN(self.input_dim, 16, dataset.num_classes).to(self.cpu_device)
        elif model_type == "GAT":
            # Use same configuration logic as server for consistency
            model_params = {}
            if cfg is not None and "model_params" in cfg and model_type in cfg["model_params"]:
                model_params = cfg["model_params"][model_type]
            
            if self.dataset_name == "Pubmed":
                self.model = PubmedGAT(self.input_dim, 8, dataset.num_classes, heads=8).to(self.cpu_device)
            else:
                hidden_dim = model_params.get("hidden_dim", 8)  # FedGAT default: 8
                num_heads = model_params.get("num_heads", 8)    # FedGAT default: 8
                dropout = model_params.get("dropout", 0.6)      # FedGAT default: 0.6
                self.model = GAT(self.input_dim, hidden_dim, dataset.num_classes, heads=num_heads, dropout=dropout).to(self.cpu_device)

        self.epochs = cfg["epochs"]
        
        # Setup batch configuration from config (used only if use_minibatch=True)
        self.batch_size = cfg.get("batch_size", DEFAULT_BATCH_SIZE)
        self.num_neighbors = cfg.get("num_neighbors", DEFAULT_NUM_NEIGHBORS)
        
        # Setup mixed precision training
        self.use_amp = bool(cfg.get("use_amp", False) and self.target_device.type == "cuda")

        optimizer_type = cfg["optimizer"]
        lr = cfg["lr"]
        weight_decay = cfg["decay"]

        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )

        if optimizer_type == "SGD":
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
        # Clear memory before training
        self._clear_memory()
        self._move_to_device(self.target_device)
        
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
                    use_amp=self.use_amp
                )
            else:
                loss, acc, loss_list, acc_list = train(
                    self.model, 
                    self.data, 
                    self.epochs, 
                    self.optimizer, 
                    self.criterion, 
                    self.writer,
                    use_amp=self.use_amp
                )
            
            for loss_item in loss_list:
                self.training_losses.append(loss_item)

            for acc_item in acc_list:
                self.training_accuracies.append(acc_item)
            
            # Clear memory after training
            self._move_to_device(self.cpu_device)
            self._clear_memory()
            
            return loss, acc
        except Exception as e:
            import traceback
            print(f"Error in client {self.client_id} training: {str(e)}")
            print(traceback.format_exc())
            # Attempt to free memory and return default values
            self._move_to_device(self.cpu_device)
            self._clear_memory()
            return 0.0, 0.0

    def evaluate(self, criterion):
        # Clear memory before evaluation
        self._clear_memory()
        self._move_to_device(self.target_device)
        
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
            
            # Clear memory after evaluation
            self._move_to_device(self.cpu_device)
            self._clear_memory()
            
            return result
        except Exception as e:
            print(f"Error in client {self.client_id} evaluation: {str(e)}")
            # Attempt to free memory
            self._move_to_device(self.cpu_device)
            self._clear_memory()
            return 0.0  # or appropriate default value

    def test(self, data=None):
        # Clear memory before testing
        self._clear_memory()
        self._move_to_device(self.target_device)

        # check input dimension of the model itself
        if hasattr(self, 'cfg') and self.cfg.get("debug", False):
            print(f"Input dim of the model: {self.model.dim_in}")
            # print input dim of the data
            print(f"Input dim of the data: {self.data.x.shape[1]}")
        if data is None:
            data = self.data
        else:
            data = data.to(self.target_device)
        
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
            
            # Clear memory after testing
            self._move_to_device(self.cpu_device)
            self._clear_memory()
            
            return result
        except Exception as e:
            print(f"Error in client {self.client_id} testing: {str(e)}")
            # Attempt to free memory
            self._move_to_device(self.cpu_device)
            self._clear_memory()
            return 0.0  # or appropriate default value

    def get_params(self) -> dict:
        self._move_to_device(self.cpu_device)
        self.optimizer.zero_grad(set_to_none=True)
        
        # Return CPU tensors to avoid keeping GPU memory
        params_cpu = tuple(p.detach().cpu() for p in self.model.parameters())
        buffers_cpu = tuple(b.detach().cpu() for b in self.model.buffers())
        
        return {
            'params': params_cpu,
            'buffers': buffers_cpu
        }

    @torch.no_grad()
    def update_params(self, params_dict: dict, current_global_epoch: int) -> None:
        # Clear memory before parameter update
        self._clear_memory()
        self._move_to_device(self.cpu_device)

        # Update parameters (weights and biases)
        for (p, mp) in zip(params_dict['params'], self.model.parameters()):
            mp.data.copy_(p.to(self.cpu_device))
        
        # Update buffers (BatchNorm running stats, etc.)
        for (b, mb) in zip(params_dict['buffers'], self.model.buffers()):
            mb.data.copy_(b.to(self.cpu_device))

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