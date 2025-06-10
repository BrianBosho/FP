import ray
from train import train, evaluate, test, train_with_minibatch, evaluate_with_minibatch, test_with_minibatch
from models import GCN, GAT, VanillaGNN, MLP, GCN_arxiv, GraphSAGEProducts
import torch
import sys
import gc

# Configure batch sizes based on dataset size
LARGE_DATASET_THRESHOLD = 100000  # Number of nodes threshold for large datasets
DEFAULT_BATCH_SIZE = 1024  # Further reduced to 32 to help with memory constraints
DEFAULT_NUM_NEIGHBORS = [10, 10, 10]  # Reduced further to sample even fewer neighbors

gpu_nums = 1/20

@ray.remote(num_gpus=gpu_nums)
# @ray.remote(num_cpus=0.25)
class FLClient:
    def __init__(self, data, dataset, client_id, cfg, device, model_type="GCN"):
        # Clear any existing CUDA cache and perform garbage collection
        torch.cuda.empty_cache()
        gc.collect()

        self.DEVICE = device
        self.device = self.DEVICE
        self.data = data.to(self.device)
        self.dataset = dataset
        self.dataset_name = dataset.name if hasattr(dataset, 'name') else "unknown"

        # get input dim from data
        self.input_dim = data.x.shape[1]
        print(f"Input dim: {self.input_dim}")
        # Determine if this is a large dataset that requires mini-batching
        self.use_minibatch = False
        if hasattr(data, 'x') and data.x.shape[0] > LARGE_DATASET_THRESHOLD:
            self.use_minibatch = True
            print(f"Client {client_id}: Using mini-batch training for large dataset with {data.x.shape[0]} nodes")
        
        # For ogbn-arxiv, always use mini-batching
        if self.dataset_name == "ogbn-arxiv" or self.dataset_name == "ogbn-products":
            self.use_minibatch = True
            print(f"Client {client_id}: Using mini-batch training for {self.dataset_name} dataset")

        if model_type == "GCN":
            if self.dataset_name == "ogbn-arxiv":
                self.model = GCN_arxiv(input_dim=dataset.num_features, hidden_dim=256, output_dim=40, dropout=0.5).to(self.device)
            elif self.dataset_name == "ogbn-products":
                self.model = GraphSAGEProducts(input_dim=100, hidden_dim=256, output_dim=47, dropout=0.5, num_layers=3).to(self.device)
            else:
                self.model = GCN(self.input_dim, 16, dataset.num_classes).to(self.device)
        elif model_type == "GAT":
            self.model = GAT(dataset.num_features, 16, dataset.num_classes).to(self.device)

        self.epochs = cfg["epochs"]
        
        # Setup batch configuration from config if available
        self.batch_size = cfg.get("batch_size", DEFAULT_BATCH_SIZE)
        self.num_neighbors = cfg.get("num_neighbors", DEFAULT_NUM_NEIGHBORS)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=cfg["lr"], weight_decay=5e-4
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.writer = None

        # list of training losses & accuracies
        self.training_losses = []
        self.training_accuracies = []
        self.validation_losses = []
        self.client_id = client_id

    def _clear_memory(self):
        """Helper method to clear CUDA memory and perform garbage collection"""
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def train_client(self):
        # Clear memory before training
        self._clear_memory()
        
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
                    num_neighbors=self.num_neighbors
                )
            else:
                loss, acc, loss_list, acc_list = train(
                    self.model, 
                    self.data, 
                    self.epochs, 
                    self.optimizer, 
                    self.criterion, 
                    self.writer
                )
            
            for loss_item in loss_list:
                self.training_losses.append(loss_item)

            for acc_item in acc_list:
                self.training_accuracies.append(acc_item)
            
            # Clear memory after training
            self._clear_memory()
            
            return loss, acc
        except Exception as e:
            import traceback
            print(f"Error in client {self.client_id} training: {str(e)}")
            print(traceback.format_exc())
            # Attempt to free memory and return default values
            self._clear_memory()
            return 0.0, 0.0

    def evaluate(self, criterion):
        # Ensure model and data are on the correct device
        self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        # Clear memory before evaluation
        self._clear_memory()
        
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
            self._clear_memory()
            
            return result
        except Exception as e:
            print(f"Error in client {self.client_id} evaluation: {str(e)}")
            # Attempt to free memory
            self._clear_memory()
            return 0.0  # or appropriate default value

    def test(self, data=None):
        # Ensure model is on the correct device
        self.model.to(self.device)
        # check input dimenoso of the model itself
        print(f"Input dim of the model: {self.model.dim_in}")
        # print input dim of the data
        print(f"Input dim of the data: {self.data.x.shape[1]}")
        if data is None:
            data = self.data
        else:
            data = data.to(self.device)
        
        # Clear memory before testing
        self._clear_memory()
        
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
            self._clear_memory()
            
            return result
        except Exception as e:
            print(f"Error in client {self.client_id} testing: {str(e)}")
            # Attempt to free memory
            self._clear_memory()
            return 0.0  # or appropriate default value

    def get_params(self) -> tuple:
        self.optimizer.zero_grad(set_to_none=True)
        return tuple(self.model.parameters())

    @torch.no_grad()
    def update_params(self, params: tuple, current_global_epoch: int) -> None:
        # load global parameter from global server
        self.model.to("cpu")
        for (p, mp) in zip(params, self.model.parameters()):
            mp.data = p
        self.model.to(self.device)

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