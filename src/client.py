import ray
from train import train, evaluate, test
from models import VanillaGNN, MLP
# from models import GCN, GAT, VanillaGNN, MLP
from gnn_models import GCN, GAT, GCN_arxiv, GCN_products, SAGE_products
import torch

gpu_nums = 1/10

@ray.remote(num_gpus=gpu_nums)
# @ray.remote(num_cpus=0.25)
class FLClient:
    def __init__(self, data, dataset, client_id, cfg, device, model_type="GCN"):
        self.DEVICE = device
        self.device = self.DEVICE
        self.data = data.to(self.device)
        dataset = dataset

        if model_type == "GCN":
            self.model = GCN(dataset.num_features, 16, dataset.num_classes).to(self.device)
        elif model_type == "GAT":
            self.model = GAT(dataset.num_features, 16, dataset.num_classes).to(self.device)

        self.epochs = cfg["epochs"]
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

    def train_client(self):
        loss, acc, loss_list, acc_list = train(self.model, self.data, self.epochs, self.optimizer, self.criterion, self.writer)
        
        for loss_item in loss_list:
            self.training_losses.append(loss_item)

        for acc_item in acc_list:
            self.training_accuracies.append(acc_item)
        
        return loss, acc

    def evaluate(self, criterion):
        self.model.to(self.device)
        self.data = self.data.to(self.device)
        return evaluate(self.model, self.data, criterion)

    def test(self, data=None):
        self.model.to(self.device)
        if data is None:
            data = self.data
        else:
            data = data.to(self.device)
        return test(self.model, data)

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