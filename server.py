import ray
import torch
from train import evaluate, test


class Server():
    def __init__(self, clients, model, device) -> None:
        self.DEVICE = device
        self.device = self.DEVICE
        self.clients = clients
        self.model = model.to(self.device)
        self.num_of_trainers = len(clients)

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
        
        return train_results

    def evaluate_clients(self):
        clients = self.clients
        criterion = torch.nn.CrossEntropyLoss()
        eval_futures = [client.evaluate.remote(criterion) for client in clients]
        results = ray.get(eval_futures)
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
        return evaluate(self.model, data, criterion)
    
    def test_global_model(self, data):
        self.model.to(self.device)
        data = data.to(self.device)
        return test(self.model, data)