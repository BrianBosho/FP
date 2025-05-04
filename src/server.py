import ray
import torch
from train import evaluate, test, evaluate_with_minibatch, test_with_minibatch
import gc

LARGE_DATASET_THRESHOLD = 100000  # Number of nodes threshold for large datasets
DEFAULT_BATCH_SIZE = 1024  # Further reduced to 32 to help with memory constraints
DEFAULT_NUM_NEIGHBORS = [10, 10, 10] 

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
        return 0
        
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

    # def test_global_model(self, data):
    #     self.model.to(self.device)
    #     data = data.to(self.device)
    #     test_result = test(self.model, data)
    #     # Convert parameters to a list of numpy arrays
    #     params = [p.detach().cpu().numpy() for p in self.model.parameters()]
    #     return test_result, params