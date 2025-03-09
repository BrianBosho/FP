import ray
import torch
from enhanced_training import enhanced_evaluate, enhanced_test, to_cpu_scalar
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedServer:
    """
    Enhanced server that coordinates federated learning across clients,
    with support for different model types and dataset types.
    """
    def __init__(self, clients, model, device, dataset_type="standard") -> None:
        self.DEVICE = device
        self.device = self.DEVICE
        self.clients = clients
        self.model = model.to(self.device)
        self.num_of_trainers = len(clients)
        self.dataset_type = dataset_type
        
        # Track server-side metrics
        self.global_test_accuracies = []
        self.global_val_losses = []
        self.global_val_accuracies = []
        self.global_rounds = 0
        
        # Update the client params
        self.broadcast_params(-1)
        
    @torch.no_grad()
    def train_clients(self, current_global_epoch: int) -> list:
        """Train all clients and aggregate their models."""
        clients = self.clients
        
        # Request all clients to train
        logging.info(f"Starting client training for global round {current_global_epoch}")
        train_futures = [client.train_client.remote() for client in clients]
        
        # Ensure we get CPU values back from Ray
        try:
            train_results = ray.get(train_futures)
            
            # Ensure results are converted to Python primitives
            processed_results = []
            for result in train_results:
                # Convert each value in the result to CPU if it's a tensor
                processed_result = []
                for value in result:
                    if isinstance(value, torch.Tensor):
                        processed_result.append(to_cpu_scalar(value))
                    else:
                        processed_result.append(value)
                processed_results.append(processed_result)
            
            train_results = processed_results
        except Exception as e:
            logging.error(f"Error processing train results: {e}")
            train_results = []
        
        # Get updated parameters from all clients
        logging.info("Collecting client parameters for aggregation")
        params = [client.get_params.remote() for client in clients]
        self.zero_params()
        
        # Aggregate parameters asynchronously as they become available
        while True:
            ready, left = ray.wait(params, num_returns=1, timeout=None)
            if ready:
                for t in ready:
                    for p, mp in zip(ray.get(t), self.model.parameters()):
                        mp.data += p.to(self.device)
            params = left
            if not params:
                break
        
        # Average parameters
        for p in self.model.parameters():
             p.data /= self.num_of_trainers
        
        # Update global round counter
        self.global_rounds += 1
        
        # Broadcast updated parameters to all clients
        self.broadcast_params(current_global_epoch)
        
        logging.info(f"Completed global round {current_global_epoch}")
        return train_results

    def evaluate_clients(self, test_data=None):
        """Evaluate all clients' models."""
        clients = self.clients
        criterion = torch.nn.CrossEntropyLoss()
        
        # Get evaluation results from all clients
        eval_futures = [client.evaluate.remote(criterion) for client in clients]
        
        try:
            client_results = ray.get(eval_futures)
            
            # Ensure results are converted to Python primitives
            processed_results = []
            for result in client_results:
                if isinstance(result, tuple) and len(result) == 2:
                    loss, acc = result
                    processed_results.append((to_cpu_scalar(loss), to_cpu_scalar(acc)))
                else:
                    processed_results.append(result)  # Leave as is if not the expected format
            
            client_results = processed_results
        except Exception as e:
            logging.error(f"Error processing evaluation results: {e}")
            client_results = []
        
        # Also evaluate global model if test data is provided
        if test_data is not None:
            try:
                # Handle both single data object and list of data objects
                if isinstance(test_data, list):
                    # If test_data is a list, use the first item for global model evaluation
                    logging.info(f"Test data is a list with {len(test_data)} items. Using the first item for global model evaluation.")
                    if len(test_data) > 0:
                        global_val_loss, global_val_acc = self.evaluate_global_model(test_data[0], criterion)
                        self.global_val_losses.append(to_cpu_scalar(global_val_loss))
                        self.global_val_accuracies.append(to_cpu_scalar(global_val_acc))
                        logging.info(f"Global model - Validation Loss: {global_val_loss:.4f}, Accuracy: {global_val_acc:.4f}")
                else:
                    # If test_data is a single object, use it directly
                    global_val_loss, global_val_acc = self.evaluate_global_model(test_data, criterion)
                    self.global_val_losses.append(to_cpu_scalar(global_val_loss))
                    self.global_val_accuracies.append(to_cpu_scalar(global_val_acc))
                    logging.info(f"Global model - Validation Loss: {global_val_loss:.4f}, Accuracy: {global_val_acc:.4f}")
            except Exception as e:
                logging.error(f"Error evaluating global model: {e}")
        
        return client_results
    
    def test_clients(self, test_data=None):
        """Test all clients' models."""
        clients = self.clients
        
        # Get test results from all clients
        test_futures = [client.test.remote(test_data) for client in clients]
        
        try:
            client_results = ray.get(test_futures)
            
            # Ensure results are converted to Python primitives
            processed_results = []
            for result in client_results:
                processed_results.append(to_cpu_scalar(result))
            
            client_results = processed_results
        except Exception as e:
            logging.error(f"Error processing test results: {e}")
            client_results = []
        
        # Also test global model if test data is provided
        if test_data is not None:
            try:
                # Handle both single data object and list of data objects
                if isinstance(test_data, list):
                    # If test_data is a list, use the first item for global model testing
                    logging.info(f"Test data is a list with {len(test_data)} items. Using the first item for global model testing.")
                    if len(test_data) > 0:
                        global_test_acc = self.test_global_model(test_data[0])
                        self.global_test_accuracies.append(to_cpu_scalar(global_test_acc))
                        logging.info(f"Global model - Test Accuracy: {global_test_acc:.4f}")
                else:
                    # If test_data is a single object, use it directly
                    global_test_acc = self.test_global_model(test_data)
                    self.global_test_accuracies.append(to_cpu_scalar(global_test_acc))
                    logging.info(f"Global model - Test Accuracy: {global_test_acc:.4f}")
            except Exception as e:
                logging.error(f"Error testing global model: {e}")
        
        # Calculate average client performance
        if client_results:
            avg_client_acc = sum(client_results) / len(client_results)
            logging.info(f"Average client - Test Accuracy: {avg_client_acc:.4f}")
        
        return client_results
    
    def broadcast_params(self, current_global_epoch: int) -> None:
        """Broadcast global model parameters to all clients."""
        for trainer in self.clients:
            trainer.update_params.remote(
                tuple(self.model.parameters()), current_global_epoch
            )

    @torch.no_grad()
    def zero_params(self) -> None:
        """Reset model parameters to zero before aggregation."""
        for p in self.model.parameters():
            p.zero_()

    def evaluate_global_model(self, data, criterion):
        """Evaluate the global model on provided data."""
        self.model.to(self.device)
        
        # Ensure data is on the correct device
        if hasattr(data, 'to'):
            data = data.to(self.device)
        else:
            logging.warning(f"Data does not have 'to' method. Type: {type(data)}")
            # Try to handle the case where data is a different format
            return 0.0, 0.0  # Return dummy values if we can't evaluate
            
        return enhanced_evaluate(self.model, data, criterion, dataset_type=self.dataset_type)
    
    def test_global_model(self, data):
        """Test the global model on provided data."""
        self.model.to(self.device)
        
        # Ensure data is on the correct device
        if hasattr(data, 'to'):
            data = data.to(self.device)
        else:
            logging.warning(f"Data does not have 'to' method. Type: {type(data)}")
            # Try to handle the case where data is a different format
            return 0.0  # Return dummy value if we can't test
            
        return enhanced_test(self.model, data, dataset_type=self.dataset_type)
    
    def get_metrics(self):
        """Get all tracked metrics."""
        # Convert all metrics to Python primitives to ensure they can be safely serialized
        return {
            "global_test_accuracies": [to_cpu_scalar(acc) for acc in self.global_test_accuracies],
            "global_val_losses": [to_cpu_scalar(loss) for loss in self.global_val_losses],
            "global_val_accuracies": [to_cpu_scalar(acc) for acc in self.global_val_accuracies],
            "total_rounds": self.global_rounds
        } 