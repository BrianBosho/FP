import ray
import torch
from enhanced_training import enhanced_train, enhanced_evaluate, enhanced_test
from enhanced_models import EnhancedGCN, EnhancedSAGE, EnhancedGAT, SAGE_Products, GCN_Arxiv
from memory_efficient_models import MemoryEfficientGNN, MemoryEfficientMLP
from models import VanillaGNN, MLP
from gnn_models import GCN, GAT, GCN_arxiv, GCN_products, SAGE_products
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

gpu_nums = 1/10

@ray.remote(num_gpus=gpu_nums)
class EnhancedFLClient:
    """
    Enhanced Federated Learning client that can use various training methods and models.
    Compatible with both standard models and enhanced models for large datasets.
    """
    def __init__(self, client_id, data, model_type=None, dataset_type="standard", 
                num_features=None, num_classes=None, hidden_dim=16, device=None, 
                epochs=1, batch_size=1024, num_neighbors=None):
        self.DEVICE = device
        self.client_id = client_id
        self.data = data
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        
        # Log memory usage at initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logging.info(f"Client {client_id} - Initial GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Initialize the model based on dataset type
        self._initialize_model()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Statistics tracking
        self.loss_list = []
        self.acc_list = []
        self.test_loss = 0.0
        self.test_acc = 0.0
    
    def _initialize_model(self):
        """Initialize the appropriate model based on dataset type and configuration."""
        if self.num_features is None:
            # Try to infer from data
            if hasattr(self.data, 'x'):
                self.num_features = self.data.x.size(1)
            else:
                raise ValueError("Number of features must be provided or available in data")
                
        if self.num_classes is None:
            # Try to infer from data
            if hasattr(self.data, 'y'):
                self.num_classes = int(self.data.y.max().item()) + 1
            else:
                raise ValueError("Number of classes must be provided or available in data")
        
        # Configure the model based on dataset type
        if self.dataset_type == "products":
            logging.info(f"Client {self.client_id} - Using SAGE_Products model for OGBN-Products dataset with hidden dim={self.hidden_dim}")
            self.model = SAGE_Products(
                nfeat=self.num_features, 
                nhid=self.hidden_dim, 
                nclass=self.num_classes,
                dropout=0.5,
                NumLayers=2
            ).to(self.DEVICE)
            
        elif self.dataset_type == "arxiv":
            logging.info(f"Client {self.client_id} - Using GCN_Arxiv model for OGBN-Arxiv dataset with hidden dim={self.hidden_dim}")
            self.model = GCN_Arxiv(
                nfeat=self.num_features, 
                nhid=self.hidden_dim, 
                nclass=self.num_classes,
                dropout=0.5,
                NumLayers=3
            ).to(self.DEVICE)
            
        else:
            # For standard datasets, use model_type if provided, otherwise default to GCN
            if self.model_type == "GAT" or self.model_type == "EnhancedGAT":
                logging.info(f"Client {self.client_id} - Using standard GAT model with hidden dim={self.hidden_dim}")
                self.model = EnhancedGAT(
                    nfeat=self.num_features, 
                    nhid=self.hidden_dim, 
                    nclass=self.num_classes, 
                    dropout=0.5
                ).to(self.DEVICE)
                
            elif self.model_type == "SAGE" or self.model_type == "EnhancedSAGE":
                logging.info(f"Client {self.client_id} - Using standard SAGE model with hidden dim={self.hidden_dim}")
                self.model = EnhancedSAGE(
                    nfeat=self.num_features, 
                    nhid=self.hidden_dim, 
                    nclass=self.num_classes, 
                    dropout=0.5
                ).to(self.DEVICE)
                
            elif self.model_type == "VanillaGNN" or self.model_type == "MemoryEfficientGNN":
                logging.info(f"Client {self.client_id} - Using memory-efficient VanillaGNN with hidden dim={self.hidden_dim}")
                self.model = MemoryEfficientGNN(
                    nfeat=self.num_features, 
                    nhid=self.hidden_dim, 
                    nclass=self.num_classes, 
                    dropout=0.5
                ).to(self.DEVICE)
                
            elif self.model_type == "MLP" or self.model_type == "MemoryEfficientMLP":
                logging.info(f"Client {self.client_id} - Using memory-efficient MLP with hidden dim={self.hidden_dim}")
                self.model = MemoryEfficientMLP(
                    nfeat=self.num_features, 
                    nhid=self.hidden_dim, 
                    nclass=self.num_classes, 
                    dropout=0.5
                ).to(self.DEVICE)
                
            else:
                # Default to GCN
                logging.info(f"Client {self.client_id} - Using standard GCN model with hidden dim={self.hidden_dim}")
                self.model = EnhancedGCN(
                    nfeat=self.num_features, 
                    nhid=self.hidden_dim, 
                    nclass=self.num_classes, 
                    dropout=0.5,
                    NumLayers=2
                ).to(self.DEVICE)
                
        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Client {self.client_id} - Model has {total_params} parameters")
        
    def setup_optimizer(self):
        """Setup optimizer based on model type."""
        # Default learning rate and weight decay
        lr = 0.01
        weight_decay = 5e-4
        
        # Adjust for different dataset types
        if self.dataset_type == "products":
            lr = 0.003  # Lower learning rate for stability with large datasets
            
        elif self.dataset_type == "arxiv":
            lr = 0.005  # Moderate learning rate for medium-sized datasets
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def train_client(self):
        """
        Train the client model on its local data.
        
        Returns:
            Tuple of (final_loss, final_accuracy)
        """
        # Log memory usage before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logging.info(f"Client {self.client_id} - Before training GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # For large datasets like OGBN-Products, use mini-batch training
        if self.dataset_type == "products":
            logging.info(f"Client {self.client_id} - Mini-batch training with batch size {self.batch_size}, neighbors {self.num_neighbors}")
            
            # Use mini-batch training for large datasets
            from enhanced_training import enhanced_train_minibatch
            
            loss, acc, loss_list, acc_list = enhanced_train_minibatch(
                self.model, 
                self.data, 
                self.epochs, 
                self.optimizer, 
                self.criterion, 
                None,  # No writer for clients
                batch_size=self.batch_size,
                num_neighbors=self.num_neighbors,
                dataset_type=self.dataset_type
            )
        else:
            # Train the model using the enhanced training functionality
            loss, acc, loss_list, acc_list = enhanced_train(
                self.model, 
                self.data, 
                self.epochs, 
                self.optimizer, 
                self.criterion, 
                None,  # No writer for clients
                dataset_type=self.dataset_type
            )
            
        # Store training metrics
        self.loss_list.extend(loss_list)
        self.acc_list.extend(acc_list)
        
        # Evaluate on validation set
        val_loss, val_acc = self.evaluate(self.criterion)
        
        # Log training results
        logging.info(f"Client {self.client_id} - Training completed: Final Loss: {loss:.4f}, Acc: {acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Log memory usage after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logging.info(f"Client {self.client_id} - After training GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
        # Test on local test data to get current performance
        test_acc = self.test()
        
        return (loss, acc, val_acc, test_acc)
        
    def evaluate(self, criterion=None):
        """Evaluate the model on validation data."""
        if criterion is None:
            criterion = self.criterion
        return enhanced_evaluate(self.model, self.data, criterion, dataset_type=self.dataset_type)
    
    def test(self, data=None):
        """
        Test the model on test data.
        
        Args:
            data: Optional data to test on. If None, uses client's local data.
            
        Returns:
            Test accuracy
        """
        # Use client's data if no data is provided
        if data is None:
            data = self.data
            
        # For large datasets like OGBN-Products, use mini-batch testing
        if self.dataset_type == "products" and hasattr(self.model, 'forward_batch'):
            logging.info(f"Client {self.client_id} - Using mini-batch testing for OGBN-Products dataset")
            
            try:
                # Enhanced test for mini-batch processing
                test_acc = enhanced_test(self.model, data, dataset_type=self.dataset_type)
                self.test_acc = test_acc
                return test_acc
            except Exception as e:
                logging.error(f"Client {self.client_id} - Error in testing: {e}")
                import traceback
                logging.error(traceback.format_exc())
                return 0.0
        else:
            # Standard testing for other datasets
            try:
                test_acc = enhanced_test(self.model, data, dataset_type=self.dataset_type)
                self.test_acc = test_acc
                return test_acc
            except Exception as e:
                logging.error(f"Client {self.client_id} - Error in testing: {e}")
                import traceback
                logging.error(traceback.format_exc())
                return 0.0
    
    def get_params(self) -> tuple:
        """Get the model parameters."""
        return tuple(self.model.state_dict().items())
    
    @torch.no_grad()
    def update_params(self, params: tuple, current_global_epoch: int) -> None:
        """
        Update the model parameters.
        
        Args:
            params: New parameters as a tuple of (name, value) pairs
            current_global_epoch: Current global epoch number
        """
        new_params = dict(params)
        self.model.load_state_dict(new_params)
    
    def get_loss_acc(self):
        """Get the latest loss and accuracy values."""
        return {
            'loss': self.loss_list,
            'acc': self.acc_list,
            'test_acc': self.test_acc
        } 