from client import FLClient
import torch
from dataprocessingset import load_processed_data
from models import GCN, GAT, VanillaGNN, MLP
import ray

DEVICE =  "cpu"

ray.init()

clients_num = 5
beta = 0.5

# load the dataset
data, cora_dataset, clients_data = load_processed_data(num_clients=clients_num, beta=beta)
data = data.to(DEVICE)

# print the dataset
print(cora_dataset)
print(len(clients_data))

# instantiate the model
model = GCN(cora_dataset.num_features, 16, cora_dataset.num_classes)
# model = VanillaGNN(input_dim=cora_dataset.num_node_features, hidden_dim=16, output_dim=cora_dataset.num_classes)

# create an FL client with each item in the clients_data
clients = [FLClient.remote(data) for data in clients_data]
# clients = [FLClient(data, model) for data in clients_data]

num_clients = len(clients)

print(f"{num_clients} clients created")

# train the model for 10 epochs
epochs = 20
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
clients[0].train.remote(epochs, optimizer, criterion)
# clients[0].train(epochs, optimizer, criterion)

# evaluate the model
# loss, acc = clients[0].evaluate.remote(criterion)
result_ref = clients[0].evaluate.remote(criterion)
# result_ref = clients[0].evaluate(criterion)
loss, acc = ray.get(result_ref)
# loss, acc = result_ref
print(f"Validation Loss: {loss:.3f}, Validation Accuracy: {acc:.3f}")



