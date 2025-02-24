#######################################
#         Model Definition            #
#######################################
import json
import uuid

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

#######################################
#         Global Variables            #
#######################################
# Unique Run ID
run_id = str(uuid.uuid4())

# Load configuration from file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Additional configurable values
DROPOUT = config.get('dropout', 0.2)



class GAT_VGAE(nn.Module):
    def __init__(self, num_features, num_neurons, embedding_size, num_heads, num_nodes, dropout=DROPOUT):
        super().__init__()
        self.num_nodes = num_nodes
        self.gat1 = GATConv(num_features, num_neurons, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(num_neurons * num_heads, embedding_size, heads=1, concat=False, dropout=dropout)
        self.mu_net = nn.Linear(embedding_size, embedding_size)
        self.log_var_net = nn.Linear(embedding_size, embedding_size)
        self.decoder = nn.Linear(embedding_size, num_nodes * num_nodes)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, edge_index, x):
        hidden = F.relu(self.gat1(x, edge_index))
        embedding = self.gat2(hidden, edge_index)
        mu, log_var = self.mu_net(embedding), self.log_var_net(embedding)
        self.mu, self.log_var = mu, log_var
        return self.reparameterize(mu, log_var)

    def decode(self, z):
        z = z.mean(dim=0)
        decoded = torch.sigmoid(self.decoder(z)).view(self.num_nodes, self.num_nodes)
        return decoded

    def forward(self, edge_index, x):
        z = self.encode(edge_index, x)
        return self.decode(z)


