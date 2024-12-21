import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
import optuna
import os
import csv
import uuid

# Define the model
class GAT_VGAE(nn.Module):
    def __init__(self, num_neurons, num_features, embedding_size, num_heads=8):
        super(GAT_VGAE, self).__init__()
        self.num_neurons = num_neurons
        self.num_features = num_features
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        self.gat1 = GATConv(self.num_features, self.num_neurons, heads=self.num_heads, concat=True)
        self.gat2 = GATConv(self.num_neurons * self.num_heads, self.embedding_size, heads=1, concat=False)
        self.decoder = nn.Linear(self.embedding_size, self.num_features ** 2)

    def encode(self, edge_index, x_features):
        hidden_0 = self.gat1(x_features.T, edge_index)
        hidden_0 = F.relu(hidden_0)
        self.GCN_mu = self.gat2(hidden_0, edge_index)
        self.GCN_mu = F.relu(self.GCN_mu)
        self.GCN_sigma = torch.exp(self.GCN_mu)
        z = self.GCN_mu + torch.randn_like(self.GCN_sigma) * self.GCN_sigma
        return z

    def decode(self, z):
        x_hat_flat = torch.sigmoid(self.decoder(z))
        expected_size = self.num_features ** 2
        assert x_hat_flat.size(1) == expected_size, (
            f"Decoder output size {x_hat_flat.size(1)} does not match expected size {expected_size}"
        )

        x_hat = x_hat_flat.view(-1, self.num_features, self.num_features)
        return x_hat[0]

    def forward(self, edge_index, x_features):
        z = self.encode(edge_index, x_features)
        x_hat = self.decode(z)
        return x_hat


# Function to construct adjacency matrix
def construct_adjacency_matrix(expr_data, threshold=0.7):
    corr_matrix = np.corrcoef(expr_data)
    adj_matrix = (corr_matrix > threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix


# Read the data
expr_df = pd.read_csv('/Users/rameshsubramani/Desktop/GAAE/data/expr.csv')
gene_names = expr_df['Unnamed: 0'].values
expr_values = expr_df.drop(columns=['Unnamed: 0']).values

scaler = StandardScaler()
normalized_expr = scaler.fit_transform(expr_values.T).T
adj_matrix = construct_adjacency_matrix(normalized_expr, threshold=0.5)

adj_matrix_tensor = torch.FloatTensor(adj_matrix)
expr_tensor = torch.FloatTensor(normalized_expr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read ground truth adjacency matrix
net_df = pd.read_csv('/Users/rameshsubramani/Desktop/GAAE/data/net.csv')
true_adj_matrix = pd.DataFrame(0, index=gene_names, columns=gene_names)
for _, row in net_df.iterrows():
    gene1, gene2 = row['Gene1'], row['Gene2']
    true_adj_matrix.at[gene1, gene2] = 1
    true_adj_matrix.at[gene2, gene1] = 1


# Log the run information
def log_run_info(run_id, neurons, embedding_size, lr, heads, roc_auc, precision, recall, f1, total_ground_truth,
                 total_predicted_edges, true_positives_top_20_ground_truth, top_predicted_ground_truth_overlap, model):
    run_info_path = '/Users/rameshsubramani/Desktop/GAAE/files/run_info_vgae.csv'
    header = ['Run ID', 'Num Neurons', 'Embedding Size', 'Learning Rate', 'Num Heads', 'ROC-AUC', 'Precision', 'Recall',
              'F1', '#Edges in GT', '#Predicted edges', 'Top 20% overlap', '#Overlapping edges', 'Model']

    run_info_data = [[run_id, neurons, embedding_size, lr, heads, roc_auc, precision, recall, f1, total_ground_truth,
                      total_predicted_edges, true_positives_top_20_ground_truth, top_predicted_ground_truth_overlap, model]]

    # Check if the file exists and write accordingly
    if not os.path.exists(run_info_path):
        with open(run_info_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    with open(run_info_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(run_info_data)


# Log the epoch information
def log_epoch_info(run_id, epoch, loss):
    epoch_info_path = '/Users/rameshsubramani/Desktop/GAAE/files/epoch_info_vgae.csv'
    header = ['Run ID', 'Epoch', 'Loss']

    epoch_info_data = [[run_id, epoch, loss.item()]]

    # Check if the file exists and write accordingly
    if not os.path.exists(epoch_info_path):
        with open(epoch_info_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    with open(epoch_info_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(epoch_info_data)


# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters to optimize
    num_neurons = trial.suggest_categorical('num_neurons', [32, 64, 128])
    embedding_size = trial.suggest_categorical('embedding_size', [8, 16, 32, 64, 128])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8, 16])
    num_epochs = trial.suggest_int('num_epochs', 100, 500)

    # Initialize model and optimizer
    model = GAT_VGAE(num_neurons=num_neurons, num_features=normalized_expr.shape[0], embedding_size=embedding_size, num_heads=num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long).to(device)
    x_features = torch.FloatTensor(normalized_expr).to(device)



    run_id = str(uuid.uuid4())
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        x_hat = model(edge_index, x_features)
        x_hat = x_hat.T
        x_hat_flat = x_hat.reshape(-1)
        adj_matrix_flat = adj_matrix_tensor.view(-1).to(device)

        loss = torch.nn.functional.binary_cross_entropy(x_hat_flat, adj_matrix_flat)
        loss.backward()
        optimizer.step()

        # Log epoch-level loss
        log_epoch_info(run_id, epoch + 1, loss)

    # Inferred adjacency matrix
    inferred_adjacency = (x_hat.detach().cpu().numpy() > 0.5).astype(int)

    # Compute evaluation metrics
    inferred_flat = inferred_adjacency.flatten()
    ground_truth_flat = true_adj_matrix.values.flatten()

    # Calculate precision, recall, and F1 score
    precision = precision_score(ground_truth_flat, inferred_flat)
    recall = recall_score(ground_truth_flat, inferred_flat)
    f1 = f1_score(ground_truth_flat, inferred_flat)
    roc_auc = roc_auc_score(ground_truth_flat, inferred_flat)

    total_predicted_edges = np.sum(inferred_flat)
    total_ground_truth = np.sum(ground_truth_flat)

    # Log the results
    log_run_info(run_id, num_neurons, embedding_size, lr, num_heads, roc_auc, precision, recall, f1, total_ground_truth,
                 total_predicted_edges, 0, 0, model)

    # Return the ROC-AUC as the optimization goal
    return roc_auc


# Run the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

# Print the best hyperparameters
print("Best Hyperparameters:", study.best_params)
