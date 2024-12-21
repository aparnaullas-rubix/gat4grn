import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from datetime import datetime
import uuid
import os
from torch.utils.tensorboard import SummaryWriter


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
expr_df = pd.read_csv('/data/expr.csv')
gene_names = expr_df['Unnamed: 0'].values
expr_values = expr_df.drop(columns=['Unnamed: 0']).values

scaler = StandardScaler()
normalized_expr = scaler.fit_transform(expr_values.T).T
adj_matrix = construct_adjacency_matrix(normalized_expr, threshold=0.5)

adj_matrix_tensor = torch.FloatTensor(adj_matrix)
expr_tensor = torch.FloatTensor(normalized_expr)

vgae_model = GAT_VGAE(num_neurons=64, num_features=normalized_expr.shape[0], embedding_size=4, num_heads=4)
optimizer = torch.optim.Adam(vgae_model.parameters(), lr=0.005)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgae_model.to(device)

edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)
edge_index = edge_index.to(device)

x_features = torch.FloatTensor(normalized_expr).to(device)

# Generate a unique run ID
run_id = str(uuid.uuid4())

# TensorBoard setup
log_dir = f'runs/{run_id}'
writer = SummaryWriter(log_dir)  # Initialize TensorBoard writer


# Log the run information
def log_run_info(run_id, neurons, embedding_size, lr, heads, roc_auc, precision_ground_truth, recall_ground_truth,
                 f1_ground_truth, total_ground_truth,
                 total_predicted_edges, true_positives_top_20_ground_truth, top_predicted_ground_truth_overlap, model):
    run_info_path = '/files/run_info_vgae.csv'
    header = ['Run ID', 'Num Neurons', 'Embedding Size', 'Learning Rate', 'Num Heads', 'ROC-AUC', 'Precision', 'Recall',
              'F1', '#Edges in GT', '#Predicted edges', 'Top 20% overlap', '#Overlapping edges', 'Model']

    run_info_data = [[run_id, neurons, embedding_size, lr, heads, roc_auc, precision_ground_truth, recall_ground_truth,
                      f1_ground_truth, total_ground_truth, total_predicted_edges, true_positives_top_20_ground_truth,
                      top_predicted_ground_truth_overlap, model]]
    # Check if the file exists and write accordingly
    if not os.path.exists(run_info_path):
        with open(run_info_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    with open(run_info_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(run_info_data)


# Log the detailed epoch information
def log_epoch_info(run_id, epoch, loss):
    epoch_info_path = 'epoch_info.csv'
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


# Train the model and log details
num_epochs = 200
for epoch in range(num_epochs):
    optimizer.zero_grad()

    x_hat = vgae_model(edge_index, x_features)

    x_hat = x_hat.T

    x_hat_flat = x_hat.reshape(-1)
    adj_matrix_flat = adj_matrix_tensor.view(-1).to(device)

    loss = torch.nn.functional.binary_cross_entropy(x_hat_flat, adj_matrix_flat)

    loss.backward()
    optimizer.step()

    # Log epoch details
    log_epoch_info(run_id, epoch + 1, loss)

    # Log loss to TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Inferred adjacency matrix
inferred_adjacency = (x_hat.detach().cpu().numpy() > 0.5).astype(int)

# Read ground truth adjacency matrix
net_df = pd.read_csv('/data/net.csv')
true_adj_matrix = pd.DataFrame(0, index=gene_names, columns=gene_names)
for _, row in net_df.iterrows():
    gene1, gene2 = row['Gene1'], row['Gene2']
    true_adj_matrix.at[gene1, gene2] = 1
    true_adj_matrix.at[gene2, gene1] = 1

# ROC-AUC Score
roc_auc = roc_auc_score(true_adj_matrix.values.flatten(), inferred_adjacency.flatten())
print(f"ROC-AUC Score: {roc_auc:.4f}")

original_adjacency = adj_matrix_tensor.numpy()
reconstructed_adjacency = x_hat.detach().cpu().numpy()

predicted_edges = (reconstructed_adjacency > 0.5).astype(int)
ground_truth_edges = true_adj_matrix.values

predicted_scores = reconstructed_adjacency.flatten()
computed_flat = adj_matrix.flatten()
ground_truth_flat = ground_truth_edges.flatten()

predicted_binary = (predicted_scores > 0.5).astype(int)

total_predicted_edges = np.sum(predicted_binary)
total_computed_edges = np.sum(computed_flat)
total_ground_truth = np.sum(ground_truth_flat)

print("####################################################")

print(f"Total Predicted Edges: {total_predicted_edges}")
print(f"Total MI computed edges (Training adjacency matrix): {total_computed_edges}")
true_positives_computed = np.sum((predicted_binary == 1) & (computed_flat == 1))
false_positives_computed = np.sum((predicted_binary == 1) & (computed_flat == 0))
false_negatives_computed = np.sum((predicted_binary == 0) & (computed_flat == 1))

true_positives_ground_truth = np.sum((predicted_binary == 1) & (ground_truth_flat == 1))
false_positives_ground_truth = np.sum((predicted_binary == 1) & (ground_truth_flat == 0))
false_negatives_ground_truth = np.sum((predicted_binary == 0) & (ground_truth_flat == 1))

sorted_indices = np.argsort(-predicted_scores)
top_20_percent = int(0.2 * len(predicted_scores))
top_indices = sorted_indices[:top_20_percent]

top_predicted_computed_overlap = computed_flat[top_indices]
top_predicted_ground_truth_overlap = ground_truth_flat[top_indices]

true_positives_top_20_computed = np.sum(top_predicted_computed_overlap)
true_positives_top_20_ground_truth = np.sum(top_predicted_ground_truth_overlap)

print("Comparison with Computed Adjacency Matrix (Training):")
print(f"True Positives (All Predicted): {true_positives_computed}")
print(f"False Positives: {false_positives_computed}")
print(f"False Negatives: {false_negatives_computed}")
print(f"True Positives (Top 20% Ranked Edges): {true_positives_top_20_computed}")

print("####################################################")

print(f"Total Predicted Edges: {total_predicted_edges}")
print(f"Total (External Network) Ground Truth Edges: {total_ground_truth}")
print("Comparison with (External Network) Ground Truth Edges:")
print(f"True Positives (All Predicted): {true_positives_ground_truth}")
print(f"False Positives: {false_positives_ground_truth}")
print(f"False Negatives: {false_negatives_ground_truth}")
print(f"True Positives (Top 20% Ranked Edges): {true_positives_top_20_ground_truth}")

print("####################################################")

from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1 score for the computed adjacency matrix
precision_computed = precision_score(computed_flat, predicted_binary)
recall_computed = recall_score(computed_flat, predicted_binary)
f1_computed = f1_score(computed_flat, predicted_binary)

# Calculate precision, recall, and F1 score for the ground truth adjacency matrix
precision_ground_truth = precision_score(ground_truth_flat, predicted_binary)
recall_ground_truth = recall_score(ground_truth_flat, predicted_binary)
f1_ground_truth = f1_score(ground_truth_flat, predicted_binary)

# Print the metrics
print("####################################################")

print(f"Precision (Computed Adjacency Matrix): {precision_computed:.4f}")
print(f"Recall (Computed Adjacency Matrix): {recall_computed:.4f}")
print(f"F1 Score (Computed Adjacency Matrix): {f1_computed:.4f}")

print("####################################################")

print(f"Precision (Ground Truth Adjacency Matrix): {precision_ground_truth:.4f}")
print(f"Recall (Ground Truth Adjacency Matrix): {recall_ground_truth:.4f}")
print(f"F1 Score (Ground Truth Adjacency Matrix): {f1_ground_truth:.4f}")

print("####################################################")

# Log ROC-AUC score to TensorBoard
writer.add_scalar('Precision', precision_ground_truth, num_epochs)
writer.add_scalar('Recall', recall_ground_truth, num_epochs)
writer.add_scalar('F1', f1_ground_truth, num_epochs)
writer.add_scalar('ROC-AUC', roc_auc, num_epochs)

log_run_info(run_id, vgae_model.num_neurons, vgae_model.embedding_size, optimizer.param_groups[0]['lr'],
             vgae_model.num_heads, roc_auc,
             precision_ground_truth, recall_ground_truth, f1_ground_truth, total_ground_truth, total_predicted_edges,
             true_positives_top_20_ground_truth, top_predicted_ground_truth_overlap, vgae_model)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(original_adjacency, cmap="Blues", cbar=True)
plt.title("Original Adjacency Matrix")
plt.xlabel("Genes")
plt.ylabel("Genes")

plt.subplot(1, 2, 2)
sns.heatmap(reconstructed_adjacency, cmap="Blues", cbar=True)
plt.title("Reconstructed Adjacency Matrix")
plt.xlabel("Genes")
plt.ylabel("Genes")

plt.tight_layout()
plt.show()

# Close TensorBoard writer
writer.close()
