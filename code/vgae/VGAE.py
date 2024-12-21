import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
import seaborn as sns
import csv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uuid
import os

run_id = str(uuid.uuid4())


def log_run_info(run_id, neurons, embedding_size, lr, heads, roc_auc, precision_ground_truth, recall_ground_truth,
                 f1_ground_truth, total_ground_truth,
                 total_predicted_edges, true_positives_top_20_ground_truth, top_predicted_ground_truth_overlap, model):
    run_info_path = '/Users/rameshsubramani/Desktop/GAAE/files/run_info_vgae.csv'
    header = ['Run ID', 'Num Neurons', 'Embedding Size', 'Learning Rate', 'Num Heads', 'ROC-AUC', 'Precision', 'Recall',
              'F1', '#Edges in GT', '#Predicted edges', 'Top 20% overlap', '#Overlapping edges', 'Model']

    run_info_data = [[run_id, neurons, embedding_size, lr, heads, roc_auc, precision_ground_truth, recall_ground_truth,
                      f1_ground_truth, total_ground_truth, total_predicted_edges, true_positives_top_20_ground_truth,
                      top_predicted_ground_truth_overlap, model]]

    if not os.path.exists(run_info_path):
        with open(run_info_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    with open(run_info_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(run_info_data)


def log_epoch_info(run_id, epoch, BCE_Loss, KL_Loss, Total_Loss):
    epoch_info_path = '/Users/rameshsubramani/Desktop/GAAE/files/epoch_info_vgae.csv'
    header = ['Run ID', 'Epoch', 'BCE_Loss', 'KL_Loss', 'Total_Loss']

    epoch_info_data = [[run_id, epoch, BCE_Loss.item(), KL_Loss.item(), Total_Loss.item()]]

    if not os.path.exists(epoch_info_path):
        with open(epoch_info_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    with open(epoch_info_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(epoch_info_data)


class GAT_VGAE(nn.Module):
    def __init__(self, num_features, num_neurons, embedding_size, num_heads, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.gat1 = GATConv(num_features, num_neurons, heads=num_heads, concat=True)
        self.gat2 = GATConv(num_neurons * num_heads, embedding_size, heads=1, concat=False)
        self.num_neurons = num_neurons
        self.num_features = num_features
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_nodes = num_nodes

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

        mu = self.mu_net(embedding)
        log_var = self.log_var_net(embedding)

        z = self.reparameterize(mu, log_var)
        self.mu, self.log_var = mu, log_var
        return z

    def decode(self, z):
        z = z.mean(dim=0)
        decoded = self.decoder(z)
        decoded = torch.sigmoid(decoded)
        decoded = decoded.view(self.num_nodes, self.num_nodes)
        return decoded

    def forward(self, edge_index, x):
        z = self.encode(edge_index, x)
        return self.decode(z)


def construct_adjacency_matrix(expr_data, threshold=0.7):
    corr_matrix = np.corrcoef(expr_data)
    adj_matrix = (corr_matrix > threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix


expr_df = pd.read_csv('/Users/rameshsubramani/Desktop/GAAE/data/m1139_expr.csv')
gene_names = expr_df['Unnamed: 0'].values
expr_values = expr_df.drop(columns=['Unnamed: 0']).values

scaler = StandardScaler()
normalized_expr = scaler.fit_transform(expr_values.T).T


def construct_hybrid_adjacency_matrix(expr_data, threshold=0.7, noise_factor=0.5):
    corr_matrix = np.corrcoef(expr_data)
    adj_matrix = (corr_matrix > threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)
    random_noise = np.random.rand(*adj_matrix.shape) < noise_factor
    adj_matrix = np.logical_or(adj_matrix, random_noise).astype(float)
    return adj_matrix

def construct_pcc_adjacency_matrix(expr_data, threshold=0.7):
    corr_matrix = np.corrcoef(expr_data)
    adj_matrix = (corr_matrix > threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix


adj_matrix_noise = construct_hybrid_adjacency_matrix(normalized_expr, threshold=0.5, noise_factor=0.1)

adj_matrix_noise_tensor = torch.FloatTensor(adj_matrix_noise)
expr_tensor = torch.FloatTensor(normalized_expr)

edge_index = torch.tensor(np.array(np.where(adj_matrix_noise == 1)), dtype=torch.long)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edge_index = edge_index.to(device)
vgae_model = GAT_VGAE(
    num_neurons=64,
    num_features=normalized_expr.shape[1],
    embedding_size=16,
    num_nodes=adj_matrix_noise.shape[0],
    num_heads=4
)
optimizer = torch.optim.Adam(vgae_model.parameters(), lr=0.005)
bce_losses = []
kl_losses = []
total_losses = []

num_epochs = 200
pcc_adj_matrix = construct_pcc_adjacency_matrix(normalized_expr, threshold=0.5)
pcc_adj_matrix_tensor = torch.FloatTensor(pcc_adj_matrix).to(device)

for epoch in range(num_epochs):
    optimizer.zero_grad()

    x_hat = vgae_model(edge_index, expr_tensor).view(-1).T
    pcc_matrix_flat = pcc_adj_matrix_tensor.view(-1).to(device)

    bce_loss = F.binary_cross_entropy(x_hat, pcc_matrix_flat)

    kl_weight = 0.9
    kl_loss = -0.5 * torch.sum(1 + vgae_model.log_var - vgae_model.mu.pow(2) - vgae_model.log_var.exp()) / \
              pcc_adj_matrix.shape[0]

    total_loss = bce_loss + kl_weight * kl_loss
    total_loss.backward()
    optimizer.step()

    bce_losses.append(bce_loss.item())
    kl_losses.append(kl_loss.item())
    total_losses.append(total_loss.item())
    log_epoch_info(run_id, epoch, bce_loss, kl_loss, total_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/200, BCE Loss: {bce_loss:.4f}, KL Loss: {kl_loss:.4f},  Total Loss: {total_loss:.4f}")


inferred_adjacency = (x_hat.detach().cpu().numpy() > 0.5).astype(int)

net_df = pd.read_csv('/Users/rameshsubramani/Desktop/GAAE/data/m1139_net.csv')
true_adj_matrix = pd.DataFrame(0, index=gene_names, columns=gene_names)
for _, row in net_df.iterrows():
    gene1, gene2 = row['Gene1'], row['Gene2']
    true_adj_matrix.at[gene1, gene2] = 1
    true_adj_matrix.at[gene2, gene1] = 1

roc_auc = roc_auc_score(true_adj_matrix.values.flatten(), inferred_adjacency.flatten())
print(f"ROC-AUC Score: {roc_auc:.4f}")

original_adjacency = pcc_adj_matrix_tensor.numpy()
reconstructed_adjacency = x_hat.detach().cpu().numpy()
num_nodes = int(np.sqrt(reconstructed_adjacency.size))
reconstructed_adjacency = reconstructed_adjacency.reshape(num_nodes, num_nodes)

predicted_edges = (reconstructed_adjacency > 0.5).astype(int)
ground_truth_edges = true_adj_matrix.values

predicted_scores = reconstructed_adjacency.flatten()
computed_flat = adj_matrix_noise.flatten()
ground_truth_flat = ground_truth_edges.flatten()

predicted_binary = (predicted_scores > 0.5).astype(int)

total_predicted_edges = np.sum(predicted_binary)
total_computed_edges = np.sum(computed_flat)
total_ground_truth = np.sum(ground_truth_flat)

print("####################################################")

print(f"Total Predicted Edges: {total_predicted_edges}")
print(f"Total PCC computed edges (Training adjacency matrix): {total_computed_edges}")
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

precision_computed = precision_score(computed_flat, predicted_binary)
recall_computed = recall_score(computed_flat, predicted_binary)
f1_computed = f1_score(computed_flat, predicted_binary)

precision_ground_truth = precision_score(ground_truth_flat, predicted_binary)
recall_ground_truth = recall_score(ground_truth_flat, predicted_binary)
f1_ground_truth = f1_score(ground_truth_flat, predicted_binary)

print("####################################################")

print(f"Precision (Computed Adjacency Matrix): {precision_computed:.4f}")
print(f"Recall (Computed Adjacency Matrix): {recall_computed:.4f}")
print(f"F1 Score (Computed Adjacency Matrix): {f1_computed:.4f}")

print("####################################################")

print(f"Precision (Ground Truth Adjacency Matrix): {precision_ground_truth:.4f}")
print(f"Recall (Ground Truth Adjacency Matrix): {recall_ground_truth:.4f}")
print(f"F1 Score (Ground Truth Adjacency Matrix): {f1_ground_truth:.4f}")

print("####################################################")

log_run_info(run_id, vgae_model.num_neurons, vgae_model.embedding_size, optimizer.param_groups[0]['lr'],
             vgae_model.num_heads, roc_auc,
             precision_ground_truth, recall_ground_truth, f1_ground_truth, total_ground_truth, total_predicted_edges,
             true_positives_top_20_ground_truth, top_predicted_ground_truth_overlap, vgae_model)

########################################
# Plotting Original and Reconstructed Adjacency Matrices
########################################
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

########################################
# ROC AUC Curve
########################################
fpr, tpr, thresholds = roc_curve(ground_truth_flat, predicted_scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line (random chance)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

########################################
# Plotting Loss Curves
########################################
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), bce_losses, label='BCE Loss', color='blue')
plt.plot(range(1, num_epochs + 1), kl_losses, label='KL Loss', color='orange')
plt.plot(range(1, num_epochs + 1), total_losses, label='Total Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve during Training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
