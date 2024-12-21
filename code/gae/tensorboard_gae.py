import uuid

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, mutual_info_score
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
import tensorflow as tf
import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
from model import *
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter


dropout_rate = 0.5
ngsl = 0.01
epochs = 200
norm_p = 2
learning_rate = 0.015
hidden_dim = 1024
latent_dim = 8
weight_decay = 4.9359513263401676e-05
threshold = 0.4

# Initialize TensorBoard writer
run_id = str(uuid.uuid4())
log_dir = f'runs/{run_id}'
tensorboard_writer = SummaryWriter(log_dir=log_dir)

csv_file = "/Users/rameshsubramani/Desktop/GAAE/files/training_log.csv"
headers = ['iteration', 'num_of_epochs', 'model', 'dropout_rate', 'relu_neg_slope', 'in_dim', 'hidden_dim',
           'latent_dim', 'op_learning_rate', 'op_weight_decay', 'norm_penality', 'loss', 'eval_threshold',
           'precision', 'recall', 'f1', 'aucroc', 'top_k', 'message']
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

data_path = '/Users/rameshsubramani/Desktop/GAAE/data/expr.csv'
R = pd.read_csv(data_path, index_col=0)

# Cap outliers
R_values_capped = cap_outliers(R.values, lower_percentile=1, upper_percentile=99)
features = torch.tensor(R_values_capped, dtype=torch.float)

A_mn = np.load('/Users/rameshsubramani/Desktop/GAAE/files/adjacency_matrix_capped.npy')
adj_matrix = torch.tensor(A_mn, dtype=torch.float)

edge_index, edge_weight = adjacency_to_edge_index(adj_matrix)
print("edge_index: ", edge_index)
print("edge_weight: ", edge_weight)

loss_fn = nn.L1Loss()

model = GATLinkAutoencoder(
    in_dim=features.shape[1],
    hidden_dim=hidden_dim,
    latent_dim=latent_dim
)

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

data = Data(x=features, edge_index=edge_index)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    reconstructed, z = model(data.x, data.edge_index)
    loss = loss_fn(reconstructed, features)

    norm_penalty = torch.norm(z, p=norm_p)
    loss += 1e-5 * norm_penalty

    loss.backward()
    optimizer.step()
    scheduler.step()

    model.eval()

    # Log loss to TensorBoard
    tensorboard_writer.add_scalar('Loss/train', loss.item(), epoch)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Inferred adjacency matrix
inferred_adjacency = (reconstructed.detach().cpu().numpy() > 0.5).astype(int)

# Evaluation
A_pred = cosine_similarity(z.detach().cpu().numpy())
A_binary = (A_pred > threshold).astype(int)

ground_truth_path = '/Users/rameshsubramani/Desktop/GAAE/data/net.csv'
expression_path = '/Users/rameshsubramani/Desktop/GAAE/data/expr.csv'

expression_data = pd.read_csv(expression_path, index_col=0)
final_gene_list = expression_data.index.tolist()

ground_truth = pd.read_csv(ground_truth_path)
num_genes = len(final_gene_list)
A_ground_truth = np.zeros((num_genes, num_genes))

gene_to_index = {gene: i for i, gene in enumerate(final_gene_list)}

for _, row in ground_truth.iterrows():
    gene1, gene2 = row['Gene1'], row['Gene2']
    if gene1 in gene_to_index and gene2 in gene_to_index:
        i, j = gene_to_index[gene1], gene_to_index[gene2]
        A_ground_truth[i, j] = 1
        A_ground_truth[j, i] = 1

# ROC-AUC Score
A_ground_truth_flat = A_ground_truth.flatten()
A_pred_flat = A_pred.flatten()

precision = precision_score(A_ground_truth_flat, A_binary.flatten())
recall = recall_score(A_ground_truth_flat, A_binary.flatten())
f1 = f1_score(A_ground_truth_flat, A_binary.flatten())
auc = roc_auc_score(A_ground_truth_flat, A_pred_flat)

print("Validation Results:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUROC: {auc:.4f}")

# Log evaluation metrics to TensorBoard
tensorboard_writer.add_scalar('Precision', precision, epochs)
tensorboard_writer.add_scalar('Recall', recall, epochs)
tensorboard_writer.add_scalar('F1', f1, epochs)
tensorboard_writer.add_scalar('ROC-AUC', auc, epochs)

# Log training message to TensorBoard
tensorboard_writer.add_text('Training Message', 'Training completed successfully.', epochs)

total_ground_truth_edges = np.sum(A_ground_truth)
total_predicted_edges = np.sum(A_binary)
A_ground_truth_bool = A_ground_truth.astype(bool)
A_binary_bool = A_binary.astype(bool)
overlapping_edges = np.sum(A_ground_truth_bool & A_binary_bool)

print(f"Total predicted edges: {total_predicted_edges}")
print(f"Edges also in ground truth: {overlapping_edges}")
print(f"Total edges in the ground truth adjacency matrix: {int(total_ground_truth_edges)}")

top_k_percentage = 0.4
A_bin_flat = A_binary.flatten()
top_k = int(len(A_bin_flat) * top_k_percentage)
top_k_indices = np.argsort(A_bin_flat)[-top_k:][::-1]
top_k_edges = [divmod(idx, A_binary.shape[0]) for idx in top_k_indices]
A_ground_truth_flat = A_ground_truth.flatten()
top_k_edges_in_ground_truth = 0
for edge in top_k_edges:
    i, j = edge
    if A_ground_truth[i, j] == 1:
        top_k_edges_in_ground_truth += 1

print(f"Number of top {top_k} edges that are in the ground truth: {top_k_edges_in_ground_truth}")


# Heatmap of predicted adjacency matrix
plt.figure(figsize=(10, 8))
sns.heatmap(A_pred, cmap='Blues')
plt.title("Predicted Adjacency Matrix")
plt.show()
# Heatmap of GT adjacency matrix
plt.figure(figsize=(10, 8))
sns.heatmap(A_ground_truth, cmap='Blues')
plt.title("GT Adjacency Matrix")
plt.show()

# Histogram of predicted values
plt.figure(figsize=(10, 6))
plt.hist(A_pred.flatten(), bins=50)
plt.title("Histogram of Predicted Adjacency Values")
plt.show()

message = ""
log_training_data(csv_file, 118, epochs, model, dropout_rate, ngsl, features.shape[0], hidden_dim, latent_dim, learning_rate,
                  weight_decay, norm_p, loss.item(), threshold, precision, recall, f1, auc, top_k_edges_in_ground_truth,
                  message)

# Save predicted matrices and embeddings
np.save('/Users/rameshsubramani/Desktop/GAAE/files/predicted_adjacency.npy', A_pred)
np.save('/Users/rameshsubramani/Desktop/GAAE/files/binary_adjacency.npy', A_binary)
np.save('/Users/rameshsubramani/Desktop/GAAE/files/ground_truth_adjacency.npy', A_ground_truth)

# Close TensorBoard writer
tensorboard_writer.close()
