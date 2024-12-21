import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, mutual_info_score
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import torch.nn.functional as F
import tensorflow as tf
import datetime
import csv
import os
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

dropout_rate = 0.5
ngsl = 0.01


# class EarlyStopping:
#     def __init__(self, patience=10, delta=0):
#
#         self.patience = patience
#         self.delta = delta
#         self.best_loss = np.inf
#         self.counter = 0
#         self.early_stop = False
#         self.best_weights = None
#
#     def __call__(self, val_loss, model):
#         if val_loss < self.best_loss - self.delta:
#             self.best_loss = val_loss
#             self.counter = 0
#             self.best_weights = model.state_dict()
#         else:
#             self.counter += 1
#
#         if self.counter >= self.patience:
#             self.early_stop = True
#             print(f"Early stopping triggered at epoch {epoch + 1}")
#         return self.early_stop


# early_stopping = EarlyStopping(patience=10, delta=0.01)

def log_training_data(iteration, epochs, model, dropout_rate, relu_neg_slope, in_dim, hidden_dim, latent_dim,
                      op_learning_rate, op_weight_decay, norm_penality, loss, eval_threshold, precision,
                      recall, f1, aucroc, top_k, message):
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [iteration, epochs, model, dropout_rate, relu_neg_slope, in_dim, hidden_dim, latent_dim,
             op_learning_rate, op_weight_decay, norm_penality, loss, eval_threshold, precision,
             recall, f1, aucroc, top_k, message])

# Mutual Information
# def compute_adjacency_matrix(xmn, theta=0.05):
#     number_of_samples = xmn.shape[0]  # This will now be the number of rows (samples)
#     similarity_matrix = np.zeros((number_of_samples, number_of_samples))  # Adjacency matrix for rows
#
#     for i in range(number_of_samples):
#         print(i)
#         for j in range(number_of_samples):
#             if i != j:
#                 # Compute mutual information instead of Pearson correlation
#                 similarity_matrix[i, j] = mutual_info_score(xmn[i, :], xmn[j, :])  # Compare rows (samples)
#
#     np.fill_diagonal(similarity_matrix, 0)  # Remove diagonal elements (self-similarity)
#     max_similarity = np.max(similarity_matrix)
#     adjacency_matrix = np.where(similarity_matrix >= theta * max_similarity, similarity_matrix, 0)
#
#     return adjacency_matrix

# Mutual Information
def compute_adjacency_matrix(xmn, theta=0.05):
    number_of_genes = xmn.shape[0]  # This will be the number of rows (genes)
    similarity_matrix = np.zeros((number_of_genes, number_of_genes))  # Adjacency matrix for genes

    for i in range(number_of_genes):
        print(i)
        for j in range(number_of_genes):
            if i != j:
                similarity_matrix[i, j] = mutual_info_score(xmn[:, i], xmn[:, j])
                similarity_matrix[j, i] = mutual_info_score(xmn[:, i], xmn[:, j])

    np.fill_diagonal(similarity_matrix, 0)  # Remove diagonal elements (self-similarity)
    max_similarity = np.max(similarity_matrix)
    adjacency_matrix = np.where(similarity_matrix >= theta * max_similarity, similarity_matrix, 0)

    return adjacency_matrix


# PCC and MI
def hybrid_similarity(xmn, alpha=0.5, theta=0.05):
    number_of_samples = xmn.shape[0]
    similarity_matrix = np.zeros((number_of_samples, number_of_samples))

    for i in range(number_of_samples):
        for j in range(number_of_samples):
            if i != j:
                # Compute Mutual Information (MI)
                mi = mutual_info_score(xmn[i, :], xmn[j, :])
                # Compute Pearson Correlation Coefficient (PCC)
                pcc = pearsonr(xmn[i, :], xmn[j, :])[0]
                # Combine MI and PCC with a weighted sum
                similarity_matrix[i, j] = alpha * mi + (1 - alpha) * pcc

    np.fill_diagonal(similarity_matrix, 0)  # Remove diagonal elements (self-similarity)

    # Apply thresholding
    max_similarity = np.max(similarity_matrix)
    adjacency_matrix = np.where(similarity_matrix >= theta * max_similarity, similarity_matrix, 0)

    return adjacency_matrix



def perform_pca(data, n_components=50):
    print('Performing PCA...')
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data


def adjacency_to_edge_index(adj):
    edge_index = torch.nonzero(adj, as_tuple=False).T
    edge_weight = adj[edge_index[0], edge_index[1]]
    return edge_index, edge_weight


class GATLinkAutoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, num_heads=4):
        super(GATLinkAutoencoder, self).__init__()

        self.encoder1 = GATConv(in_dim, hidden_dim, heads=num_heads, dropout=0.5)
        self.encoder2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.5)
        # self.encoder3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout_rate)  # Added third GAT layer
        self.latent_layer = GATConv(hidden_dim * num_heads, latent_dim, heads=1, dropout=0.5)
        self.dropout = nn.Dropout(p=0.5)
        self.decoder1 = GATConv(latent_dim, hidden_dim, heads=1, dropout=0.5)
        self.decoder2 = nn.Linear(hidden_dim, in_dim)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, edge_index):
        x1 = self.encoder1(x, edge_index)
        x1 = self.leaky_relu(x1)
        x1 = self.dropout(x1)

        x2 = self.encoder2(x1, edge_index)
        x2 = self.leaky_relu(x2)
        x2 = self.dropout(x2)

        # x3 = self.encoder3(x2, edge_index)
        # x3 = self.leaky_relu(x3)
        # x3 = self.dropout(x3)

        # # Skip connection between encoder layers
        # x3 = x3 + x2

        z = self.latent_layer(x2, edge_index)
        x_reconstructed = self.decoder1(z, edge_index)
        x_reconstructed = self.leaky_relu(x_reconstructed)
        reconstructed = self.decoder2(x_reconstructed)

        reconstructed = reconstructed + x  # skip connection between encoder and decoder

        return reconstructed, z


def cap_outliers(data, lower_percentile=1, upper_percentile=99):
    """
    Cap the outliers in the dataset based on specified percentiles.
    """
    # Calculate lower and upper bounds for each column (gene)
    lower_bound = np.percentile(data, lower_percentile, axis=0)
    upper_bound = np.percentile(data, upper_percentile, axis=0)

    # Apply the capping for each gene (column)
    for i in range(data.shape[1]):
        data[:, i] = np.clip(data[:, i], lower_bound[i], upper_bound[i])

    return data


# class GCNLinkAutoencoder(nn.Module):
#     def __init__(self, in_dim, hidden_dim, latent_dim):
#         super(GCNLinkAutoencoder, self).__init__()
#
#         # Add more GCN layers for increased depth
#         self.encoder1 = GCNConv(in_dim, hidden_dim)
#         self.encoder2 = GCNConv(hidden_dim, hidden_dim)
#         self.encoder3 = GCNConv(hidden_dim, hidden_dim)  # Added third GCN layer
#         self.latent_layer = GCNConv(hidden_dim, latent_dim)
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.decoder1 = GCNConv(latent_dim, hidden_dim)
#         self.decoder2 = nn.Linear(hidden_dim, in_dim)
#
#         self.leaky_relu = nn.LeakyReLU(negative_slope=ngsl)
#
#     def forward(self, x, edge_index):
#         # Encoder part with additional layers
#         x1 = self.encoder1(x, edge_index)  # First GCN layer
#         x1 = self.leaky_relu(x1)
#         x1 = self.dropout(x1)
#
#         x2 = self.encoder2(x1, edge_index)  # Second GCN layer
#         x2 = self.leaky_relu(x2)
#         x2 = self.dropout(x2)
#
#         x3 = self.encoder3(x2, edge_index)  # Third GCN layer
#         x3 = self.leaky_relu(x3)
#         x3 = self.dropout(x3)
#
#         # Skip connection between encoder layers
#         x3 = x3 + x2
#
#         z = self.latent_layer(x3, edge_index)
#         x_reconstructed = self.decoder1(z, edge_index)
#         x_reconstructed = self.leaky_relu(x_reconstructed)
#         reconstructed = self.decoder2(x_reconstructed)
#
#         reconstructed = reconstructed + x  # Skip connection between decoder and input
#
#         return reconstructed, z


log_dir = "/Users/rameshsubramani/Desktop/GAAE/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

csv_file = "../../files/training_log.csv"
headers = ['iteration', 'num_of_epochs', 'model', 'dropout_rate', 'relu_neg_slope', 'in_dim', 'hidden_dim',
           'latent_dim', 'op_learning_rate', 'op_weight_decay', 'norm_penality', 'loss', 'eval_threshold',
           'precision', 'recall', 'f1', 'aucroc', 'top_k', 'message']
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

data_path = '/data/expr.csv'
R = pd.read_csv(data_path, index_col=0)

# Cap outliers
R_values_capped = cap_outliers(R.values, lower_percentile=1, upper_percentile=99)

# Continue with the rest of the pipeline
features = torch.tensor(R_values_capped, dtype=torch.float)


# Recompute the adjacency matrix
A_mn = compute_adjacency_matrix(R_values_capped, theta=0.05)
np.save('adjacency_matrix_capped.npy', A_mn)

adj_matrix = torch.tensor(A_mn, dtype=torch.float)

edge_index, edge_weight = adjacency_to_edge_index(adj_matrix)
print("edge_index: ", edge_index)
print("edge_weight: ", edge_weight)

# Assuming you have a validation set (you need to load or split it properly)
validation_data = Data(x=features, edge_index=edge_index)  # This is a placeholder, replace with actual validation data


def compute_loss(model, data):
    reconstructed, _ = model(data.x, data.edge_index)
    return loss_fn(reconstructed, data.x)  # This can be replaced with whatever loss you need

loss_fn = nn.L1Loss()


def objective(trial):
    epochs = 200
    norm_p = 2
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    hidden_dim = trial.suggest_categorical('hidden_dim', [8, 16, 32, 64, 128, 256, 512, 1024])
    latent_dim = trial.suggest_categorical('latent_dim', [8, 16, 32, 64, 128, 256, 512, 1024])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    threshold = trial.suggest_loguniform('threshold', 0.3, 0.5)

    # Model training
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

        # Training step
        reconstructed, z = model(data.x, data.edge_index)
        loss = loss_fn(reconstructed, features)

        norm_penalty = torch.norm(z, p=norm_p)
        loss += 1e-5 * norm_penalty

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation step
        model.eval()  # Set model to evaluation mode
        val_loss = compute_loss(model, validation_data)  # Compute validation loss

        # Early stopping check
        # if early_stopping(val_loss, model):
        #     print(f"Training stopped at epoch {epoch+1}")
        #     break

        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    print("Finished Training")
    print("Latent: ", z)
    torch.save(z.detach().numpy(), 'latent_embeddings.npy')

    Z = z.detach().numpy()
    # A_pred = 1 / (1 + np.exp(-np.dot(Z, Z.T)))
    # A_pred = np.tanh(np.dot(Z, Z.T))
    # A_pred = np.exp(np.dot(Z, Z.T)) / np.sum(np.exp(np.dot(Z, Z.T)), axis=1, keepdims=True)
    A_pred = cosine_similarity(Z)
    # A_pred = rbf_kernel(Z)
    # A_pred = np.dot(Z, Z.T)
    # A_pred = np.exp(-np.linalg.norm(Z[:, None] - Z, axis=2) ** 2)

    np.save('predicted_adjacency.npy', A_pred)
    print("Predicted Adjacency: ", A_pred)

    # Evaluation
    A_binary = (A_pred > threshold).astype(int)
    np.save('binary_adjacency.npy', A_binary)

    ground_truth_path = '/data/net.csv'
    expression_path = '/data/expr.csv'

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

    print(f"Ground truth adjacency matrix shape: {A_ground_truth.shape}")
    np.save('ground_truth_adjacency.npy', A_ground_truth)

    if A_pred.shape != A_ground_truth.shape:
        raise ValueError("Dimension mismatch between predicted and ground truth adjacency matrices.")

    A_ground_truth_flat = A_ground_truth.flatten()
    A_pred_flat = A_pred.flatten()
    A_binary_flat = A_binary.flatten()

    # tp = np.sum((np.triu(A_binary) == 1) & (np.triu(A_ground_truth) == 1))
    # fp = np.sum((np.triu(A_binary) == 1) & (np.triu(A_ground_truth) == 0))
    # fn = np.sum((np.triu(A_binary) == 0) & (np.triu(A_ground_truth) == 1))
    # tn = np.sum((np.triu(A_binary) == 0) & (np.triu(A_ground_truth) == 0))

    tp = np.sum((A_binary == 1) & (A_ground_truth == 1))
    fp = np.sum((A_binary == 1) & (A_ground_truth == 0))
    fn = np.sum((A_binary == 0) & (A_ground_truth == 1))
    tn = np.sum((A_binary == 0) & (A_ground_truth == 0))

    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")

    precision = precision_score(A_ground_truth_flat, A_binary_flat)
    recall = recall_score(A_ground_truth_flat, A_binary_flat)
    f1 = f1_score(A_ground_truth_flat, A_binary_flat)
    auc = roc_auc_score(A_ground_truth_flat, A_pred_flat)

    print("Validation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUROC: {auc:.4f}")

    # total_ground_truth_edges = np.sum(np.triu(A_ground_truth))
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

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plot heatmap of predicted adjacency matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(A_pred, cmap='Blues')
    plt.title("Predicted Adjacency Matrix")
    plt.show()
    # Plot heatmap of GT adjacency matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(A_ground_truth, cmap='Blues')
    plt.title("GT Adjacency Matrix")
    plt.show()

    # Plot histogram of predicted values
    plt.figure(figsize=(10, 6))
    plt.hist(A_pred.flatten(), bins=50)
    plt.title("Histogram of Predicted Adjacency Values")
    plt.show()

    message = ("Optuna")
    log_training_data(96, epochs, model, dropout_rate, ngsl, features.shape[0], hidden_dim, latent_dim, learning_rate,
                      weight_decay, norm_p, loss.item(), threshold, precision, recall, f1, auc, top_k_edges_in_ground_truth,
                      message)

    return top_k_edges_in_ground_truth


# Optuna study
study = optuna.create_study(direction='maximize')  # Maximize AUC
study.optimize(objective, n_trials=3)  # Run for 50 trials

# After optimization, print the best hyperparameters
print(f"Best hyperparameters: {study.best_params}")
print(f"Best AUC: {study.best_value}")


best_params = study.best_params
