#######################################
#               Imports               #
#######################################
import csv
import json
import os
import time
import uuid

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna.samplers import GridSampler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             accuracy_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv

#######################################
#         Global Variables            #
#######################################
# Unique Run ID
run_id = str(uuid.uuid4())

# Load configuration from file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# File paths and hyperparameters from config
RUN_INFO_PATH = config['run_info_path']
EPOCH_INFO_PATH = config['epoch_info_path']
EXPR_FILE = config['expr_file']
NETWORK_FILE = config['network_file']
DATASET = config['dataset']

NUM_NEURONS = config['num_neurons']
EMBEDDING_SIZE = config['embedding_size']
NUM_HEADS = config['num_heads']
LEARNING_RATE = config['learning_rate']
NUM_EPOCHS = config['num_epochs']
THRESHOLD = config['threshold']
NOISE_FACTOR = config['noise_factor']

# Additional configurable values
DROPOUT = config.get('dropout', 0.2)
K_FRACTION = config.get('k_fraction', 0.1)  # Used in EPR calculation

TUNE_HYPERPARAMETERS = config['tune_hyperparameters']


#######################################
#         Logging Functions           #
#######################################
def write_to_csv(file_path, header, data):
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            csv.writer(file).writerow(header)
    with open(file_path, mode='a', newline='') as file:
        csv.writer(file).writerows(data)


def log_run_info(run_id, neurons, embedding_size, lr, heads, roc_auc, precision, recall, f1, epr, acc,
                 total_gt_edges, total_pred_edges, overlap_top20, dataset):
    header = ['Run ID', 'Num Neurons', 'Embedding Size', 'Learning Rate', 'Num Heads', 'ROC-AUC',
              'Precision', 'Recall', 'F1', 'EPR', 'ACC', '#GT Edges', '#Predicted Edges',
              '#Overlapping in Top20', 'Dataset']
    data = [[run_id, neurons, embedding_size, lr, heads, roc_auc, precision, recall, f1, epr, acc,
             total_gt_edges, total_pred_edges, overlap_top20, dataset]]
    write_to_csv(RUN_INFO_PATH, header, data)


def log_epoch_info(run_id, epoch, bce_loss, kl_loss, total_loss):
    header = ['Run ID', 'Epoch', 'BCE_Loss', 'KL_Loss', 'Total_Loss']
    data = [[run_id, epoch, bce_loss.item(), kl_loss.item(), total_loss.item()]]
    write_to_csv(EPOCH_INFO_PATH, header, data)


#######################################
#         Evaluation Functions        #
#######################################
# def evaluate_model(true_adj_matrix, reconstructed_adjacency):
#     true_flat = true_adj_matrix.values.flatten()
#     pred_flat = reconstructed_adjacency.flatten()
#     roc_auc = roc_auc_score(true_flat, pred_flat)
#     precision_val = precision_score(true_flat, pred_flat > 0.5)
#     recall_val = recall_score(true_flat, pred_flat > 0.5)
#     f1 = f1_score(true_flat, pred_flat > 0.5)
#     return roc_auc, precision_val, recall_val, f1

def evaluate_model(true_adj_matrix, reconstructed_adjacency):
    # Flatten the adjacency matrices for metric calculation
    true_flat = true_adj_matrix.values.flatten()
    pred_flat = reconstructed_adjacency.flatten()

    # Calculate ROC-AUC
    roc_auc = roc_auc_score(true_flat, pred_flat)

    # Threshold predictions at 0.5 for other metrics
    pred_binary = pred_flat > 0.5

    precision = precision_score(true_flat, pred_binary)
    recall = recall_score(true_flat, pred_binary)
    f1 = f1_score(true_flat, pred_binary)

    # Edge Prediction Rate (EPR)
    true_edges = true_flat.sum()
    correct_predictions = np.sum(pred_binary & (true_flat == 1))
    edge_prediction_rate = correct_predictions / true_edges if true_edges > 0 else 0

    # Overall Accuracy
    true_positives = np.sum((pred_binary == 1) & (true_flat == 1))
    true_negatives = np.sum((pred_binary == 0) & (true_flat == 0))
    accuracy = (true_positives + true_negatives) / len(true_flat)

    gt_df = pd.read_csv(NETWORK_FILE)
    num_gt_edges = len(gt_df)

    n = reconstructed_adjacency.shape[0]
    predicted_edges = [
        (i, j, reconstructed_adjacency[i, j])
        for i in range(n) for j in range(i + 1, n)
        if reconstructed_adjacency[i, j] > 0.3
    ]
    predicted_edges.sort(key=lambda x: x[2], reverse=True)
    top_percentage = 0.2
    num_top_edges = int(len(predicted_edges) * top_percentage)
    top_predicted_edges = predicted_edges[:num_top_edges]
    overlap_count = sum(1 for i, j, score in top_predicted_edges if true_adj_matrix.values[i, j] == 1)

    print(f"\nNumber of top 20% predicted unique edges: {len(top_predicted_edges)}")
    print(f"Number of overlapping edges in top 20% predictions: {overlap_count}")

    return roc_auc, precision, recall, f1, edge_prediction_rate, accuracy, num_gt_edges, n, overlap_count


def calculate_early_precision_rate(predicted_adj_matrix, true_adj_matrix, k_fraction=K_FRACTION):
    """
    Calculates the Edge Prediction Rate (EPR) based on the top k fraction of predictions.
    """
    true_flat = true_adj_matrix.values.flatten()
    pred_flat = predicted_adj_matrix.flatten()
    k = int(k_fraction * len(pred_flat))
    top_k_indices = np.argsort(pred_flat)[-k:][::-1]
    true_positives_top_k = np.sum(true_flat[top_k_indices])
    early_precision = true_positives_top_k / k
    edge_density = true_flat.mean()
    return early_precision / edge_density if edge_density > 0 else 0


def plot_precision_recall_curve(true_adj_matrix, reconstructed_adjacency):
    """
    Plots the precision-recall curve based on the true and predicted adjacency matrices.
    """
    true_flat = true_adj_matrix.values.flatten()
    pred_flat = reconstructed_adjacency.flatten()
    precision_vals, recall_vals, _ = precision_recall_curve(true_flat, pred_flat)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='blue', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


#######################################
#         Model Definition            #
#######################################
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


#######################################
#  Data Preprocessing & Adjacency     #
#######################################
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    gene_names = df['Unnamed: 0'].values
    expr_values = df.drop(columns=['Unnamed: 0']).values
    scaler = StandardScaler()
    normalized_expr = scaler.fit_transform(expr_values.T).T
    return normalized_expr, gene_names


def construct_adjacency_matrix(expr_data, threshold=THRESHOLD):
    corr_matrix = np.corrcoef(expr_data)
    adj_matrix = (corr_matrix > threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix


def construct_adjacency_matrix_with_noise(expr_data, threshold=THRESHOLD, noise_factor=NOISE_FACTOR):
    adj_matrix = construct_adjacency_matrix(expr_data, threshold)
    random_noise = np.random.rand(*adj_matrix.shape) < noise_factor
    return np.logical_or(adj_matrix, random_noise).astype(float)


def create_true_adjacency_matrix(network_file, gene_names):
    # Each row in the network file represents one unique edge.
    net_df = pd.read_csv(network_file)
    true_adj_matrix = pd.DataFrame(0, index=gene_names, columns=gene_names)
    for _, row in net_df.iterrows():
        gene1, gene2 = row['Gene1'], row['Gene2']
        true_adj_matrix.at[gene1, gene2] = 1
        true_adj_matrix.at[gene2, gene1] = 1
    return true_adj_matrix


#######################################
#     Edge Ranking (Unique Edges)     #
#######################################
def rank_edges(predicted_adj_matrix, top_percent=0.2):
    # Iterate over upper-triangular indices (i < j) to get unique edges.
    n = predicted_adj_matrix.shape[0]
    edges = [(i, j, predicted_adj_matrix[i, j]) for i in range(n) for j in range(i + 1, n)]
    edges.sort(key=lambda x: x[2], reverse=True)
    top_k = int(top_percent * len(edges))
    return edges[:top_k]


#######################################
#          Training Function          #
#######################################
def train_and_evaluate(model, edge_index, expr_tensor, adj_matrix_tensor, true_adj_matrix, num_epochs, optimizer,
                       num_neurons, embedding_size, lr, heads):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        x_hat = model(edge_index, expr_tensor).view(-1)
        target_flat = adj_matrix_tensor.view(-1)
        bce_loss = F.binary_cross_entropy(x_hat, target_flat)
        kl_loss = -0.5 * torch.sum(
            1 + model.log_var - model.mu.pow(2) - model.log_var.exp()
        ) / adj_matrix_tensor.shape[0]
        total_loss = bce_loss + 1.5 * kl_loss
        total_loss.backward()
        optimizer.step()
        log_epoch_info(run_id, epoch, bce_loss, kl_loss, total_loss)

        if (epoch + 1) % 10 == 0:
            reconstructed = model(edge_index, expr_tensor).detach().cpu().numpy()
            roc_auc, prec, rec, f1, epr, acc, num_gt_edges, n, overlap_count = evaluate_model(true_adj_matrix,
                                                                                              reconstructed)
            epr = calculate_early_precision_rate(reconstructed, true_adj_matrix)
            print(f"Epoch {epoch + 1}/{num_epochs}, ROC-AUC: {roc_auc:.4f}, Precision: {prec:.4f}, "
                  f"Recall: {rec:.4f}, F1: {f1:.4f}, EPR: {epr:.4f}, acc: {acc:.4f}")
            log_run_info(run_id, num_neurons, embedding_size, lr, heads,
                         roc_auc, prec, rec, f1, epr, acc, num_gt_edges, n, overlap_count, DATASET)


#######################################
#         Visualization Functions     #
#######################################
def visualize_grn(predicted_adj_matrix, gene_names, threshold=0.5):
    G = nx.Graph()
    G.add_nodes_from(gene_names)
    n = len(gene_names)
    for i in range(n):
        for j in range(i + 1, n):
            if predicted_adj_matrix[i, j] > threshold:
                G.add_edge(gene_names[i], gene_names[j], weight=predicted_adj_matrix[i, j])
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, edge_color="gray")
    plt.title("Predicted Gene Regulatory Network (GRN)")
    plt.show()


def visualize_embeddings(model, expr_tensor, edge_index, method="tsne"):
    model.eval()
    with torch.no_grad():
        z = model.encode(edge_index, expr_tensor).cpu().numpy()
    if method == "pca":
        reduced = PCA(n_components=2).fit_transform(z)
    else:
        reduced = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(z)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, c='blue', edgecolors='black')
    for i, label in enumerate(gene_names):
        plt.text(reduced[i, 0], reduced[i, 1], label, fontsize=8, alpha=0.75)
    plt.title(f"Gene Embeddings Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid()
    plt.show()


def identify_hub_genes(predicted_adj_matrix, gene_names, top_k=10, threshold=0.5):
    G = nx.Graph()
    G.add_nodes_from(gene_names)
    n = len(gene_names)
    for i in range(n):
        for j in range(i + 1, n):
            if predicted_adj_matrix[i, j] > threshold:
                G.add_edge(gene_names[i], gene_names[j], weight=predicted_adj_matrix[i, j])
    degree_centrality = nx.degree_centrality(G)
    sorted_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    print("\nTop Hub Genes in the Predicted GRN:")
    for gene, cent in sorted_hubs[:top_k]:
        print(f"{gene}: {cent:.4f}")
    return [gene for gene, _ in sorted_hubs[:top_k]]


def visualize_grn_with_hubs(predicted_adj_matrix, gene_names, hub_genes, threshold=0.5):
    G = nx.Graph()
    G.add_nodes_from(gene_names)
    n = len(gene_names)
    for i in range(n):
        for j in range(i + 1, n):
            if predicted_adj_matrix[i, j] > threshold:
                G.add_edge(gene_names[i], gene_names[j], weight=predicted_adj_matrix[i, j])
    node_colors = ["red" if gene in hub_genes else "blue" for gene in gene_names]
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8, edge_color="gray")
    plt.title("Predicted GRN with Hub Genes Highlighted")
    plt.show()


def calculate_whole_network_overlap(predicted_adj_matrix, true_adj_matrix, pcc_adj_matrix):
    """
    Calculates the number of overlapping edges between:
    1. Whole PCC network and Ground Truth (GT).
    2. Whole predicted network and Ground Truth (GT).
    """
    # Convert adjacency matrices to binary (edges present or not)
    pred_binary = (predicted_adj_matrix > 0.3).astype(int)
    pcc_binary = (pcc_adj_matrix > 0.9).astype(int)
    true_binary = (true_adj_matrix.values > 0).astype(int)

    # Compute the overlap
    pcc_overlap = np.sum((pcc_binary == 1) & (true_binary == 1))
    pred_overlap = np.sum((pred_binary == 1) & (true_binary == 1))

    print(f"Total edges in GT: {np.sum(true_binary)}")
    print(f"Total edges in PCC network: {np.sum(pcc_binary)}")
    print(f"Total edges in Predicted network: {np.sum(pred_binary)}")
    print(f"Overlapping edges (PCC vs GT): {pcc_overlap}")
    print(f"Overlapping edges (Predicted vs GT): {pred_overlap}")

    return pcc_overlap, pred_overlap


# ---------------------------
# Objective Function for Exhaustive Search
# ---------------------------
def objective(trial):
    num_neurons = trial.suggest_categorical('num_neurons', [16, 32, 64, 128])
    embedding_size = trial.suggest_categorical('embedding_size', [8, 16, 32, 64])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8, 16])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0001, 0.005, 0.0005])

    print(f"Trial {trial.number}: num_neurons={num_neurons}, embedding_size={embedding_size}, "
          f"num_heads={num_heads}, learning_rate={learning_rate:.5f}")

    expr_data, gene_names = preprocess_data(EXPR_FILE)
    adj_matrix = construct_adjacency_matrix_with_noise(expr_data, threshold=THRESHOLD, noise_factor=NOISE_FACTOR)
    expr_tensor = torch.FloatTensor(expr_data)
    edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)
    adj_matrix_tensor = torch.FloatTensor(adj_matrix)
    true_adj_matrix = create_true_adjacency_matrix(NETWORK_FILE, gene_names)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = edge_index.to(device)
    expr_tensor = expr_tensor.to(device)
    adj_matrix_tensor = adj_matrix_tensor.to(device)

    model = GAT_VGAE(
        num_features=expr_data.shape[1],
        num_neurons=num_neurons,
        embedding_size=embedding_size,
        num_heads=num_heads,
        num_nodes=adj_matrix.shape[0]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_and_evaluate(model, edge_index, expr_tensor, adj_matrix_tensor, true_adj_matrix, num_epochs=NUM_EPOCHS,
                       optimizer=optimizer, num_neurons=num_neurons, embedding_size=embedding_size, lr=learning_rate,
                       heads=num_heads)

    reconstructed_adjacency_eval = model(edge_index, expr_tensor).detach().cpu().numpy()
    roc_auc_eval, prec, rec, f1, epr, acc, num_gt_edges, n, overlap_count = evaluate_model(true_adj_matrix,
                                                                                           reconstructed_adjacency_eval)

    print(f"Trial {trial.number} completed with ROC-AUC: {roc_auc_eval:.4f}")
    log_run_info(run_id, num_neurons, embedding_size, learning_rate, num_heads,
                 roc_auc_eval, prec, rec, f1, epr, acc, num_gt_edges, n, overlap_count, DATASET)

    return roc_auc_eval


#######################################
#          Main Execution Block       #
#######################################
if __name__ == "__main__":
    if __name__ == "__main__":
        if TUNE_HYPERPARAMETERS:
            search_space = {
                "num_neurons": [16, 32, 64, 128],
                "embedding_size": [8, 16, 32, 64],
                "num_heads": [2, 4, 8, 16],
                "learning_rate": [0.001, 0.0005]
            }
            sampler = GridSampler(search_space)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=256)

            best_params = study.best_trial.params
            print(f"Using best hyperparameters from tuning: {best_params}")

        else:
            best_params = {
                "num_neurons": config["num_neurons"],
                "embedding_size": config["embedding_size"],
                "num_heads": config["num_heads"],
                "learning_rate": config["learning_rate"]
            }
            print(f"Using predefined hyperparameters from config.json: {best_params}")

        # --- Data Preprocessing and Adjacency Construction ---
        expr_data, gene_names = preprocess_data(EXPR_FILE)
        adj_matrix = construct_adjacency_matrix_with_noise(expr_data, threshold=THRESHOLD, noise_factor=NOISE_FACTOR)
        expr_tensor = torch.FloatTensor(expr_data)
        edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)
        adj_matrix_tensor = torch.FloatTensor(adj_matrix)
        true_adj_matrix = create_true_adjacency_matrix(NETWORK_FILE, gene_names)

        # --- Model Setup ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        edge_index = edge_index.to(device)
        expr_tensor = expr_tensor.to(device)

        # <-- Use best_params instead of your global config variables:
        model = GAT_VGAE(
            num_features=expr_data.shape[1],
            num_neurons=best_params["num_neurons"],
            embedding_size=best_params["embedding_size"],
            num_heads=best_params["num_heads"],
            num_nodes=adj_matrix.shape[0],
            dropout=DROPOUT
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])

        # --- Training ---
        start_time = time.time()
        train_and_evaluate(
            model, edge_index, expr_tensor, adj_matrix_tensor, true_adj_matrix,
            NUM_EPOCHS, optimizer,
            num_neurons=best_params["num_neurons"],
            embedding_size=best_params["embedding_size"],
            lr=best_params["learning_rate"],
            heads=best_params["num_heads"]
        )
        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time:.4f} s")

        # --- Final Evaluation ---
        reconstructed_adjacency = model(edge_index, expr_tensor).detach().cpu().numpy()
        roc_auc, prec, rec, f1, epr, acc, num_gt_edges, n, overlap_count = evaluate_model(true_adj_matrix,
                                                                                          reconstructed_adjacency)

        # Compute accuracy by thresholding at 0.5
        true_flat = true_adj_matrix.values.flatten()
        pred_flat = reconstructed_adjacency.flatten()
        accuracy = accuracy_score(true_flat, (pred_flat > 0.5).astype(int))

        # Compute EPR with the existing function
        epr_final = calculate_early_precision_rate(reconstructed_adjacency, true_adj_matrix)

        print(f"\nFinal ROC-AUC: {roc_auc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, "
              f"Accuracy: {accuracy:.4f}, EPR: {epr_final:.4f}")

        # plot_precision_recall_curve(true_adj_matrix, reconstructed_adjacency)

        ###################################
        #      Unique Edge Analysis       #
        ###################################
        # gt_df = pd.read_csv(NETWORK_FILE)
        # num_gt_edges = len(gt_df)
        print(f"\nNumber of ground truth unique edges (from network file): {num_gt_edges}")

        ###################################
        #      Simplified Top-20% Analysis
        ###################################
        # n = reconstructed_adjacency.shape[0]
        # predicted_edges = [
        #     (i, j, reconstructed_adjacency[i, j])
        #     for i in range(n) for j in range(i + 1, n)
        #     if reconstructed_adjacency[i, j] > 0.3
        # ]
        # predicted_edges.sort(key=lambda x: x[2], reverse=True)
        # top_percentage = 0.2
        # num_top_edges = int(len(predicted_edges) * top_percentage)
        # top_predicted_edges = predicted_edges[:num_top_edges]
        # overlap_count = sum(1 for i, j, score in top_predicted_edges if true_adj_matrix.values[i, j] == 1)

        print(f"Number of overlapping edges in top 20% predictions: {overlap_count}")

        # Assuming `pcc_matrix` is your Pearson Correlation Coefficient adjacency matrix
        pcc_matrix = construct_adjacency_matrix(expr_data, threshold=THRESHOLD)

        # Run the comparison
        pcc_overlap, pred_overlap = calculate_whole_network_overlap(reconstructed_adjacency, true_adj_matrix,
                                                                    pcc_matrix)

        ###################################
        #      Visualization Calls        #
        ###################################
        # visualize_grn(reconstructed_adjacency, gene_names)
        # visualize_embeddings(model, expr_tensor, edge_index, method="tsne")
        # hub_genes = identify_hub_genes(reconstructed_adjacency, gene_names, top_k=10, threshold=0.5)
        # visualize_grn_with_hubs(reconstructed_adjacency, gene_names, hub_genes, threshold=0.5)

os.system("afplay /System/Library/Sounds/Glass.aiff")
