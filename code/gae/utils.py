import numpy as np
import csv

import torch.nn.functional as F

import torch
import torch.nn as nn

from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score


class EarlyStopping:
    def __init__(self, patience=10, delta=0):

        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


def log_training_data(csv_file, iteration, epochs, model, dropout_rate, relu_neg_slope, in_dim, hidden_dim, latent_dim,
                      op_learning_rate, op_weight_decay, norm_penality, loss, eval_threshold, precision,
                      recall, f1, aucroc, top_k, message):
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [iteration, epochs, model, dropout_rate, relu_neg_slope, in_dim, hidden_dim, latent_dim,
             op_learning_rate, op_weight_decay, norm_penality, loss, eval_threshold, precision,
             recall, f1, aucroc, top_k, message])


# Mutual Information
def compute_adjacency_matrix(xmn, theta=0.05):
    number_of_genes = xmn.shape[0]
    similarity_matrix = np.zeros((number_of_genes, number_of_genes))
    print("Xmn shape: ", xmn.shape)  # This will show the number of rows and columns
    for i in range(number_of_genes):
        print(i)
        for j in range(number_of_genes):
            if i != j:
                similarity_matrix[i, j] = mutual_info_score(xmn[i, :], xmn[j, :])
                similarity_matrix[j, i] = mutual_info_score(xmn[i, :], xmn[j, :])

    np.fill_diagonal(similarity_matrix, 0)
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
                mi = mutual_info_score(xmn[i, :], xmn[j, :])
                pcc = pearsonr(xmn[i, :], xmn[j, :])[0]
                similarity_matrix[i, j] = alpha * mi + (1 - alpha) * pcc

    np.fill_diagonal(similarity_matrix, 0)
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


def cap_outliers(data, lower_percentile=1, upper_percentile=99):
    lower_bound = np.percentile(data, lower_percentile, axis=0)
    upper_bound = np.percentile(data, upper_percentile, axis=0)
    for i in range(data.shape[1]):
        data[:, i] = np.clip(data[:, i], lower_bound[i], upper_bound[i])

    return data


loss_fn = nn.L1Loss()


def loss_function(reconstructed_adj, z_mean, z_log_std, adj_labels):
    # Reconstruction Loss (binary cross entropy)
    recon_loss = F.binary_cross_entropy_with_logits(reconstructed_adj, adj_labels)

    # KL Divergence Loss
    kl_loss = -0.5 * torch.mean(1 + z_log_std - z_mean.pow(2) - z_log_std.exp())

    return recon_loss + kl_loss


def compute_loss(model, data):
    reconstructed, _ = model(data.x, data.edge_index)
    return loss_fn(reconstructed, data.x)  # This can be replaced with whatever loss you need
