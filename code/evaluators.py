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
import json
import uuid

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from loggers import log_epoch_info, log_run_info
import torch.nn.functional as F


#######################################
#         Global Variables            #
#######################################
# Unique Run ID
run_id = str(uuid.uuid4())

# Load configuration from file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# File paths and hyperparameters from config
NETWORK_FILE = config['network_file']
DATASET = config['dataset']

K_FRACTION = config.get('k_fraction', 0.1)



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
