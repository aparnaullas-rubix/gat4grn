#######################################
#  Data Preprocessing & Adjacency     #
#######################################
import json
import uuid

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


#######################################
#         Global Variables            #
#######################################
# Unique Run ID
run_id = str(uuid.uuid4())

# Load configuration from file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

THRESHOLD = config['threshold']
NOISE_FACTOR = config['noise_factor']



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