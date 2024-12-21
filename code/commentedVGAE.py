import torch  # Importing the PyTorch library for tensor operations and deep learning
import torch.nn as nn  # Importing the neural network module from PyTorch
import torch.nn.functional as F  # Importing functional operations for neural networks
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score  # Importing metrics for model evaluation
from sklearn.preprocessing import StandardScaler  # Importing scaler for data normalization
from torch_geometric.nn import GATConv  # Importing Graph Attention Network convolution layer
import pandas as pd  # Importing pandas for data manipulation and analysis
import numpy as np  # Importing NumPy for numerical operations
import seaborn as sns  # Importing Seaborn for data visualization
import csv  # Importing CSV module for file operations
from sklearn.metrics import roc_curve, auc  # Importing additional metrics for ROC analysis
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
import uuid  # Importing UUID for generating unique identifiers
import os  # Importing OS module for interacting with the operating system

run_id = str(uuid.uuid4())  # Generating a unique run ID for tracking experiments


def log_run_info(run_id, neurons, embedding_size, lr, heads, roc_auc, precision_ground_truth, recall_ground_truth,
                 f1_ground_truth, total_ground_truth,
                 total_predicted_edges, true_positives_top_20_ground_truth, top_predicted_ground_truth_overlap, model):
    run_info_path = '/Users/rameshsubramani/Desktop/GAAE/files/run_info_vgae.csv'  # Path to save run information
    header = ['Run ID', 'Num Neurons', 'Embedding Size', 'Learning Rate', 'Num Heads', 'ROC-AUC', 'Precision', 'Recall',
              'F1', '#Edges in GT', '#Predicted edges', 'Top 20% overlap', '#Overlapping edges', 'Model']  # Header for CSV file

    run_info_data = [[run_id, neurons, embedding_size, lr, heads, roc_auc, precision_ground_truth, recall_ground_truth,
                      f1_ground_truth, total_ground_truth, total_predicted_edges, true_positives_top_20_ground_truth,
                      top_predicted_ground_truth_overlap, model]]  # Data to log

    if not os.path.exists(run_info_path):  # Check if the file exists
        with open(run_info_path, mode='w', newline='') as file:  # Open file in write mode
            writer = csv.writer(file)  # Create a CSV writer object
            writer.writerow(header)  # Write the header to the CSV file
    with open(run_info_path, mode='a', newline='') as file:  # Open file in append mode
        writer = csv.writer(file)  # Create a CSV writer object
        writer.writerows(run_info_data)  # Write the run data to the CSV file


def log_epoch_info(run_id, epoch, loss):
    epoch_info_path = '/Users/rameshsubramani/Desktop/GAAE/files/epoch_info_vgae.csv'  # Path to save epoch information
    header = ['Run ID', 'Epoch', 'Loss']  # Header for CSV file

    epoch_info_data = [[run_id, epoch, loss.item()]]  # Data to log for the epoch

    if not os.path.exists(epoch_info_path):  # Check if the file exists
        with open(epoch_info_path, mode='w', newline='') as file:  # Open file in write mode
            writer = csv.writer(file)  # Create a CSV writer object
            writer.writerow(header)  # Write the header to the CSV file
    with open(epoch_info_path, mode='a', newline='') as file:  # Open file in append mode
        writer = csv.writer(file)  # Create a CSV writer object
        writer.writerows(epoch_info_data)  # Write the epoch data to the CSV file


class GAT_VGAE(nn.Module):  # Defining the Graph Attention Variational Graph Autoencoder class
    def __init__(self, num_features, num_neurons, embedding_size, num_heads, num_nodes):
        super().__init__()  # Initialize the parent class
        self.num_nodes = num_nodes  # Store the number of nodes
        self.gat1 = GATConv(num_features, num_neurons, heads=num_heads, concat=True)  # First GAT layer
        self.gat2 = GATConv(num_neurons * num_heads, embedding_size, heads=1, concat=False)  # Second GAT layer
        self.num_neurons = num_neurons  # Store the number of neurons
        self.num_features = num_features  # Store the number of features
        self.embedding_size = embedding_size  # Store the embedding size
        self.num_heads = num_heads  # Store the number of attention heads
        self.num_nodes = num_nodes  # Store the number of nodes

        self.mu_net = nn.Linear(embedding_size, embedding_size)  # Linear layer for mean
        self.log_var_net = nn.Linear(embedding_size, embedding_size)  # Linear layer for log variance

        self.decoder = nn.Linear(embedding_size, num_nodes * num_nodes)  # Decoder layer to reconstruct the adjacency matrix

    def reparameterize(self, mu, log_var):  # Reparameterization trick for variational inference
        std = torch.exp(0.5 * log_var)  # Calculate standard deviation
        eps = torch.randn_like(std)  # Sample from a standard normal distribution
        return mu + eps * std  # Return the reparameterized latent variable

    def encode(self, edge_index, x):  # Encoding function to generate embeddings
        hidden = F.relu(self.gat1(x, edge_index))  # Apply first GAT layer with ReLU activation
        embedding = self.gat2(hidden, edge_index)  # Apply second GAT layer

        mu = self.mu_net(embedding)  # Calculate mean from embedding
        log_var = self.log_var_net(embedding)  # Calculate log variance from embedding

        z = self.reparameterize(mu, log_var)  # Reparameterize to get latent variable
        self.mu, self.log_var = mu, log_var  # Store mean and log variance for later use
        return z  # Return the latent variable

    def decode(self, z):  # Decoding function to reconstruct the adjacency matrix
        z = z.mean(dim=0)  # Average the latent variable across the batch
        decoded = self.decoder(z)  # Decode the latent variable
        decoded = torch.sigmoid(decoded)  # Apply sigmoid activation to get probabilities
        decoded = decoded.view(self.num_nodes, self.num_nodes)  # Reshape to adjacency matrix
        return decoded  # Return the reconstructed adjacency matrix

    def forward(self, edge_index, x):  # Forward pass through the model
        z = self.encode(edge_index, x)  # Encode the input features
        return self.decode(z)  # Decode the latent representation


def construct_adjacency_matrix(expr_data, threshold=0.7):  # Function to construct adjacency matrix from expression data
    corr_matrix = np.corrcoef(expr_data)  # Calculate correlation matrix
    adj_matrix = (corr_matrix > threshold).astype(float)  # Create adjacency matrix based on threshold
    np.fill_diagonal(adj_matrix, 0)  # Set diagonal to zero (no self-loops)
    return adj_matrix  # Return the adjacency matrix


expr_df = pd.read_csv('/Users/rameshsubramani/Desktop/GAAE/data/expr.csv')  # Load expression data from CSV
gene_names = expr_df['Unnamed: 0'].values  # Extract gene names from the first column
expr_values = expr_df.drop(columns=['Unnamed: 0']).values  # Extract expression values, dropping the gene names column

scaler = StandardScaler()  # Initialize the scaler for normalization
normalized_expr = scaler.fit_transform(expr_values.T).T  # Normalize the expression values
adj_matrix = construct_adjacency_matrix(normalized_expr, threshold=0.5)  # Construct adjacency matrix from normalized data

adj_matrix_tensor = torch.FloatTensor(adj_matrix)  # Convert adjacency matrix to PyTorch tensor
expr_tensor = torch.FloatTensor(normalized_expr)  # Convert normalized expression data to PyTorch tensor

num_nodes = adj_matrix.shape[0]  # Get the number of nodes from the adjacency matrix
vgae_model = GAT_VGAE(  # Initialize the GAT_VGAE model
    num_neurons=64,  # Set number of neurons
    num_features=normalized_expr.shape[1],  # Set number of features
    embedding_size=16,  # Set embedding size
    num_nodes=num_nodes,  # Set number of nodes
    num_heads=4  # Set number of attention heads
)
optimizer = torch.optim.Adam(vgae_model.parameters(), lr=0.0005)  # Initialize Adam optimizer for model parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available
vgae_model.to(device)  # Move model to the specified device

edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)  # Create edge index from adjacency matrix
edge_index = edge_index.to(device)  # Move edge index to the specified device

x_features = torch.FloatTensor(normalized_expr).to(device)  # Convert normalized expression data to tensor and move to device

bce_losses = []  # List to store binary cross-entropy losses
kl_losses = []  # List to store KL divergence losses
total_losses = []  # List to store total losses

num_epochs = 200  # Set number of training epochs
for epoch in range(num_epochs):  # Loop over epochs
    optimizer.zero_grad()  # Zero the gradients before backward pass

    x_hat = vgae_model(edge_index, x_features).view(-1).T  # Forward pass to get predicted adjacency matrix
    adj_matrix_flat = adj_matrix_tensor.view(-1).to(device)  # Flatten the true adjacency matrix

    bce_loss = F.binary_cross_entropy(x_hat, adj_matrix_flat)  # Calculate binary cross-entropy loss

    kl_weight = 0.12  # Set weight for KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + vgae_model.log_var - vgae_model.mu.pow(2) - vgae_model.log_var.exp()) / \
              adj_matrix.shape[0]  # Calculate KL divergence loss
    loss = bce_loss + kl_weight * kl_loss  # Combine losses

    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update model parameters

    bce_losses.append(bce_loss.item())  # Append BCE loss to the list
    kl_losses.append(kl_loss.item())  # Append KL loss to the list
    total_losses.append(loss.item())  # Append total loss to the list

    if (epoch + 1) % 10 == 0:  # Print loss every 10 epochs
        print(
            f"Epoch {epoch + 1}/{num_epochs}, BCE Loss: {bce_loss:.4f}, KL Loss: {kl_loss:.4f}, Total Loss: {loss:.4f}")

inferred_adjacency = (x_hat.detach().cpu().numpy() > 0.5).astype(int)  # Threshold the predicted adjacency matrix

net_df = pd.read_csv('/Users/rameshsubramani/Desktop/GAAE/data/net.csv')  # Load true adjacency data from CSV
true_adj_matrix = pd.DataFrame(0, index=gene_names, columns=gene_names)  # Initialize true adjacency matrix

for _, row in net_df.iterrows():  # Iterate over rows in the true adjacency data
    gene1, gene2 = row['Gene1'], row['Gene2']  # Extract gene pairs
    true_adj_matrix.at[gene1, gene2] = 1  # Set adjacency for gene1 and gene2
    true_adj_matrix.at[gene2, gene1] = 1  # Set adjacency for gene2 and gene1 (undirected graph)

roc_auc = roc_auc_score(true_adj_matrix.values.flatten(), inferred_adjacency.flatten())  # Calculate ROC-AUC score
print(f"ROC-AUC Score: {roc_auc:.4f}")  # Print the ROC-AUC score

# Convert the adjacency matrix tensor to a NumPy array for further processing
original_adjacency = adj_matrix_tensor.numpy()

# Detach the tensor from the computation graph and move it to CPU, then convert to NumPy array
reconstructed_adjacency = x_hat.detach().cpu().numpy()

# Infer the number of nodes by taking the square root of the size of the reconstructed adjacency matrix
num_nodes = int(np.sqrt(reconstructed_adjacency.size))

# Reshape the reconstructed adjacency array into a square matrix of size num_nodes x num_nodes
reconstructed_adjacency = reconstructed_adjacency.reshape(num_nodes, num_nodes)

# Create a binary matrix of predicted edges based on a threshold of 0.5
predicted_edges = (reconstructed_adjacency > 0.5).astype(int)

# Extract the ground truth edges from the true adjacency matrix
ground_truth_edges = true_adj_matrix.values

# Flatten the reconstructed adjacency matrix to a 1D array for easier comparison
predicted_scores = reconstructed_adjacency.flatten()

# Flatten the original adjacency matrix for comparison
computed_flat = adj_matrix.flatten()

# Flatten the ground truth edges for comparison
ground_truth_flat = ground_truth_edges.flatten()

# Create a binary representation of predicted scores based on a threshold of 0.5
predicted_binary = (predicted_scores > 0.5).astype(int)

# Calculate the total number of predicted edges
total_predicted_edges = np.sum(predicted_binary)

# Calculate the total number of edges in the computed adjacency matrix
total_computed_edges = np.sum(computed_flat)

# Calculate the total number of edges in the ground truth adjacency matrix
total_ground_truth = np.sum(ground_truth_flat)

# Print a separator for clarity in the output
print("####################################################")

# Output the total number of predicted edges
print(f"Total Predicted Edges: {total_predicted_edges}")

# Output the total number of edges in the computed adjacency matrix
print(f"Total PCC computed edges (Training adjacency matrix): {total_computed_edges}")

# Calculate true positives, false positives, and false negatives for the computed adjacency matrix
true_positives_computed = np.sum((predicted_binary == 1) & (computed_flat == 1))
false_positives_computed = np.sum((predicted_binary == 1) & (computed_flat == 0))
false_negatives_computed = np.sum((predicted_binary == 0) & (computed_flat == 1))

# Calculate true positives, false positives, and false negatives for the ground truth adjacency matrix
true_positives_ground_truth = np.sum((predicted_binary == 1) & (ground_truth_flat == 1))
false_positives_ground_truth = np.sum((predicted_binary == 1) & (ground_truth_flat == 0))
false_negatives_ground_truth = np.sum((predicted_binary == 0) & (ground_truth_flat == 1))

# Sort the predicted scores in descending order to identify the top predictions
sorted_indices = np.argsort(-predicted_scores)

# Determine the number of top predictions to consider (20% of total predictions)
top_20_percent = int(0.2 * len(predicted_scores))

# Get the indices of the top predicted edges
top_indices = sorted_indices[:top_20_percent]

# Extract the top predicted edges from the computed adjacency matrix
top_predicted_computed_overlap = computed_flat[top_indices]

# Extract the top predicted edges from the ground truth adjacency matrix
top_predicted_ground_truth_overlap = ground_truth_flat[top_indices]

# Calculate true positives for the top 20% of predicted edges in the computed adjacency matrix
true_positives_top_20_computed = np.sum(top_predicted_computed_overlap)

# Calculate true positives for the top 20% of predicted edges in the ground truth adjacency matrix
true_positives_top_20_ground_truth = np.sum(top_predicted_ground_truth_overlap)

# Print a comparison header for computed adjacency matrix results
print("Comparison with Computed Adjacency Matrix (Training):")

# Output the true positives for all predicted edges in the computed adjacency matrix
print(f"True Positives (All Predicted): {true_positives_computed}")

# Output the false positives for the computed adjacency matrix
print(f"False Positives: {false_positives_computed}")

# Output the false negatives for the computed adjacency matrix
print(f"False Negatives: {false_negatives_computed}")

# Output the true positives for the top 20% ranked edges in the computed adjacency matrix
print(f"True Positives (Top 20% Ranked Edges): {true_positives_top_20_computed}")

# Print a separator for clarity in the output
print("####################################################")

# Output the total number of predicted edges
print(f"Total Predicted Edges: {total_predicted_edges}")

# Output the total number of edges in the ground truth adjacency matrix
print(f"Total (External Network) Ground Truth Edges: {total_ground_truth}")

# Print a comparison header for ground truth edges results
print("Comparison with (External Network) Ground Truth Edges:")

# Output the true positives for all predicted edges in the ground truth adjacency matrix
print(f"True Positives (All Predicted): {true_positives_ground_truth}")

# Output the false positives for the ground truth adjacency matrix
print(f"False Positives: {false_positives_ground_truth}")

# Output the false negatives for the ground truth adjacency matrix
print(f"False Negatives: {false_negatives_ground_truth}")

# Output the true positives for the top 20% ranked edges in the ground truth adjacency matrix
print(f"True Positives (Top 20% Ranked Edges): {true_positives_top_20_ground_truth}")

# Print a separator for clarity in the output
print("####################################################")

# Calculate precision, recall, and F1 score for the computed adjacency matrix
precision_computed = precision_score(computed_flat, predicted_binary)
recall_computed = recall_score(computed_flat, predicted_binary)
f1_computed = f1_score(computed_flat, predicted_binary)

# Calculate precision, recall, and F1 score for the ground truth adjacency matrix
precision_ground_truth = precision_score(ground_truth_flat, predicted_binary)
recall_ground_truth = recall_score(ground_truth_flat, predicted_binary)
f1_ground_truth = f1_score(ground_truth_flat, predicted_binary)

# Print a separator for clarity in the output
print("####################################################")

# Output precision for the computed adjacency matrix
print(f"Precision (Computed Adjacency Matrix): {precision_computed:.4f}")

# Output recall for the computed adjacency matrix
print(f"Recall (Computed Adjacency Matrix): {recall_computed:.4f}")

# Output F1 score for the computed adjacency matrix
print(f"F1 Score (Computed Adjacency Matrix): {f1_computed:.4f}")

# Print a separator for clarity in the output
print("####################################################")

# Output precision for the ground truth adjacency matrix
print(f"Precision (Ground Truth Adjacency Matrix): {precision_ground_truth:.4f}")

# Output recall for the ground truth adjacency matrix
print(f"Recall (Ground Truth Adjacency Matrix): {recall_ground_truth:.4f}")

# Output F1 score for the ground truth adjacency matrix
print(f"F1 Score (Ground Truth Adjacency Matrix): {f1_ground_truth:.4f}")

# Print a separator for clarity in the output
print("####################################################")

# Log the run information including various metrics and model parameters
log_run_info(run_id, vgae_model.num_neurons, vgae_model.embedding_size, optimizer.param_groups[0]['lr'],
             vgae_model.num_heads, roc_auc,
             precision_ground_truth, recall_ground_truth, f1_ground_truth, total_ground_truth, total_predicted_edges,
             true_positives_top_20_ground_truth, top_predicted_ground_truth_overlap, vgae_model)