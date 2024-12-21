import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from torch_geometric.nn import GATConv
import seaborn as sns


# Define the GAT-VGAE model
class GAT_VGAE(nn.Module):
    def __init__(self, num_features, num_neurons, embedding_size, num_heads, num_nodes, alpha=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.alpha = alpha  # Perturbation factor for initial graph structure
        self.gat1 = GATConv(num_features, num_neurons, heads=num_heads, concat=True)
        self.gat2 = GATConv(num_neurons * num_heads, embedding_size, heads=1, concat=False)
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


# Load and preprocess the gene expression data
expr_df = pd.read_csv('/Users/rameshsubramani/Desktop/GAAE/data/expr.csv')
gene_names = expr_df['Unnamed: 0'].values
expr_values = expr_df.drop(columns=['Unnamed: 0']).values

# Normalize the expression data
scaler = StandardScaler()
normalized_expr = scaler.fit_transform(expr_values.T).T

# Apply PCA for dimensionality reduction (reduce to 16 dimensions as an example)
pca = PCA(n_components=16)
reduced_expr = pca.fit_transform(normalized_expr.T).T  # Shape will be (num_genes, 16)

# Step 1: Compute the true adjacency matrix based on PCC
pcc_matrix = np.corrcoef(reduced_expr)  # Compute pairwise PCC between genes
threshold = 0.5  # Define a threshold for strong correlation
true_adj_matrix = (pcc_matrix > threshold).astype(int)  # Create binary adjacency matrix based on PCC

# Step 2: Initialize the model with hybrid graph structure (PCC matrix + small random perturbation)
num_nodes = reduced_expr.shape[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
expr_tensor = torch.FloatTensor(reduced_expr).to(device)

# Add small random perturbation to the PCC matrix (e.g., 10% noise)
noise = np.random.rand(num_nodes, num_nodes) < 0.1  # 10% noise
initial_adj_matrix = np.logical_or(true_adj_matrix, noise)  # Combine true adjacency matrix with noise

# Convert edge indices for the hybrid graph structure
edge_index = torch.tensor(np.array(np.where(initial_adj_matrix == 1)), dtype=torch.long).to(device)

# Step 3: Define the model
vgae_model = GAT_VGAE(
    num_neurons=64,
    num_features=reduced_expr.shape[1],
    embedding_size=16,
    num_nodes=num_nodes,
    num_heads=4,
    alpha=0.1  # Perturbation factor
)

# Initialize optimizer
optimizer = torch.optim.Adam(vgae_model.parameters(), lr=0.0001)
vgae_model.to(device)

# Step 4: Train the model
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Get the predicted adjacency matrix (from the model's decoder)
    x_hat = vgae_model(edge_index, expr_tensor).view(-1).T
    true_adj_matrix_flat = torch.FloatTensor(true_adj_matrix).flatten().to(device)

    # Calculate the reconstruction loss using the PCC-based true adjacency matrix
    bce_loss = F.binary_cross_entropy(x_hat, true_adj_matrix_flat)

    # Regularization to ensure the learned graph doesn't deviate too far from the true PCC matrix
    adj_loss = F.mse_loss(x_hat, true_adj_matrix_flat)

    # Optionally add KL divergence or other losses here
    kl_loss = -0.5 * torch.sum(1 + vgae_model.log_var - vgae_model.mu.pow(2) - vgae_model.log_var.exp()) / num_nodes
    total_loss = bce_loss + 0.9 * kl_loss + 0.1 * adj_loss  # Add adjacency loss for regularization

    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, BCE Loss: {bce_loss:.4f}, KL Loss: {kl_loss:.4f}, Adj Loss: {adj_loss:.4f}, Total Loss: {total_loss:.4f}")

# Step 5: Evaluate the model
# After training, you can evaluate the model's reconstruction quality by comparing the predicted and true adjacency matrices
reconstructed_adj_matrix = (x_hat.detach().cpu().numpy() > 0.5).astype(int)
roc_auc = roc_auc_score(true_adj_matrix.flatten(), reconstructed_adj_matrix.flatten())
print(f"ROC-AUC Score: {roc_auc:.4f}")

import matplotlib.pyplot as plt

# Reshape the reconstructed adjacency matrix to the correct shape (num_nodes x num_nodes)
reconstructed_adj_matrix_reshaped = reconstructed_adj_matrix.reshape(num_nodes, num_nodes)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(true_adj_matrix, cmap="Blues", cbar=True)
plt.title("Original Adjacency Matrix")
plt.xlabel("Genes")
plt.ylabel("Genes")

plt.subplot(1, 2, 2)
sns.heatmap(reconstructed_adj_matrix_reshaped, cmap="Blues", cbar=True)
plt.title("Reconstructed Adjacency Matrix")
plt.xlabel("Genes")
plt.ylabel("Genes")

plt.tight_layout()
plt.show()
