from torch_geometric.nn import GATConv, GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch


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




class GCNLinkAutoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(GCNLinkAutoencoder, self).__init__()

        self.encoder1 = GCNConv(in_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, hidden_dim)
        self.encoder3 = GCNConv(hidden_dim, hidden_dim)  # Added third GCN layer
        self.latent_layer = GCNConv(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.decoder1 = GCNConv(latent_dim, hidden_dim)
        self.decoder2 = nn.Linear(hidden_dim, in_dim)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, edge_index):
        x1 = self.encoder1(x, edge_index)
        x1 = self.leaky_relu(x1)
        x1 = self.dropout(x1)

        x2 = self.encoder2(x1, edge_index)
        x2 = self.leaky_relu(x2)
        x2 = self.dropout(x2)

        x3 = self.encoder3(x2, edge_index)
        x3 = self.leaky_relu(x3)
        x3 = self.dropout(x3)

        x3 = x3 + x2

        z = self.latent_layer(x3, edge_index)
        x_reconstructed = self.decoder1(z, edge_index)
        x_reconstructed = self.leaky_relu(x_reconstructed)
        reconstructed = self.decoder2(x_reconstructed)

        reconstructed = reconstructed + x
        return reconstructed, z





class VGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(VGAE, self).__init__()

        self.encoder1 = GCNConv(in_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, hidden_dim)
        self.mean_layer = GCNConv(hidden_dim, latent_dim)
        self.log_std_layer = GCNConv(hidden_dim, latent_dim)

        self.decoder = InnerProductDecoder()

    def forward(self, x, edge_index):
        # Encoder
        h = F.relu(self.encoder1(x, edge_index))
        h = F.relu(self.encoder2(h, edge_index))

        # Variational Latent Space
        z_mean = self.mean_layer(h, edge_index)
        z_log_std = self.log_std_layer(h, edge_index)
        z = self.reparameterize(z_mean, z_log_std)

        # Decoder
        adj_pred = self.decoder(z, edge_index)

        return adj_pred, z_mean, z_log_std

    def reparameterize(self, mean, log_std):
        std = torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        return mean + eps * std


class InnerProductDecoder(nn.Module):
    def forward(self, z, edge_index):
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=-1)

