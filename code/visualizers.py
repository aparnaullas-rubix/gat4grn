#######################################
#         Visualization Functions     #
#######################################
import networkx as nx
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve


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


# def visualize_embeddings(model, expr_tensor, edge_index, method="tsne"):
#     model.eval()
#     with torch.no_grad():
#         z = model.encode(edge_index, expr_tensor).cpu().numpy()
#     if method == "pca":
#         reduced = PCA(n_components=2).fit_transform(z)
#     else:
#         reduced = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(z)
#     plt.figure(figsize=(8, 6))
#     plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, c='blue', edgecolors='black')
#     for i, label in enumerate(gene_names):
#         plt.text(reduced[i, 0], reduced[i, 1], label, fontsize=8, alpha=0.75)
#     plt.title(f"Gene Embeddings Visualization ({method.upper()})")
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.grid()
#     plt.show()


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
