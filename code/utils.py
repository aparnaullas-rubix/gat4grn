import numpy as np


#######################################
#     Early Stopping     #
#######################################
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




