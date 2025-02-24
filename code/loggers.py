#######################################
#         Logging Functions           #
#######################################
import csv
import json
import os
import uuid

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
