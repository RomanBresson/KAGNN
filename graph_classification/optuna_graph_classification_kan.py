import argparse
import json
import numpy as np
import optuna

import torch
from torch_geometric.loader import DataLoader

from graph_classification_utils import *

from model import KAGIN

def train_model_with_parameters(lr, hidden_layers, hidden_dim, dropout, grid_size, spline_order, train_loader, val_loader, test_loader=None):
    model = KAGIN(args.nb_gnn_layers, dataset_num_features, hidden_dim, dataset.num_classes, hidden_layers, grid_size, spline_order, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    early_stopper = EarlyStopper(patience=args.patience)
    for epoch in range(1, args.epochs+1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = val(model, val_loader, device)
        if best_val_loss >= val_loss:
            best_val_loss = val_loss 
            if test_loader is not None:
                test_acc = test(model, test_loader, device)
        if early_stopper.early_stop(val_loss):
            print(f"Stopped at epoch {epoch}")
            break
    if test_loader is None:
        return best_val_loss
    else:
        return test_acc

def objective(trial, train_loader, val_loader):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
    hidden_dim = trial.suggest_int('hidden_dim', 2, 128)
    grid_size = trial.suggest_int('grid_size', 1, 16)
    spline_order = trial.suggest_int('spline_order', 1, 8)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    best_val_loss = train_model_with_parameters(lr, hidden_layers, hidden_dim, dropout, grid_size, spline_order, train_loader, val_loader)
    return best_val_loss

# Argument parser
parser = argparse.ArgumentParser(description='KAGIN')
parser.add_argument('--dataset', default='MUTAG', help='Dataset name')
parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training')
parser.add_argument('--nb_gnn_layers', type=int, default=4, help='Number of message passing layers')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train')
parser.add_argument('--patience', type=int, default=20, help='Patience of early stopping')
args = parser.parse_args()

use_node_attr = False
if args.dataset == 'ENZYMES' or args.dataset == 'PROTEINS_full':
    use_node_attr = True

if args.dataset in unlabeled_datasets:
    dataset = TUDataset(root='datasets/'+args.dataset, name=args.dataset, transform=Degree())
else:
    dataset = TUDataset(root='datasets/'+args.dataset, name=args.dataset, use_node_attr=use_node_attr)

dataset_num_features = dataset[0].x.shape[1]

with open('data_splits/'+args.dataset+'_splits.json','rt') as f:
    for line in f:
        splits = json.loads(line)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(device)

log_file = f'logs/KAN_{args.dataset}'

parameters_finder(train_model_with_parameters, objective, log_file, splits, dataset, args)