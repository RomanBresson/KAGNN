import argparse
import json
import numpy as np
import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.loader import DataLoader

from graph_classification_utils import *

from model import KAGIN

# Argument parser
parser = argparse.ArgumentParser(description='KAGIN')
parser.add_argument('--dataset', default='MUTAG', help='Dataset name')
parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training')
parser.add_argument('--nb_gnn_layers', type=int, default=4, help='Input batch size for training')
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

all_best_hyperparams = []
all_best_sizes = []
accs = []
curr_config = 0

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

for it in range(0,10):
    torch.cuda.empty_cache()
    train_index = splits[it]['model_selection'][0]['train']
    val_index = splits[it]['model_selection'][0]['validation']
    
    val_dataset = dataset[val_index]
    train_dataset = dataset[train_index]

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=100)
    best_hyperparams = study.best_params

    test_accs = []
    for _ in range(3):
        train_index = splits[it]['model_selection'][0]['train']
        val_index = splits[it]['model_selection'][0]['validation']
        test_index = splits[it]['test']

        test_dataset = dataset[test_index]
        val_dataset = dataset[val_index]
        train_dataset = dataset[train_index]

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        print('---------------- Split {} ----------------'.format(it))
        test_acc = train_model_with_parameters(best_hyperparams['lr'], best_hyperparams['hidden_layers'], best_hyperparams['hidden_dim'], best_hyperparams['dropout'], best_hyperparams['grid_size'], best_hyperparams['spline_order'], train_loader, val_loader, test_loader)
        test_accs.append(test_acc)
    
    accs.append(np.mean(test_accs))
    all_best_hyperparams.append(best_hyperparams)
    print(accs)
    print(all_best_hyperparams)
    #all_best_sizes.append(count_params(model))
    with open(log_file, 'a') as file:
        file.write(f'SPLIT {it}\n')
        file.write(f'Accuracies {accs}\n')
        file.write(f'Params {all_best_hyperparams}\n')
        #file.write(f'Sizes {all_best_sizes}\n')
        file.write('\n')



accs = torch.tensor(accs)
print('---------------- Final Result ----------------')
print('Mean: {:7f}, Std: {:7f}'.format(accs.mean(), accs.std()))
print(all_best_hyperparams)
#print(all_best_sizes)
with open(log_file, 'a') as file:
    file.write(f'SPLIT {it}\n')
    file.write(f'Accuracies {accs}\n')
    file.write(f'Params {all_best_hyperparams}')
    #file.write(f'Sizes {all_best_sizes}')
