import argparse
import json
import numpy as np

import torch
from torch_geometric.loader import DataLoader

from graph_classification_utils import *
import time

from model import FASTKAGIN, KAGIN, GIN

# Argument parser
parser = argparse.ArgumentParser(description='FASTKAGIN')
parser.add_argument('--dataset', default='NCI1', help='Dataset name')
parser.add_argument('--batch-size', type=int, default=128, help='Input batch size for training')
parser.add_argument('--nb_gnn_layers', type=int, default=5, help='Number of message passing layers')
parser.add_argument('--hidden_layers', type=int, default=2, help='Size of hidden layers')
parser.add_argument('--model', type=str, default='mlp', help="Model to use: mlp/kan/fkan")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

use_node_attr = False
if args.dataset == 'ENZYMES' or args.dataset == 'PROTEINS_full':
    use_node_attr = True

if args.dataset in unlabeled_datasets:
    dataset = TUDataset(root='datasets/'+args.dataset, name=args.dataset, transform=Degree())
else:
    if device == torch.device("cuda"):
        dataset = TUDataset(root='datasets/'+args.dataset, name=args.dataset, use_node_attr=use_node_attr, transform=to_cuda())
    else:
        dataset = TUDataset(root='datasets/'+args.dataset, name=args.dataset, use_node_attr=use_node_attr)

with open('data_splits/'+args.dataset+'_splits.json','rt') as f:
    for line in f:
        splits = json.loads(line)

log_file = f'logs/time_{args.dataset}'

curr_config = 0

def make_model(model_name, params, dataset):
    lr, hidden_layers, hidden_dim, dropout, grid_size, spline_order = params['lr'], params['hidden_layers'], params['hidden_dim'], params['dropout'], params['grid_size'], params['spline_order']
    if model_name=='kan':
        model = KAGIN(args.nb_gnn_layers, dataset.num_features, hidden_dim, dataset.num_classes, hidden_layers, grid_size, spline_order, dropout).to(device)
    elif model_name=='fastkan':
        model = FASTKAGIN(args.nb_gnn_layers, dataset.num_features, hidden_dim, dataset.num_classes, hidden_layers, grid_size, dropout).to(device)
    elif model_name=='mlp':
        model = GIN(args.nb_gnn_layers, dataset.num_features, hidden_dim, hidden_layers, dataset.num_classes, dropout).to(device)
    return model

def go(model_name, params, dataset):
    nb_splits = 3
    Accs = []
    Spents = []
    for it in range(nb_splits):
        model = make_model(model_name, params, dataset)
        dataset_num_features = dataset[it].x.shape[1]
        dataset_num_classes = dataset.num_classes
        train_index = splits[it]['model_selection'][0]['train']
        train_dataset = dataset[train_index]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_index = splits[it]['model_selection'][0]['validation']
        val_dataset = dataset[val_index]
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_index = splits[it]['test']
        test_dataset = dataset[test_index]
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        test_acc, spent, param_count = train_model(model, train_loader, val_loader, test_loader, params['lr'])
        Accs.append(test_acc)
        Spents.append(spent)
    return(Accs, Spents, param_count)

def train_model(model, train_loader, val_loader, test_loader, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    early_stopper = EarlyStopper(patience=args.patience)
    time0 = time.time()
    for epoch in range(1, args.epochs+1):
        train(model, train_loader, optimizer, device)
        val_loss = val(model, val_loader, device)
        if best_val_loss >= val_loss:
            best_val_loss = val_loss
            if test_loader is not None:
                test_acc = test(model, test_loader, device)
        if early_stopper.early_stop(val_loss):
            print(f"Stopped at epoch {epoch}")
            break
    time1 = time.time()
    spent = time1-time0
    print(test_acc, spent)
    return test_acc, spent, count_params(model)

params = {'lr':0.001, 'dropout':0.2, 'grid_size':0, 'spline_order':0}

time_gcn  = []
time_kan  = []
time_fkan = []

log_prefix = 'time_'

for hidden_dim in [2, 4, 8, 16, 64, 256, 512]:
    params['hidden_dim'] = hidden_dim
    for hidden_layers in [2,3,4,5,6]:
        params['hidden_layers'] = hidden_layers
        try:
            Accs, Spents, param_count = go('mlp', params, dataset)
        except:
            Accs, Spents, param_count = [np.nan, np.nan, np.nan]
        time_gcn.append((params, param_count, np.mean(Accs), np.mean(Spents)))
        to_print = str(time_gcn[-1]) + '\n'
        with open(log_prefix+'mlp', 'a') as file:
            file.write(to_print)
        for grid_size in [2,4,8,16]:
            params['grid_size'] = grid_size
            try:
                Accs, Spents, param_count = go('fastkan', params, dataset)
            except:
                Accs, Spents, param_count = [np.nan, np.nan, np.nan]
            time_fkan.append((params, param_count, np.mean(Accs), np.mean(Spents)))
            to_print = str(time_fkan[-1]) + '\n'
            with open(log_prefix+'fkan', 'a') as file:
                file.write(to_print)
            for spline_order in [1,2,4,8,16]:
                params['spline_order'] = spline_order
                try:
                    Accs, Spents, param_count = go('kan', params, dataset)
                except:
                    Accs, Spents, param_count = [np.nan, np.nan, np.nan]
                time_kan.append((params, param_count, np.mean(Accs), np.mean(Spents)))
                to_print = str(time_kan[-1]) + '\n'
                with open(log_prefix+'kan', 'a') as file:
                    file.write(to_print)

