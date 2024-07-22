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

dataset_num_features = dataset[0].x.shape[1]
dataset_num_classes = dataset.num_classes
train_index = splits[0]['model_selection'][0]['train']
train_dataset = dataset[train_index]
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

log_file = f'logs/time_{args.dataset}'

curr_config = 0

def time_model(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    nb_epochs = 30
    t0 = time.time()
    for _ in range(nb_epochs):
        for data in train_loader:
            optimizer.zero_grad()
            loss = F.nll_loss(model(data), data.y)
            loss.backward()
            optimizer.step()
    t1 = time.time()
    return(np.round((t1-t0)/nb_epochs, 3))

for hidden_dim in [16, 64, 256, 512]:
    for hidden_layers in [2,3,4]:
        model = GIN(args.nb_gnn_layers, dataset_num_features, hidden_dim, hidden_layers, dataset_num_classes, dropout=0.).to(device)
        t = time_model(model, train_loader)
        print(f"GIN & {hidden_dim} & {hidden_layers} & NA & NA & {count_params(model)} & {t}\\\\")

for hidden_dim in [16, 32, 64, 256]:
    for hidden_layers in [2,4]:
            for grid_size in [1, 4]:
                for spline_order in [1, 4]:
                    model = KAGIN(args.nb_gnn_layers, dataset_num_features, hidden_dim, dataset_num_classes, hidden_layers, grid_size, spline_order, dropout=0.,).to(device)
                    t = time_model(model, train_loader)
                    print(f"KAN & {hidden_dim} & {hidden_layers} & {grid_size} & {spline_order} & {count_params(model)} & {t}\\\\")

for hidden_dim in [16, 64, 256, 512]:
    for hidden_layers in [2,4]:
            for grid_size in [2, 5]:
                model = FASTKAGIN(args.nb_gnn_layers, dataset_num_features, hidden_dim, dataset_num_classes, hidden_layers, grid_size, dropout=0.).to(device)
                t = time_model(model, train_loader)
                print(f"FASTKAN & {hidden_dim} & {hidden_layers} & {grid_size-1} & NA & {count_params(model)} & {t}\\\\")