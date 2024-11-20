# Install required packages.
import os
import json
import optuna
import torch
import torch_geometric as pyg
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Actor, WebKB
from ogb.nodeproppred import PygNodePropPredDataset
from utils import set_seed, experiment_node_class, sparse_diag, dataset_layers
from optuna.trial import Trial
from models import GFASTKAN_Nodes

def objective( trial: Trial,
              data: pyg.data.Data,
              dataset_name: str,
              dataset: pyg.data.Dataset,
              conv_type: str,
              skip: bool,
              n_epochs: int,
              device: str) -> float:
    grid_size = trial.suggest_int('grid_size', 2, 16)
    if conv_type=='gin':
        hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
    else:
        hidden_layers = trial.suggest_int('hidden_layers', 0, 0)
    hidden_channels = trial.suggest_int('hidden_channels', 2, 64)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0, 0.9)
    val_losses = []
    for _ in range(5):
        val_loss, _, _ = train_and_evaluate_model(hidden_channels, lr, hidden_layers, dropout, data, dataset_name, dataset,
                                conv_type, skip, grid_size, n_epochs, device)
        val_losses.append(val_loss)
    return torch.tensor(val_losses).mean()

def train_and_evaluate_model(hidden_channels: int,
                             lr: float,
                             hidden_layers: int,
                             dropout: float,
                             data: pyg.data.Data,
                             dataset_name: str,
                             dataset: pyg.data.Dataset,
                             conv_type: str,
                             skip: bool,
                             grid_size: int,
                             n_epochs: int,
                             device: str = "cuda") -> float:
    best_val_loss_full = []
    best_test_acc_full = []
    criterion =  torch.nn.CrossEntropyLoss()
    mp_layers = dataset_layers[dataset_name]
    if dataset_name == 'ogbn-arxiv':
        split_idx = dataset.get_idx_split()
        data = data.to(device)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool).squeeze().to(device)
        valid_mask  = torch.zeros(data.num_nodes, dtype=torch.bool).squeeze().to(device)
        train_mask  = torch.zeros(data.num_nodes, dtype=torch.bool).squeeze().to(device)
        data.y = data.y.squeeze()
        train_mask[split_idx['train']] = True
        valid_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True
        model = GFASTKAN_Nodes(conv_type=conv_type, mp_layers=mp_layers, num_features=dataset.num_features, hidden_channels= hidden_channels,
                num_classes=dataset.num_classes, skip=skip, grid_size=grid_size, hidden_layers=hidden_layers, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_loss, best_test_acc, time_ = experiment_node_class(train_mask,  valid_mask, test_mask, model, data, optimizer, criterion, n_epochs)
    elif dataset_name in ['Cora', 'CiteSeer']:
        train_mask = data.train_mask
        valid_mask = data.val_mask
        test_mask = data.test_mask
        model = GFASTKAN_Nodes(conv_type=conv_type, mp_layers=mp_layers, num_features=dataset.num_features, hidden_channels= hidden_channels,
                num_classes=dataset.num_classes, skip=skip, grid_size=grid_size, hidden_layers=hidden_layers, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_loss, best_test_acc, time_ = experiment_node_class(train_mask,  valid_mask, test_mask, model, data, optimizer, criterion, n_epochs)
    else:
        best_val_loss_full = []
        best_test_acc_full = []
        for sim in range(len(data.train_mask[0])):
            model = GFASTKAN_Nodes(conv_type=conv_type, mp_layers=mp_layers, num_features=dataset.num_features, hidden_channels= hidden_channels,
                    num_classes=dataset.num_classes, skip=skip, grid_size=grid_size, hidden_layers=hidden_layers, dropout=dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            train_mask = data.train_mask[:,sim]
            valid_mask = data.val_mask[:,sim]
            test_mask = data.test_mask[:,sim]
            best_val_loss, best_test_acc, time_ = experiment_node_class(train_mask, valid_mask, test_mask, model, data, optimizer, criterion, n_epochs)
            best_val_loss_full.append(best_val_loss)
            best_test_acc_full.append(best_test_acc)
        best_test_acc = torch.tensor(best_test_acc_full).mean().item()
        best_val_loss = torch.tensor(best_val_loss_full).mean().item()
    return best_val_loss, best_test_acc, time_

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = 500
    skip = True
    n_trials = 100
    if not os.path.exists('data'):
        os.makedirs('data')
    set_seed(1)
    for dataset_name in ['Cora', 'CiteSeer', 'Actor', 'Texas', 'Cornell', 'Wisconsin', 'ogbn-arxiv']:
        for conv_type in ['gcn','gin']:
            print(dataset_name+ " "+conv_type)
            torch.cuda.empty_cache()
            if dataset_name == 'ogbn-arxiv':
                dataset = PygNodePropPredDataset(name=dataset_name, root='data/'+dataset_name)
            elif dataset_name in ['Cora', 'CiteSeer']:
                dataset = Planetoid(root='data/'+dataset_name, name=dataset_name, transform=NormalizeFeatures())
            elif dataset_name in ['Actor']:
                dataset = Actor(root='data/'+dataset_name)
                dataset.transform = NormalizeFeatures()
            else:
                dataset = WebKB(root='data/'+dataset_name, name=dataset_name, transform=NormalizeFeatures())
            data = dataset[0]
            log =f'results/best_params_fastkan_{conv_type}_{dataset_name}.json'
            data = data.to(device)
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, data, dataset_name, dataset, conv_type, skip, n_epochs, device), n_trials=n_trials)
            best_params =  study.best_params
            with open(log, 'w') as f:
                json.dump(best_params, f)

if __name__ == "__main__":
    main()