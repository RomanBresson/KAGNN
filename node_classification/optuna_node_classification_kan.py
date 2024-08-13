
# Install required packages.
import os

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Actor, WebKB
import torch_geometric as pyg
from utils import sparse_diag, set_seed,experiment_node_class
from models import GKAN_Nodes
import numpy as np
import optuna
import json 
from optuna.trial import Trial


def objective( trial: Trial, 
              data: pyg.data.Data, 
              dataset_name: str, 
              dataset: pyg.data.Dataset, 
              conv_type: str, 
              skip: bool, 
              n_epochs: int, 
              device: str) -> float:
    grid_size = trial.suggest_int('grid_size', 3, 5)
    spline_order = trial.suggest_int('spline_order', 1,4)
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64, 128])
    lr = trial.suggest_float('lr', 0.001, 0.01, log=True)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
    regularizer = trial.suggest_categorical('regularizer', [0, 5e-4])

    accuracy = train_and_evaluate_model(spline_order, hidden_channels, lr, hidden_layers, regularizer, data, dataset_name, dataset,
                               conv_type, skip, grid_size,  n_epochs, device )
    
    return accuracy



def train_and_evaluate_model(spline_order: int, 
                             hidden_channels: int, 
                             lr: float, 
                             hidden_layers: int, 
                             regularizer: float, 
                             data: pyg.data.Data, 
                             dataset_name: str, 
                             dataset: pyg.data.Dataset,
                             conv_type: str, 
                             skip: bool, 
                             grid_size: int, 
                             n_epochs: int, 
                             device: str = "cuda") -> float:
    best_val_acc_full = []
    best_test_acc_full = []
    criterion =  torch.nn.CrossEntropyLoss()

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
        best_val_acc_full = [] 
        best_test_acc_full = []
        
        model = GKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = dataset.num_features, hidden_channels= hidden_channels, 
                        num_classes = dataset.num_classes, skip = skip, grid_size=grid_size, spline_order=spline_order).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularizer)
        
        best_val_acc, best_test_acc, time_ = experiment_node_class(train_mask,  valid_mask, test_mask, model, data, optimizer, criterion, n_epochs)
        

    elif dataset_name in ['Cora', 'CiteSeer']:
        train_mask = data.train_mask
        valid_mask = data.val_mask
        test_mask = data.test_mask
        
            
        model = GKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = dataset.num_features, hidden_channels= hidden_channels, 
                        num_classes = dataset.num_classes, skip = skip, grid_size=grid_size, spline_order=spline_order).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularizer)
            
        best_val_acc, best_test_acc, time_ = experiment_node_class(train_mask,  valid_mask, test_mask, model, data, optimizer, criterion, n_epochs)
        
    else:
        best_val_acc_full = [] 
        best_test_acc_full = []
        
        for sim in range(len(data.train_mask[0])):
            model = GKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = dataset.num_features, hidden_channels= hidden_channels, 
                               num_classes = dataset.num_classes, skip = skip, grid_size=grid_size, spline_order=spline_order).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularizer)
            train_mask = data.train_mask[:,sim]
            valid_mask = data.val_mask[:,sim]
            test_mask = data.test_mask[:,sim]

            best_val_acc, best_test_acc, time_ = experiment_node_class(train_mask, valid_mask, test_mask, model, data, optimizer, criterion, n_epochs)
            best_val_acc_full.append(best_val_acc)
            best_test_acc_full.append(best_test_acc)

        best_val_acc = np.mean(best_val_acc_full)
    return best_val_acc






def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_epochs = 1000
    skip = True
    n_trials = 100
    if not os.path.exists('data'):
        os.makedirs('data')
      
    set_seed(1)
    for dataset_name in ['Cora', 'CiteSeer', 'Actor', 'Texas','Cornell','Wisconsin', 'ogbn-arxiv']:
        for conv_type in ['gin','gcn']:                     
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
        
            log =f'results/best_params_kan_{conv_type}_{dataset_name}.json'
            
            if conv_type=='gcn':
                N = data.edge_index.max().item() + 1
                data.edge_index = data.edge_index.to("cpu")

                A = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)), (N, N))
                I = torch.sparse_coo_tensor(torch.arange(N).repeat(2,1), torch.ones(N), (N, N))

                A_hat = A + I
                # can do that because D_hat is a vector here
                D_hat = torch.sparse.sum(A_hat, dim=1).to_dense()
                D_hat =  1.0 / torch.sqrt(D_hat) 
                D_hat_inv_sqrt = sparse_diag(D_hat)

                data.edge_index  = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

            data = data.to(device) 
             
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, data, dataset_name, dataset,  conv_type ,skip, n_epochs, device ), n_trials=n_trials)
            
            best_params =  study.best_params

            with open(log, 'w') as f:
                json.dump(best_params, f)

            


if __name__ == "__main__":
    main()

