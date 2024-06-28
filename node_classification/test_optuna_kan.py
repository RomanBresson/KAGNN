# Install required packages.
import os

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor
from torch_geometric.datasets import Actor, WebKB
import torch_geometric as pyg
from utils import sparse_diag, experiment_node_class, set_seed
from models import GKAN_Nodes
import numpy as np
import numpy as np
import json 




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
                             device: str = "cuda", 
                             number_of_runs = 10) -> float:

    time_l = []
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

        best_test_acc_full = []
        
        for run in range(number_of_runs):
            print(f' in run {run}')
            torch.manual_seed(run)
            np.random.seed(run)
            model = GKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = dataset.num_features, hidden_channels= hidden_channels, 
                        num_classes = dataset.num_classes, skip = skip, grid_size=grid_size, spline_order=spline_order).to(device)
        
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularizer)
            
            best_val_acc, best_test_acc, time_ = experiment_node_class(train_mask,  valid_mask, test_mask, model, data, optimizer, criterion, n_epochs)
            time_l.append(time_)
            best_test_acc_full.append(best_test_acc)

        best_test_acc = np.mean(best_test_acc_full)
        best_test_sd = np.std(best_test_acc_full)

    elif dataset_name in ['Cora', 'CiteSeer']:
        train_mask = data.train_mask
        valid_mask = data.val_mask
        test_mask = data.test_mask
        
        best_test_acc_full = []

        for run in range(number_of_runs):
            print(f' in run {run}')
            torch.manual_seed(run)
            np.random.seed(run)
            model = GKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = dataset.num_features, hidden_channels= hidden_channels, 
                            num_classes = dataset.num_classes, skip = skip, grid_size=grid_size, spline_order=spline_order).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularizer)
            
            best_val_acc, best_test_acc, time_ = experiment_node_class(train_mask,  valid_mask, test_mask, model, data, optimizer, criterion, n_epochs)
            time_l.append(time_)
            best_test_acc_full.append(best_test_acc)
        
        best_test_acc = np.mean(best_test_acc_full)
        best_test_sd = np.std(best_test_acc_full)


    else:
        best_test_acc_full = []
        
        for sim in range(len(data.train_mask[0])):
            model = GKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = dataset.num_features, hidden_channels= hidden_channels, 
                               num_classes = dataset.num_classes, skip = skip, grid_size=grid_size, spline_order=spline_order).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularizer)
            train_mask = data.train_mask[:,sim]
            valid_mask = data.val_mask[:,sim]
            test_mask = data.test_mask[:,sim]

            best_val_acc, best_test_acc, time_ = experiment_node_class(train_mask, valid_mask, test_mask, model, data, optimizer, criterion, n_epochs)
            time_l.append(time_)
            best_test_acc_full.append(best_test_acc)

        best_test_acc = np.mean(best_test_acc_full)
        best_test_sd = np.std(best_test_acc_full)
    
    return best_test_acc, best_test_sd, np.mean(time_l)





def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_epochs = 100
    skip = True

    log = open("results/test_log_kan_time.txt", "a")
    set_seed(1)
    
    for dataset_name in [ 'Cora', 'CiteSeer', 'Wisconsin','Cornell','Texas', 'Actor', 'ogbn-arxiv']: 
        for conv_type in ['gcn','gin']:                     
            print(dataset_name+ " "+conv_type)

            # free the torch memory
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
        
            with open(f'results/best_params_kan_{conv_type}_{dataset_name}.json', 'r') as f:
                best_params = json.load(f)

            grid_size = best_params['grid_size']
            spline_order = best_params['spline_order']
            hidden_channels = best_params['hidden_channels']
            lr = best_params['lr']
            hidden_layers = best_params['hidden_layers']
            regularizer = best_params['regularizer']    

            if conv_type=='gcn':
                N = data.edge_index.max().item() + 1
                data.edge_index = data.edge_index.to("cpu")

                A = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], 
                                value=torch.ones(data.edge_index.size(1)), 
                                sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()

                I = SparseTensor(row=torch.arange(N), col=torch.arange(N), 
                                value=torch.ones(N), 
                                sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()

                A_hat = A + I
                D_hat = torch.sparse.sum(A_hat, dim=1).to_dense()
                D_hat =  1.0 / torch.sqrt(D_hat) 
                D_hat_inv_sqrt = sparse_diag(D_hat)

                data.edge_index  = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

            data = data.to(device) 
             
            test_mean, test_sd, time_  = train_and_evaluate_model(spline_order, hidden_channels, lr, hidden_layers, regularizer, data, dataset_name, dataset, conv_type, skip, grid_size,  n_epochs, device )
            
            print(f"Test mean acc: {test_mean:.4f}, Test sd: {test_sd:.4f}\n, Time: {time_:.4f}")    
            log.write(f"{dataset_name} {conv_type} accuracy mean {test_mean:.4f} , sd {test_sd:.4f}, Time: {time_:.4f}\n")
    


if __name__ == "__main__":
    main()
