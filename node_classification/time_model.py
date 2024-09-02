#%%
import json
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset

from utils import *
import time

from models import GNN_Nodes, GKAN_Nodes, GFASTKAN_Nodes

#%%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
"""
dataset_name = "Cora"
dataset = Planetoid(root='data/'+dataset_name, name=dataset_name, transform=NormalizeFeatures())
"""
dataset_name = "ogbn-arxiv"
dataset = PygNodePropPredDataset(name=dataset_name, root='data/'+dataset_name)

mp_layers = dataset_layers[dataset_name]

#%%
log_file = f'logs/time_{dataset_name}'

curr_config = 0

def time_model(model, data, mask):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    nb_epochs = 20
    t0 = time.time()
    for _ in range(nb_epochs):
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        out = torch.softmax(out, dim=1)
        loss = criterion(out[mask], data.y[mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()
    t1 = time.time()
    return(np.round((t1-t0)/nb_epochs, 3))

def count_params(model):
    s = 0
    for k in model.parameters():
        s+= torch.prod(torch.tensor(k.shape))
    return s

#%%
time.sleep(1)
for conv_type in ['gcn', 'gin']:
    data = dataset[0]
    if dataset_name == 'ogbn-arxiv':
        split_idx = dataset.get_idx_split()
        data = data.to(device)
        data.y = data.y.squeeze()
        train_mask  = torch.zeros(data.num_nodes, dtype=torch.bool).squeeze().to(device)
        train_mask[split_idx['train']] = True
    else:
        data = data.to(device)
        data.y = data.y.squeeze()
        train_mask = data.train_mask
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
        data.edge_index  = (D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt).to(device)

    HIDDENS = [2,4] if conv_type == 'gin' else [0]

    for hidden_channels in [4, 64, 256, 1024]:
        for hidden_layers in HIDDENS:
            model = GNN_Nodes(conv_type=conv_type, mp_layers=mp_layers, num_features=dataset.num_features, hidden_channels=hidden_channels,
                num_classes=dataset.num_classes, skip=True, hidden_layers=hidden_layers, dropout=0).to(device)
            t = time_model(model, data, train_mask)
            print(f"{conv_type} & {hidden_channels} & {hidden_layers if hidden_layers!=0 else 'NA'} & NA & NA & {count_params(model)} & {t}\\\\")

    for hidden_channels in [16, 32, 64, 128]:
        for hidden_layers in HIDDENS:
                for grid_size in [1, 8]:
                    for spline_order in [1, 4]:
                        model = GKAN_Nodes(conv_type=conv_type, mp_layers=mp_layers, num_features=dataset.num_features, hidden_channels=hidden_channels,
                          num_classes=dataset.num_classes, skip=True, hidden_layers=hidden_layers, spline_order=spline_order, dropout=0).to(device)
                        t = time_model(model, data, train_mask)
                        print(f"BSKAN {conv_type} & {hidden_channels} & {hidden_layers if hidden_layers!=0 else 'NA'} & {grid_size} & {spline_order} & {count_params(model)} & {t}\\\\")

    for hidden_channels in [16, 64, 256, 512]:
        for hidden_layers in HIDDENS:
                for grid_size in [2, 9]:
                    model = GFASTKAN_Nodes(conv_type=conv_type, mp_layers=mp_layers, num_features=dataset.num_features, hidden_channels=hidden_channels,
                          num_classes=dataset.num_classes, skip=True, hidden_layers=hidden_layers, dropout=0).to(device)
                    t = time_model(model, data, train_mask)
                    print(f"RBFKAN {conv_type} & {hidden_channels} & {hidden_layers if hidden_layers!=0 else 'NA'} & {grid_size-1} & NA & {count_params(model)} & {t}\\\\")
# %%
