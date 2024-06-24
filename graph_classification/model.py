import torch
import torch.nn as nn
import torch.nn.functional as F
from ekan import *

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool

def make_mlp(num_features, hidden_dim, out_dim, hidden_layers, batch_norm=True):
    if hidden_layers>=2:
        if batch_norm:
            list_hidden = [nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim))]
        else:
            list_hidden = [nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU())]
        for _ in range(hidden_layers-2):
            if batch_norm:
                list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)))
            else:
                list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, out_dim, nn.ReLU())))
    else:
        list_hidden = [nn.Sequential(nn.Linear(num_features, out_dim), nn.ReLU())]
    MLP = nn.Sequential(*list_hidden)
    return(MLP)

class GIN(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, hidden_layers, num_classes, dropout):
        super(GIN, self).__init__()
        self.n_layers = gnn_layers
        lst = list()
        lst.append(GINConv(make_mlp(num_features, hidden_dim, hidden_dim, hidden_layers, batch_norm=True)))
        for i in range(gnn_layers-1):
            lst.append(GINConv(make_mlp(hidden_dim, hidden_dim, hidden_dim, hidden_layers, batch_norm=True)))
        self.conv = nn.ModuleList(lst)
        
        self.mlp = make_mlp(hidden_dim, hidden_dim, num_classes, hidden_layers, batch_norm=False)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.dropout(x)

        x = global_add_pool(x, data.batch)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)

def make_kan(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order):
    sizes = [num_features] + [hidden_dim]*(hidden_layers-2) + [out_dim]
    return(KAN(layers_hidden=sizes, grid_size=grid_size, spline_order=spline_order))
    
class GKAN(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, num_classes, hidden_layers, grid_size, spline_order, dropout):
        super(GKAN, self).__init__()
        self.n_layers = gnn_layers
        lst = list()
        lst.append(GINConv(make_kan(num_features, hidden_dim, hidden_dim, hidden_layers, grid_size, spline_order)))
        for i in range(gnn_layers-1):
            lst.append(GINConv(make_kan(hidden_dim, hidden_dim, hidden_dim, hidden_layers, grid_size, spline_order)))
        self.conv = nn.ModuleList(lst)

        lst = list()
        for i in range(gnn_layers):
            lst.append(nn.BatchNorm1d(hidden_dim))
        self.bn = nn.ModuleList(lst)

        self.kan = make_kan(hidden_dim, hidden_dim, num_classes, hidden_layers, grid_size, spline_order)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.bn[i](x)
            x = self.dropout(x)

        x = global_add_pool(x, data.batch)
        x = self.kan(x)
        return F.log_softmax(x, dim=1)
