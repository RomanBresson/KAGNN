import torch
import torch.nn as nn
import torch.nn.functional as F
from ekan import *

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool

def make_mlp(num_features, hidden_dim, out_dim, hidden_layers):
    if hidden_layers>=2:
        list_hidden = [nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU())]
        for _ in range(hidden_layers-2):
            list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        list_hidden.append(nn.Linear(hidden_dim, out_dim))
    else:
        return nn.Sequential(nn.Linear(num_features, out_dim), nn.ReLU())
    MLP = nn.Sequential(*list_hidden)
    return(MLP)

class GIN(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, hidden_layers, n_targets, dropout, embedding_layer=False):
        super(GIN, self).__init__()
        self.n_layers = gnn_layers
        self.embedding_layer = embedding_layer
        lst = list()
        if embedding_layer:
            self.node_emb = nn.Embedding(num_features, 100)
            lst.append(GINConv(make_mlp(100, hidden_dim, hidden_dim, hidden_layers)))
        else:
            lst.append(GINConv(make_mlp(num_features, hidden_dim, hidden_dim, hidden_layers)))
        for i in range(gnn_layers-1):
            lst.append(GINConv(make_mlp(hidden_dim, hidden_dim, hidden_dim, hidden_layers)))
        self.conv = nn.ModuleList(lst)

        self.mlp = make_mlp(hidden_dim, 64, n_targets, 2)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.embedding_layer:
            x = self.node_emb(x).squeeze()
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.dropout(x)

        x = global_add_pool(x, data.batch)
        x = self.mlp(x)
        return x

def make_kan(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order):
    sizes = [num_features] + [hidden_dim]*(hidden_layers-2) + [out_dim]
    return(KAN(layers_hidden=sizes, grid_size=grid_size, spline_order=spline_order))
    
class GKAN(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, hidden_layers, grid_size, spline_order, n_targets, dropout, embedding_layer=False):
        super(GKAN, self).__init__()
        self.n_layers = gnn_layers
        self.embedding_layer = embedding_layer
        lst = list()
        if embedding_layer:
            self.node_emb = nn.Embedding(num_features, 100)
            lst.append(GINConv(make_kan(100, hidden_dim, hidden_dim, hidden_layers, grid_size, spline_order)))
        else:
            lst.append(GINConv(make_kan(num_features, hidden_dim, hidden_dim, hidden_layers, grid_size, spline_order)))
        for i in range(gnn_layers-1):
            lst.append(GINConv(make_kan(hidden_dim, hidden_dim, hidden_dim, hidden_layers, grid_size, spline_order)))
        self.conv = nn.ModuleList(lst)

        self.kan = make_kan(hidden_dim, 64, n_targets, 2, grid_size, spline_order)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.embedding_layer:
            x = self.node_emb(x).squeeze()
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.dropout(x)

        x = global_add_pool(x, data.batch)
        x = self.kan(x)
        return x
