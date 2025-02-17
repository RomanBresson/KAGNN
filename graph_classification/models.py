import torch.nn as nn
import torch.nn.functional as F
from ekan import KAN, KANLinear
from fastkan import FastKAN, FastKANLayer

from torch_geometric.nn import GINConv, GCNConv, GATConv
from torch_geometric.nn import global_add_pool, global_mean_pool

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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)

class GCN(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, num_classes, dropout):
        super(GCN, self).__init__()
        self.n_layers = gnn_layers
        lst = list()
        lst.append(GCNConv(num_features, hidden_dim))
        for _ in range(gnn_layers-1):
            lst.append(GCNConv(hidden_dim, hidden_dim))
        self.conv = nn.ModuleList(lst)
        self.readout = make_mlp(hidden_dim, hidden_dim, num_classes, 1, batch_norm=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = F.silu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = self.readout(x)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, num_classes, dropout, heads):
        super(GAT, self).__init__()
        self.n_layers = gnn_layers
        lst = list()
        lst.append(GATConv(num_features, hidden_dim, heads))
        for _ in range(gnn_layers-1):
            lst.append(GATConv(hidden_dim*heads, hidden_dim, heads))
        self.conv = nn.ModuleList(lst)
        self.readout = make_mlp(hidden_dim*heads, hidden_dim, num_classes, 1, batch_norm=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = F.silu(x)
            x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        x = self.readout(x)
        return F.log_softmax(x, dim=1)

def make_kan(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order):
    sizes = [num_features] + [hidden_dim]*(hidden_layers-1) + [out_dim]
    return(KAN(layers_hidden=sizes, grid_size=grid_size, spline_order=spline_order))
    
class KAGIN(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, num_classes, hidden_layers, grid_size, spline_order, dropout):
        super(KAGIN, self).__init__()
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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.bn[i](x)
            x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        x = self.kan(x)
        return F.log_softmax(x, dim=1)

def make_fastkan(num_features, hidden_dim, out_dim, hidden_layers, grid_size):
    sizes = [num_features] + [hidden_dim]*(hidden_layers-1) + [out_dim]
    return(FastKAN(layers_hidden=sizes, num_grids=grid_size))

class FASTKAGIN(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, num_classes, hidden_layers, grid_size, dropout):
        super(FASTKAGIN, self).__init__()
        self.n_layers = gnn_layers
        lst = list()
        lst.append(GINConv(make_fastkan(num_features, hidden_dim, hidden_dim, hidden_layers, grid_size)))
        for i in range(gnn_layers-1):
            lst.append(GINConv(make_fastkan(hidden_dim, hidden_dim, hidden_dim, hidden_layers, grid_size)))
        self.conv = nn.ModuleList(lst)

        lst = list()
        for i in range(gnn_layers):
            lst.append(nn.BatchNorm1d(hidden_dim))
        self.bn = nn.ModuleList(lst)

        self.kan = make_fastkan(hidden_dim, hidden_dim, num_classes, hidden_layers, grid_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.bn[i](x)
            x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        x = self.kan(x)
        return F.log_softmax(x, dim=1)

class KANLayer(KANLinear):
    def __init__(self, input_dim, output_dim, grid_size=4, spline_order=3):
        super(KANLayer, self).__init__(in_features=input_dim, out_features=output_dim, grid_size=grid_size, spline_order=spline_order)

class KAGCN_Layer(GCNConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4,
                 spline_order:int=3):
        super(KAGCN_Layer, self).__init__(in_feat, out_feat)
        self.lin = KANLayer(in_feat, out_feat, grid_size, spline_order)

class KAGAT_Layer(GATConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 heads:int,
                 grid_size:int=4,
                 spline_order:int=3):
        super(KAGAT_Layer, self).__init__(in_feat, out_feat, heads)
        self.lin = KANLayer(in_feat, out_feat*heads, grid_size, spline_order)

class KAGCN(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, num_classes, grid_size, spline_order, dropout):
        super(KAGCN, self).__init__()
        self.n_layers = gnn_layers
        lst = list()
        lst.append(KAGCN_Layer(num_features, hidden_dim, grid_size, spline_order))
        for _ in range(gnn_layers-1):
            lst.append(KAGCN_Layer(hidden_dim, hidden_dim, grid_size, spline_order))
        self.conv = nn.ModuleList(lst)
        self.readout = make_kan(hidden_dim, hidden_dim, num_classes, 1, grid_size, spline_order)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = F.silu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = self.readout(x)
        return F.log_softmax(x, dim=1)

class KAGAT(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, num_classes, grid_size, spline_order, dropout, heads):
        super(KAGAT, self).__init__()
        self.n_layers = gnn_layers
        lst = list()
        lst.append(KAGAT_Layer(num_features, hidden_dim, heads, grid_size, spline_order))
        for _ in range(gnn_layers-1):
            lst.append(KAGAT_Layer(hidden_dim*heads, hidden_dim, heads, grid_size, spline_order))
        self.conv = nn.ModuleList(lst)
        self.readout = make_kan(hidden_dim*heads, hidden_dim, num_classes, 1, grid_size, spline_order)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = F.silu(x)
            x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        x = self.readout(x)
        return F.log_softmax(x, dim=1)

class FKANLayer(FastKANLayer):
    def __init__(self, input_dim, output_dim, num_grids=4):
        super(FKANLayer, self).__init__(input_dim=input_dim, output_dim=output_dim, num_grids=num_grids)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids
    
    def reset_parameters(self):
        self.__init__(self.input_dim, self.output_dim, self.num_grids)

class FASTKAGCN_Layer(GCNConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4):
        super(FASTKAGCN_Layer, self).__init__(in_channels=in_feat, out_channels=out_feat)
        self.grid_size = grid_size
        self.lin = FKANLayer(in_feat, out_feat, num_grids=grid_size)

class FASTKAGAT_Layer(GATConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 heads:int,
                 grid_size:int=4,):
        super(FASTKAGAT_Layer, self).__init__(in_feat, out_feat, heads)
        self.grid_size = grid_size
        self.lin = FKANLayer(in_feat, out_feat*heads, grid_size)

class FASTKAGCN(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, num_classes, grid_size, dropout):
        super(FASTKAGCN, self).__init__()
        self.n_layers = gnn_layers
        lst = list()
        lst.append(FASTKAGCN_Layer(num_features, hidden_dim, grid_size))
        for _ in range(gnn_layers-1):
            lst.append(FASTKAGCN_Layer(hidden_dim, hidden_dim, grid_size))
        self.conv = nn.ModuleList(lst)
        self.readout = make_fastkan(hidden_dim, hidden_dim, num_classes, 1, grid_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = F.silu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = self.readout(x)
        return F.log_softmax(x, dim=1)

class FASTKAGAT(nn.Module):
    def __init__(self, gnn_layers, num_features, hidden_dim, num_classes, grid_size, dropout, heads):
        super(FASTKAGAT, self).__init__()
        self.n_layers = gnn_layers
        self.heads = heads
        lst = list()
        lst.append(FASTKAGAT_Layer(num_features, hidden_dim, heads=heads, grid_size=grid_size))
        for _ in range(gnn_layers-1):
            lst.append(FASTKAGAT_Layer(hidden_dim*heads, hidden_dim, heads=heads, grid_size=grid_size))
        self.conv = nn.ModuleList(lst)
        self.readout = make_fastkan(hidden_dim*heads, hidden_dim, num_classes, 1, grid_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = F.silu(x)
            x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        x = self.readout(x)
        return F.log_softmax(x, dim=1)