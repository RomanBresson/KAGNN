from ekan import KAN as eKAN,KANLinear
from fastkan import FastKAN,FastKANLayer
import torch

import torch.nn as nn
from torch_geometric.nn import GINConv, GCNConv, GATConv

def make_mlp(num_features, hidden_dim, out_dim, hidden_layers):
    if hidden_layers>=2:
        list_hidden = [nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU())]
        for _ in range(hidden_layers-2):
            list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, out_dim, nn.ReLU())))
    else:
        list_hidden = [nn.Sequential(nn.Linear(num_features, out_dim), nn.ReLU())]
    mlp = nn.Sequential(*list_hidden)
    return(mlp)

def make_kan(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order):
    sizes = [num_features] + [hidden_dim]*(hidden_layers-1) + [out_dim]
    return(eKAN(layers_hidden=sizes, grid_size=grid_size, spline_order=spline_order))

def make_fastkan(num_features, hidden_dim, out_dim, hidden_layers, grid_size):
    sizes = [num_features] + [hidden_dim]*(hidden_layers-1) + [out_dim]
    return(FastKAN(layers_hidden=sizes, num_grids=grid_size))

class KANLayer(KANLinear):
    def __init__(self, input_dim, output_dim, grid_size=4, spline_order=3):
        super(KANLayer, self).__init__(in_features=input_dim, out_features=output_dim, grid_size=grid_size, spline_order=spline_order)

class KAGCNConv(GCNConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4,
                 spline_order:int=3):
        super(KAGCNConv, self).__init__(in_feat, out_feat)
        self.lin = KANLayer(in_feat, out_feat, grid_size, spline_order)

class KAGATConv(GATConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 heads:int,
                 grid_size:int=4,
                 spline_order:int=3):
        super(KAGATConv, self).__init__(in_feat, out_feat, heads)
        self.lin = KANLayer(in_feat, out_feat*heads, grid_size, spline_order)

class GIKANLayer(GINConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4,
                 spline_order:int=3,
                 hidden_dim:int=16,
                 nb_layers:int=2):
        kan = make_kan(in_feat, hidden_dim, out_feat, nb_layers, grid_size, spline_order)
        GINConv.__init__(self, kan)

class FKANLayer(FastKANLayer):
    def __init__(self, input_dim, output_dim, num_grids=4):
        super(FKANLayer, self).__init__(input_dim=input_dim, output_dim=output_dim, num_grids=num_grids)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids

    def reset_parameters(self):
        self.__init__(self.input_dim, self.output_dim, self.num_grids)

class FASTKAGCNConv(GCNConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4):
        super(FASTKAGCNConv, self).__init__(in_channels=in_feat, out_channels=out_feat)
        self.grid_size = grid_size
        self.lin = FKANLayer(in_feat, out_feat, num_grids=grid_size)

class FASTKAGATConv(GATConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 heads:int,
                 grid_size:int=4,):
        super(FASTKAGATConv, self).__init__(in_feat, out_feat, heads)
        self.grid_size = grid_size
        self.lin = FKANLayer(in_feat, out_feat*heads, grid_size)

class GIFASTKANLayer(GINConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4,
                 hidden_dim:int=16,
                 nb_layers:int=2):
        kan = make_fastkan(in_feat, hidden_dim, out_feat, nb_layers, grid_size)
        GINConv.__init__(self, kan)

class GNN_Nodes(torch.nn.Module):
    def __init__(self,  conv_type :str,
                 mp_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True,
                 hidden_layers:int=2,
                 dropout:float=0.,
                 heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if conv_type!='gat':
            heads = 1
        torch.nn.BatchNorm1d(hidden_channels)
        for i in range(mp_layers):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(num_features, hidden_channels))
                elif conv_type == "gat":
                    self.convs.append(GATConv(num_features, hidden_channels, heads))
                elif conv_type == "gin":
                    self.convs.append(GINConv(make_mlp(num_features, hidden_channels, hidden_channels, hidden_layers)))
                else:
                    raise ValueError("unknown conv_type")
            else:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                elif conv_type == "gat":
                    self.convs.append(GATConv(hidden_channels*heads, hidden_channels, heads))
                elif conv_type == "gin":
                    self.convs.append(GINConv(make_mlp(hidden_channels, hidden_channels, hidden_channels, hidden_layers)))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        self.skip = skip
        dim_out_message_passing = num_features+(mp_layers)*hidden_channels if skip else hidden_channels
        if conv_type == "gat":
            dim_out_message_passing = num_features+(mp_layers)*hidden_channels*heads if skip else hidden_channels*heads
        self.lay_out = torch.nn.Linear(dim_out_message_passing, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        l = []
        if self.skip:
            l.append(x)
        for conv,bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.dropout(x)
            if self.skip:
                l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.lay_out(x)
        return x

class GKAN_Nodes(torch.nn.Module):
    def __init__(self,  conv_type :str,
                 mp_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True,
                 grid_size:int = 4,
                 spline_order:int = 3,
                 hidden_layers:int=2,
                 dropout:float=0.,
                 heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if conv_type!='gat':
            heads = 1
        for i in range(mp_layers):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(KAGCNConv(num_features, hidden_channels, grid_size, spline_order))
                elif conv_type == "gat":
                    self.convs.append(KAGATConv(num_features, hidden_channels, heads, grid_size, spline_order))
                elif conv_type == "gin":
                    self.convs.append(GIKANLayer(num_features, hidden_channels, grid_size, spline_order, hidden_channels, hidden_layers))
                else:
                    raise ValueError("unknown conv_type")
            else:
                if conv_type == "gcn":
                    self.convs.append(KAGCNConv(hidden_channels, hidden_channels, grid_size, spline_order))
                elif conv_type == "gat":
                    self.convs.append(KAGATConv(hidden_channels*heads, hidden_channels, heads, grid_size, spline_order))
                else:
                    self.convs.append(GIKANLayer(hidden_channels, hidden_channels, grid_size, spline_order, hidden_channels, hidden_layers))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        self.skip = skip
        dim_out_message_passing = num_features+mp_layers*hidden_channels if skip else hidden_channels
        if conv_type == "gat":
            dim_out_message_passing = num_features+mp_layers*hidden_channels*heads if skip else hidden_channels*heads
        self.lay_out = KANLinear(dim_out_message_passing, num_classes, grid_size=grid_size, spline_order=spline_order)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv,bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.dropout(x)
            l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.lay_out(x)
        return x

class GFASTKAN_Nodes(torch.nn.Module):
    def __init__(self,  conv_type :str,
                 mp_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True,
                 grid_size:int = 4,
                 hidden_layers:int=2,
                 dropout:float=0.,
                 heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if conv_type!='gat':
            heads = 1
        for i in range(mp_layers):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(FASTKAGCNConv(num_features, hidden_channels, grid_size))
                elif conv_type == "gat":
                    self.convs.append(FASTKAGATConv(num_features, hidden_channels, heads, grid_size))
                elif conv_type== "gin":
                    self.convs.append(GIFASTKANLayer(num_features, hidden_channels, grid_size, hidden_channels, hidden_layers))
                else:
                    raise ValueError("unknown conv_type")
            else:
                if conv_type == "gcn":
                    self.convs.append(FASTKAGCNConv(hidden_channels, hidden_channels, grid_size))
                elif conv_type == "gat":
                    self.convs.append(FASTKAGATConv(hidden_channels*heads, hidden_channels, heads, grid_size))
                elif conv_type== "gin":
                    self.convs.append(GIFASTKANLayer(hidden_channels, hidden_channels, grid_size, hidden_channels, hidden_layers))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        self.skip = skip
        dim_out_message_passing = num_features+(mp_layers)*hidden_channels if skip else hidden_channels
        if conv_type == "gat":
            dim_out_message_passing = num_features+(mp_layers)*hidden_channels*heads if skip else hidden_channels*heads
        self.lay_out = FastKANLayer(dim_out_message_passing, num_classes, num_grids=grid_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv,bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.dropout(x)
            l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.lay_out(x)
        return x