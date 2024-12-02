from ekan import KAN as eKAN,KANLinear
from fastkan import FastKAN,FastKANLayer
import torch

import torch.nn as nn
from torch_geometric.nn import GINConv, GCNConv

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
                 dropout:float=0.):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(hidden_channels)
        for i in range(mp_layers-1):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(num_features, hidden_channels))
                else:
                    self.convs.append(GINConv(make_mlp(num_features, hidden_channels, hidden_channels, hidden_layers)))
            else:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                else:
                    self.convs.append(GINConv(make_mlp(hidden_channels, hidden_channels, hidden_channels, hidden_layers)))
        self.skip = skip
        dim_out_message_passing = num_features+(mp_layers-1)*hidden_channels if skip else hidden_channels
        if conv_type == "gcn":
            self.conv_out = GCNConv(dim_out_message_passing, num_classes)
        else:
            self.conv_out = GINConv(make_mlp(dim_out_message_passing, hidden_channels, num_classes, hidden_layers))

    def forward(self, x: torch.tensor , edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.nn.functional.relu(x)
            x = self.bn(x)
            x = self.dropout(x)
            l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.conv_out(x, edge_index)
        x = torch.nn.functional.relu(x)
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
                 dropout:float=0.):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bn = torch.nn.BatchNorm1d(hidden_channels)
        for i in range(mp_layers-1):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(KAGCNConv(num_features, hidden_channels, grid_size, spline_order))
                else:
                    self.convs.append(GIKANLayer(num_features, hidden_channels, grid_size, spline_order, hidden_channels, hidden_layers))
            else:
                if conv_type == "gcn":
                    self.convs.append(KAGCNConv(hidden_channels, hidden_channels, grid_size, spline_order))
                else:
                    self.convs.append(GIKANLayer(hidden_channels, hidden_channels, grid_size, spline_order, hidden_channels, hidden_layers))
        self.skip = skip
        dim_out_message_passing = num_features+(mp_layers-1)*hidden_channels if skip else hidden_channels
        if conv_type == "gcn":
            self.conv_out = KAGCNConv(dim_out_message_passing, num_classes, grid_size, spline_order)
        else:
            self.conv_out = GINConv(make_kan(dim_out_message_passing, hidden_channels, num_classes, hidden_layers, grid_size, spline_order))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.tensor , edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.bn(x)
            x = self.dropout(x)
            l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.conv_out(x, edge_index)
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
                 dropout:float=0.):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bn = torch.nn.BatchNorm1d(hidden_channels)
        for i in range(mp_layers-1):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(FASTKAGCNConv(num_features, hidden_channels, grid_size))
                else:
                    self.convs.append(GIFASTKANLayer(num_features, hidden_channels, grid_size, hidden_channels, hidden_layers))
            else:
                if conv_type == "gcn":
                    self.convs.append(FASTKAGCNConv(hidden_channels, hidden_channels, grid_size))
                else:
                    self.convs.append(GIFASTKANLayer(hidden_channels, hidden_channels, grid_size, hidden_channels, hidden_layers))
        self.skip = skip
        dim_out_message_passing = num_features+(mp_layers-1)*hidden_channels if skip else hidden_channels
        if conv_type == "gcn":
            self.conv_out = FASTKAGCNConv(dim_out_message_passing, num_classes, grid_size)
        else:
            self.conv_out = GINConv(make_fastkan(dim_out_message_passing, hidden_channels, num_classes, hidden_layers, grid_size))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.bn(x)
            x = self.dropout(x)
            l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.conv_out(x, edge_index)
        return x