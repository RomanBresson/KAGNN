from ekan import KAN as eKAN
import torch

from torch.nn import Sequential
from torch_geometric.nn import GINConv, GCNConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)



class GCKANLayer(torch.nn.Module):
    def __init__(self, in_feat:int, 
                 out_feat:int,
                 grid_size:int=4, 
                 spline_order:int=3):
        super(GCKANLayer, self).__init__()
        self.kan = eKAN([in_feat, out_feat], grid_size=grid_size, spline_order=spline_order)


    def forward(self, X, A_hat_normalized):
        return self.kan(A_hat_normalized @ X)




class GIKANLayer(GINConv):
    def __init__(self, in_feat:int, 
                 out_feat:int,
                 grid_size:int=4, 
                 spline_order:int=3):
        kan = eKAN([in_feat, out_feat], grid_size=grid_size, spline_order=spline_order)
        GINConv.__init__(self, kan)




class GNN_Nodes(torch.nn.Module):
    def __init__(self,  conv_type :str, 
                 num_layers:int, 
                 num_features:int, 
                 hidden_channels:int,
                 num_classes:int, 
                 skip:bool = True):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers-1):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(num_features, hidden_channels))
                else:
                    self.convs.append(GINConv(Sequential(
                    torch.nn.Linear(num_features, hidden_channels),
                    torch.nn.ReLU() , Linear( hidden_channels, hidden_channels ),
                    torch.nn.ReLU()) ))

            else:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                else:
                    self.convs.append(GINConv(Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU() , Linear( hidden_channels, hidden_channels ),
                    torch.nn.ReLU()) ))

        self.skip = skip

        if self.skip:
            if conv_type == "gcn":
                self.conv_out =  GCNConv(num_features+(num_layers-1)*hidden_channels, num_classes)
            else:
                self.conv_out = GINConv(Sequential(
                torch.nn.Linear(num_features+(num_layers-1)*hidden_channels, hidden_channels),
                torch.nn.ReLU() , torch.nn.Linear( hidden_channels, num_classes ),
                torch.nn.ReLU()) )
        else:
            if conv_type == "gcn":
                self.conv_out =  GCNConv(hidden_channels, num_classes)
            else:
                self.conv_out = GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU() , torch.nn.Linear( hidden_channels, num_classes ),
                torch.nn.ReLU()) )


    def forward(self, x: torch.tensor , edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.nn.functional.relu(x)
            l.append(x)

        if self.skip:
            x = torch.cat(l, dim=1)

        x = self.conv_out(x, edge_index)
        x = torch.nn.functional.relu(x)
        return x




class GKAN_Nodes(torch.nn.Module):
    def __init__(self, conv_type :str, 
                 num_layers:int, 
                 num_features:int, 
                 hidden_channels:int,
                 num_classes:int, 
                 skip:bool = True, 
                 grid_size:int = 4, 
                 spline_order:int = 3):
        super().__init__()

        dic = {"gcn": GCKANLayer , "gin": GIKANLayer}
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers-1):
            if i ==0:
                self.convs.append(dic[conv_type](num_features, hidden_channels, grid_size, spline_order))
            else:
                self.convs.append(dic[conv_type]( hidden_channels, hidden_channels, grid_size, spline_order))

        self.skip = skip

        if self.skip:
            self.conv_out = dic[conv_type](num_features+(num_layers-1)*hidden_channels,
                                       num_classes, grid_size, spline_order)
        else:
            self.conv_out = dic[conv_type](hidden_channels, num_classes, grid_size, spline_order)


    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv in self.convs:

            x = conv(x, edge_index)
            l.append(x)

        if self.skip:
            x = torch.cat(l, dim=1)

        x = self.conv_out(x,edge_index)
        return x