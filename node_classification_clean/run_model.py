#%%
import argparse
import torch
import torch.nn as nn
from utils import load_data, dataset_layers, run_experiment
from models import GNN_Nodes, GFASTKAN_Nodes, GKAN_Nodes

parser = argparse.ArgumentParser(description='Node_classif')
parser.add_argument('--dataset', default='Cora', help='Dataset name')
parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train')
parser.add_argument('--patience', type=int, default=20, help='Patience of early stopping')
parser.add_argument('--random_seed', type=int, default=12345, help='Random seed')
parser.add_argument('--model_type', default='gin', help='GIN/GCN')
parser.add_argument('--architecture', default='mlp', help='MLP/KAN/FASTKAN')
args = parser.parse_args()

data = load_data(args.dataset)
params = {'hidden_channels':8,
            'mp_layers': dataset_layers[args.dataset],
            'hidden_layers':2,
            'lr':0.01,
            'dropout':0,
            'spline_order':2,
            'grid_size':2,
            'model_type':args.model_type,
            'architecture':args.architecture,
            'num_features': data.num_features,
            'num_classes': data.num_classes,
            'patience': args.patience,
            'heads': args.heads
        }

#%%
device = 'cuda' if torch.cuda.is_available else 'cpu'

#%%
run_experiment(params, args.dataset)
# %%
