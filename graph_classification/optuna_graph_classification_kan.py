import random
import numpy as np
import argparse
import torch
from graph_classification_utils import parameters_finder, EarlyStopper, val, test, train, count_params, layers_per_dataset
from models import KAGIN, KAGCN, KAGAT

# Argument parser
parser = argparse.ArgumentParser(description='KAGIN')
parser.add_argument('--dataset', default='MUTAG', help='Dataset name')
parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train')
parser.add_argument('--patience', type=int, default=20, help='Patience of early stopping')
parser.add_argument('--random_seed', type=int, default=12345, help='Random seed')
parser.add_argument('--model_type', default='GIN', help='GIN/GCN/GAT')
parser.add_argument('--heads', type=int, default=4, help='GAT heads')
args = parser.parse_args()

#Uncomment for machine-wise reproducibility, using "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
#torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True) 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

def train_model_with_parameters(params, train_loader, val_loader, test_loader=None):
    lr, hidden_layers, hidden_dim, dropout, grid_size, spline_order = params['lr'], params['hidden_layers'], params['hidden_dim'], params['dropout'], params['grid_size'], params['spline_order']
    if args.model_type == 'GIN':
        model = KAGIN(layers_per_dataset[args.dataset], train_loader.dataset.num_features, hidden_dim, train_loader.dataset.num_classes, hidden_layers, grid_size, spline_order, dropout).to(device)
    elif args.model_type == 'GCN':
        model = KAGCN(layers_per_dataset[args.dataset], train_loader.dataset.num_features, hidden_dim, train_loader.dataset.num_classes, grid_size, spline_order, dropout).to(device)
    elif args.model_type == 'GAT':
        model = KAGAT(layers_per_dataset[args.dataset], train_loader.dataset.num_features, hidden_dim, train_loader.dataset.num_classes, grid_size, spline_order, dropout, args.heads).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    early_stopper = EarlyStopper(patience=args.patience)
    for epoch in range(1, args.epochs+1):
        train(model, train_loader, optimizer, device)
        val_loss = val(model, val_loader, device)
        if best_val_loss >= val_loss:
            best_val_loss = val_loss 
            if test_loader is not None:
                test_acc = test(model, test_loader, device)
        if early_stopper.early_stop(val_loss):
            print(f"Stopped at epoch {epoch}")
            break
    if test_loader is None:
        return best_val_loss
    else:
        torch.save(model, 'kan')
        return test_acc, count_params(model)

def objective(trial, train_loader, val_loader):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
    hidden_dim = trial.suggest_int('hidden_dim', 2, 64)
    grid_size = trial.suggest_int('grid_size', 2, 16)
    spline_order = trial.suggest_int('spline_order', 1, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.9)
    params = {'lr': lr, 'hidden_layers':hidden_layers, 'grid_size':grid_size, 'dropout':dropout, 'spline_order':spline_order, 'hidden_dim':hidden_dim, 'model_type':args.model_type}
    best_val_loss = train_model_with_parameters(params, train_loader, val_loader)
    return best_val_loss

log_file = f'logs/KAN_{args.dataset}_{args.model_type}'

parameters_finder(train_model_with_parameters, objective, log_file, args)
