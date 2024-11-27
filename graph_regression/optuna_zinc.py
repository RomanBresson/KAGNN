import os
import random
import argparse
import optuna
import torch
import numpy as np

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC

from utils import EarlyStopper
from models import GIN, GCN, KAGIN, KAGCN, FASTKAGIN, FASTKAGCN

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=256, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
parser.add_argument('--patience', type=int, default=20, help='Patience of early stopping')
parser.add_argument('--random_seed', type=int, default=12345, help='Random seed')
parser.add_argument('--gnn-type', default='GIN', choices=['GIN', 'GCN'], help='GNN model')
parser.add_argument('--model-type', default='MLP', choices=['MLP', 'KAN', 'FASTKAN'], help='Update model')
parser.add_argument('--num-gnn-layers', type=int, default=4, help='Number of message passing layers')
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
    if args.model_type == 'MLP' and args.gnn_type == 'GIN':
        model = GIN(1, 1, args.num_gnn_layers, params['hidden_dim'], params['hidden_layers'], 1, params['dropout'], True).to(device)
    elif args.model_type == 'MLP' and args.gnn_type == 'GCN':
        model = GCN(1, args.num_gnn_layers, params['hidden_dim'], 1, params['dropout'], True).to(device)
    elif args.model_type == 'KAN' and args.gnn_type == 'GIN':
        model = KAGIN(1, 1, args.num_gnn_layers, params['hidden_dim'], params['hidden_layers'], params['grid_size'], params['spline_order'], 1, params['dropout'], True).to(device)
    elif args.model_type == 'KAN' and args.gnn_type == 'GCN':
        model = KAGCN(1, args.num_gnn_layers, params['hidden_dim'], params['grid_size'], params['spline_order'], 1, params['dropout'], True).to(device)
    elif args.model_type == 'FASTKAN' and args.gnn_type == 'GIN':
        model = FASTKAGIN(1, 1, args.num_gnn_layers, params['hidden_dim'], params['hidden_layers'], params['grid_size'], 1, params['dropout'], True).to(device)
    elif args.model_type == 'FASTKAN' and args.gnn_type == 'GCN':
        model = FASTKAGCN(1, args.num_gnn_layers, params['hidden_dim'], params['grid_size'], 1, params['dropout'], True).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    loss_function = torch.nn.L1Loss()
    best_val_loss = float('inf')
    early_stopper = EarlyStopper(patience=args.patience)
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(data).squeeze(), data.y)
            loss.backward()
            train_loss += loss.item() * data.num_graphs
            optimizer.step()
        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        for data in val_loader:
            data = data.to(device)
            val_loss += loss_function(model(data).squeeze(), data.y).item() * data.num_graphs
        val_loss = val_loss / len(val_loader.dataset)

        if best_val_loss >= val_loss:
            best_val_loss = val_loss
            if test_loader is not None:
                test_loss = 0
                for data in test_loader:
                    data = data.to(device)
                    test_loss += loss_function(model(data).squeeze(), data.y).item() * data.num_graphs
                test_loss = test_loss / len(test_loader.dataset)

        if early_stopper.early_stop(val_loss):
            print(f"Stopped at epoch {epoch}")
            break
    if test_loader is None:
        return best_val_loss
    else:
        #torch.save(model, 'mlp')
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return test_loss, total_params

def objective_function(trial, train_loader, val_loader):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
    if args.model_type == 'MLP':
        hidden_dim = trial.suggest_int('hidden_dim', 2, 512)
    else:
        hidden_dim = trial.suggest_int('hidden_dim', 2, 64)
    dropout = trial.suggest_float('dropout', 0.0, 0.9)
    params = {'lr': lr, 'hidden_layers':hidden_layers, 'dropout':dropout, 'hidden_dim':hidden_dim, 'model_type':args.model_type}
    if args.model_type == 'KAN':
        grid_size = trial.suggest_int('grid_size', 2, 32)
        spline_order = trial.suggest_int('spline_order', 1, 4)
        params['grid_size'] = grid_size
        params['spline_order'] = spline_order
    elif args.model_type == 'FASTKAN':
        grid_size = trial.suggest_int('grid_size', 2, 32)
        params['grid_size'] = grid_size

    best_val_loss = train_model_with_parameters(params, train_loader, val_loader)
    return best_val_loss

def detailed_objective(trial, train_loader, val_loader, test_loader):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
    if args.model_type == 'MLP':
        hidden_dim = trial.suggest_int('hidden_dim', 2, 512)
    else:
        hidden_dim = trial.suggest_int('hidden_dim', 2, 64)
    dropout = trial.suggest_float('dropout', 0.0, 0.9)
    params = {'lr': lr, 'hidden_layers':hidden_layers, 'dropout':dropout, 'hidden_dim':hidden_dim, 'model_type':args.model_type}
    if args.model_type == 'KAN':
        grid_size = trial.suggest_int('grid_size', 2, 32)
        spline_order = trial.suggest_int('spline_order', 1, 4)
        params['grid_size'] = grid_size
        params['spline_order'] = spline_order
    elif args.model_type == 'FASTKAN':
        grid_size = trial.suggest_int('grid_size', 2, 32)
        params['grid_size'] = grid_size

    test_loss, n_params = train_model_with_parameters(params, train_loader, val_loader, test_loader)
    return test_loss, n_params

if not os.path.exists('logs'):
    os.makedirs('logs')
log_file = f'logs/QM9_{args.gnn_type}_{args.model_type}'

train_dataset = ZINC('datasets/ZINC', subset=True, split='train')
val_dataset = ZINC('datasets/ZINC', subset=True, split='val')
test_dataset = ZINC('datasets/ZINC', subset=True, split='test')

test_loss_all = []
all_best_hyperparams = []
sizes = []
for it in range(10):
    sampler = optuna.samplers.TPESampler(seed=it)
    torch.cuda.empty_cache()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda trial: objective_function(trial, train_loader, val_loader), n_trials=2, gc_after_trial=True)
    best_hyperparams = study.best_params
    best_hyperparams['model_type'] = args.model_type
    best_hyperparams['gnn_type'] = args.gnn_type
    test_loss, model_size = detailed_objective(study.best_trial, train_loader, val_loader, test_loader)
    all_best_hyperparams.append(best_hyperparams)
    sizes.append(model_size)
    test_loss_all.append(test_loss)
    print(test_loss_all)
    print(all_best_hyperparams)
    with open(log_file, 'a') as file:
        file.write(f'SPLIT {it}\n')
        file.write(f'Losses {test_loss_all}\n')
        file.write(f'Params {all_best_hyperparams}\n')
        file.write(f'Size {sizes}\n')
        file.write('\n')

tensor_loss = torch.tensor(test_loss_all)
print('---------------- Final Result ----------------')
print(f'Mean: {tensor_loss.mean()}, Std: {tensor_loss.std()}\n')
print(all_best_hyperparams)
with open(log_file, 'a') as file:
    file.write(f'SPLIT {it}\n')
    file.write(f'Losses {tensor_loss}\n')
    file.write(f'Params {all_best_hyperparams}\n\n')
    file.write(f'Mean: {tensor_loss.mean()}, Std: {tensor_loss.std()}\n')