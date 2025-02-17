#%%
import argparse
import torch
from utils import run_experiment
import optuna

parser = argparse.ArgumentParser(description='Node_classif')
parser.add_argument('--dataset', default='Cora', help='Dataset name')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
parser.add_argument('--patience', type=int, default=100, help='Patience of early stopping')
parser.add_argument('--random_seed', type=int, default=12345, help='Random seed')
parser.add_argument('--conv_type', default='gat', help='gin/gcn/gat')
parser.add_argument('--architecture', default='mlp', help='MLP/KAN/FASTKAN')
parser.add_argument('--rate_print', type=int, default=1000, help='Print frequency')
parser.add_argument('--skip', type=int, default=1, help='skip connections')
parser.add_argument('--heads', type=int, default=4, help='head count for multihead attention')
args = parser.parse_args()

def objective(trial, dataset_name, args):
    params = {'dataset': dataset_name,
            'hidden_channels':0,
            'conv_type':args.conv_type,
            'architecture':args.architecture,
            'patience': args.patience,
            'epochs': args.epochs,
            'hidden_layers': 0,
            'dropout': 0.,
            'grid_size': 0,
            'spline_order': 0,
            'rate_print': args.rate_print,
            'skip': args.skip,
            'heads': args.heads
        }
    params['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    params['dropout'] = trial.suggest_float('dropout', 0, 0.9)
    if params['conv_type']=='gin':
        params['hidden_layers'] = trial.suggest_int('hidden_layers', 1, 4)
    if params['architecture']=='mlp':
        params['hidden_channels'] = trial.suggest_int('hidden_channels', 1, 256)
    elif params['architecture']=='fastkan':
        params['hidden_channels'] = trial.suggest_int('hidden_channels', 2, 128)
        params['grid_size'] = trial.suggest_int('grid_size', 2, 32)
    elif params['architecture']=='kan':
        params['hidden_channels'] = trial.suggest_int('hidden_channels', 2, 128)
        params['grid_size'] = trial.suggest_int('grid_size', 1, 8)
        params['spline_order'] = trial.suggest_int('spline_order', 1, 3)
    mva,_,_,_ = run_experiment(params, dataset_name)
    return(mva)

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, args.dataset, args), n_trials=100)
log_file = f'logs/{args.dataset}_{args.architecture}_{args.conv_type}'
best_params = study.best_params
params = {
        'dataset': args.dataset,
        'conv_type':args.conv_type,
        'architecture':args.architecture,
        'patience': args.patience,
        'epochs': args.epochs,
        'rate_print': args.rate_print,
        'hidden_layers': 0,
        'skip': args.skip,
        'heads': args.heads
    }
for bp,v in best_params.items():
    params[bp] = v

test_accs = []
print("Evaluating")
for _ in range(3):
    _,_,_,test_acc_tens = run_experiment(params, args.dataset)
    test_accs.append(test_acc_tens)
test_accs = torch.cat(test_accs)
tm,ts = test_accs.mean(), test_accs.std()

with open(log_file + '_finished', 'a') as file:
    file.write(f'Mean: {tm.item()}, Std: {ts.item()}')
