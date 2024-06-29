import argparse
import torch
from torch.optim import Adam

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

from models import KAGIN
from utils import EarlyStopper

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability)')
parser.add_argument('--patience', type=int, default=20, help='Patience for ealry stopping')
parser.add_argument('--n-gnn-layers', type=int, default=4, help='Number of message passing layers')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = ZINC('datasets/ZINC', subset=True, split='train')
val_dataset = ZINC('datasets/ZINC', subset=True, split='val')
test_dataset = ZINC('datasets/ZINC', subset=True, split='test')

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

LR = [0.001, 0.0001]
HIDDEN_DIM = [4, 8, 16, 32, 64, 128, 256]
N_LAYERS = [2, 3, 4]
GRID_SIZE = [1, 3, 5, 8, 10]
SPLINE_ORDER = [3, 5]

best_val_mae = float('inf')
for lr in LR:
    for hidden_dim in HIDDEN_DIM:
        for n_layers in N_LAYERS:
            for grid_size in GRID_SIZE:
                for spline_order in SPLINE_ORDER:
                    print('Evaluating the following hyperparameters:')
                    print('lr:', lr, 'hidden_dim:', hidden_dim, 'n_layers:', n_layers, 'grid_size:', grid_size, 'spline_order:', spline_order)
                    model = KAGIN(args.n_gnn_layers, 21, hidden_dim, n_layers, grid_size, spline_order, 1, args.dropout, True).to(device)
                    optimizer = Adam(model.parameters(), lr=lr)

                    def train(epoch):
                        model.train()

                        total_loss = 0
                        for data in train_loader:
                            data = data.to(device)
                            optimizer.zero_grad()
                            loss = (model(data).squeeze() - data.y).abs().mean()
                            loss.backward()
                            total_loss += loss.item() * data.num_graphs
                            optimizer.step()
                        return total_loss / len(train_loader.dataset)

                    @torch.no_grad()
                    def test(loader):
                        model.eval()

                        total_error = 0
                        for data in loader:
                            data = data.to(device)
                            total_error += (model(data).squeeze() - data.y).abs().sum().item()
                        return total_error / len(loader.dataset)

                    early_stopper = EarlyStopper(patience=args.patience)
                    for epoch in range(1, args.epochs+1):
                        loss = train(epoch)
                        val_mae = test(val_loader)

                        if val_mae < best_val_mae:
                            best_val_mae = val_mae
                            best_hyperparams = {'lr': lr, 'hidden_dim': hidden_dim, 'n_layers': n_layers, 'grid_size': grid_size, 'spline_order': spline_order}
                            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_mae:.4f}')
                        
                        if early_stopper.early_stop(val_mae):
                            print(f"Stopped at epoch {epoch}")
                            break

print('Best hyperparameters:')
print('lr:', best_hyperparams['lr'])
print('hidden_dim:', best_hyperparams['hidden_dim'])
print('n_layers:', best_hyperparams['n_layers'])
print('grid_size:', best_hyperparams['grid_size'])
print('spline_order:', best_hyperparams['spline_order'])

val_maes = []
test_maes = []
for run in range(10):
    print()
    print(f'Run {run}:')
    print()

    model = KAGIN(args.n_gnn_layers, 21, best_hyperparams['hidden_dim'], best_hyperparams['n_layers'], best_hyperparams['grid_size'], best_hyperparams['spline_order'], 1, args.dropout, True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', total_params)
    print()
    optimizer = Adam(model.parameters(), lr=best_hyperparams['lr'])

    def train(epoch):
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = (model(data).squeeze() - data.y).abs().mean()
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()

        total_error = 0
        for data in loader:
            data = data.to(device)
            total_error += (model(data).squeeze() - data.y).abs().sum().item()
        return total_error / len(loader.dataset)

    best_val_mae = test_mae = float('inf')
    early_stopper = EarlyStopper(patience=args.patience)
    for epoch in range(1, args.epochs+1):
        loss = train(epoch)
        val_mae = test(val_loader)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            test_mae = test(test_loader)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                f'Val: {val_mae:.4f}, Test: {test_mae:.4f}')

        if early_stopper.early_stop(val_mae):
            print(f"Stopped at epoch {epoch}")
            break

    test_maes.append(test_mae)
    val_maes.append(best_val_mae)

test_mae = torch.tensor(test_maes)
print('===========================')
print(f'Final Test: {test_mae.mean():.4f} ± {test_mae.std():.4f}')
