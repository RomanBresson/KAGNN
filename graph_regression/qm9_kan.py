import argparse
import torch
from torch.optim import Adam

from torch_geometric.datasets import QM9
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

dataset = QM9('datasets/QM9')
dataset.data.y = dataset.data.y[:,0:12]
dataset = dataset.shuffle()

mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean.to(device), std.to(device)

tenpercent = int(len(dataset) * 0.1)
test_dataset = dataset[:tenpercent].shuffle()
val_dataset = dataset[tenpercent:2*tenpercent].shuffle()
train_dataset = dataset[2*tenpercent:].shuffle()

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

LR = [0.001, 0.0001]
HIDDEN_DIM = [4, 8, 16, 32, 64, 128, 256]
N_LAYERS = [2, 3, 4]
GRID_SIZE = [1, 3, 5, 8, 10]
SPLINE_ORDER = [3, 5]

best_val_error = float('inf')
for lr in LR:
    for hidden_dim in HIDDEN_DIM:
        for n_layers in N_LAYERS:
            for grid_size in GRID_SIZE:
                for spline_order in SPLINE_ORDER:
                    print('Evaluating the following hyperparameters:')
                    print('lr:', lr, 'hidden_dim:', hidden_dim, 'n_layers:', n_layers)
                    model = KAGIN(args.n_gnn_layers, dataset.num_features, hidden_dim, n_layers, grid_size, spline_order, 12, args.dropout).to(device)
                    optimizer = Adam(model.parameters(), lr=lr)
                    loss_function = torch.nn.L1Loss()

                    def train():
                        model.train()
                        loss_all = 0

                        for data in train_loader:
                            data = data.to(device)
                            optimizer.zero_grad()
                            loss = loss_function(model(data), data.y)

                            loss.backward()
                            loss_all += loss.item() * data.num_graphs
                            optimizer.step()
                        return (loss_all / len(train_loader.dataset))

                    @torch.no_grad()
                    def test(loader):
                        model.eval()
                        error = torch.zeros([1, 12]).to(device)

                        for data in loader:
                            data = data.to(device)
                            error += ((data.y * std - model(data) * std).abs() / std).sum(dim=0)

                        error = error / len(loader.dataset)

                        return error.mean().item()

                    early_stopper = EarlyStopper(patience=args.patience)
                    for epoch in range(1, args.epochs+1):
                        loss = train()
                        val_error = test(val_loader)

                        if val_error < best_val_error:
                            test_error = test(test_loader)
                            best_val_error = val_error
                            best_hyperparams = {'lr': lr, 'hidden_dim': hidden_dim, 'n_layers': n_layers, 'grid_size': grid_size, 'spline_order': spline_order}
                            print('Epoch: {:03d}, Loss: {:.7f}, Validation MAE: {:.7f}'.format(epoch, loss, val_error))

                        if early_stopper.early_stop(val_error):
                            print(f"Stopped at epoch {epoch}")
                            break

print('Best hyperparameters:')
print('lr:', best_hyperparams['lr'])
print('hidden_dim:', best_hyperparams['hidden_dim'])
print('n_layers:', best_hyperparams['n_layers'])
print('grid_size:', best_hyperparams['grid_size'])
print('spline_order:', best_hyperparams['spline_order'])

results = []
for _ in range(10):
    print()
    print(f'Run {run}:')
    print()

    dataset = dataset.shuffle()
    test_dataset = dataset[:tenpercent].shuffle()
    val_dataset = dataset[tenpercent:2*tenpercent].shuffle()
    train_dataset = dataset[2*tenpercent:].shuffle()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = KAGIN(args.n_gnn_layers, dataset.num_features, best_hyperparams['hidden_dim'], best_hyperparams['n_layers'], best_hyperparams['grid_size'], best_hyperparams['spline_order'], 12, args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', total_params)
    print()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparams['lr'])
    loss_function = torch.nn.L1Loss()

    def train():
        model.train()
        loss_all = 0

        lf = torch.nn.L1Loss()

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(data), data.y)

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return (loss_all / len(train_loader.dataset))

    @torch.no_grad()
    def test(loader):
        model.eval()
        error = torch.zeros([1, 12]).to(device)

        for data in loader:
            data = data.to(device)
            error += ((data.y * std - model(data) * std).abs() / std).sum(dim=0)

        error = error / len(loader.dataset)

        return error.mean().item()

    best_val_error = float('inf')
    early_stopper = EarlyStopper(patience=20)
    for epoch in range(1, args.epochs+1):
        loss = train()
        val_error = test(val_loader)
        
        if val_error < best_val_error:
            test_error = test(test_loader)
            best_val_error = val_error
            print('Epoch: {:03d}, Loss: {:.7f}, Validation MAE: {:.7f}, Test MAE: {:.7f}'.format(epoch, loss, val_error, test_error))

        if early_stopper.early_stop(val_error):
            print(f"Stopped at epoch {epoch}")
            break

    results.append(test_error)

results = torch.tensor(results)
print('===========================')
print(f'Final Test: {results.mean():.4f} ± {results.std():.4f}')