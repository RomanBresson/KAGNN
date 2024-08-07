from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import torch
import torch.nn.functional as F
import optuna
import numpy as np

unlabeled_datasets = ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Degree(object):
    def __call__(self, data):
        idx = data.edge_index[0]
        deg = torch.clip(degree(idx, data.num_nodes).long(), 0, 35)
        data.x = F.one_hot(deg, num_classes=36)
        return data

class to_cuda(object):
    def __call__(self, data):
        data.x = data.x.to(torch.device("cuda"))
        data.y = data.y.to(torch.device("cuda"))
        data.edge_index = data.edge_index.to(torch.device("cuda"))
        return data

def train(model, loader, optimizer, device):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)

def val(model, loader, device):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y, reduction='sum').item()
    return loss_all / len(loader.dataset)

def test(model, loader, device):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def count_params(model):
    s = 0
    for k in model.parameters():
        s+= torch.prod(torch.tensor(k.shape))
    return(s)

def parameters_finder(trainer_function, objective_function, log_file, splits, dataset, args):
    for random_seed in [123,1234,12345]:
        all_best_hyperparams = []
        all_best_sizes = []
        accs = []
        torch.manual_seed(random_seed)
        sampler = optuna.samplers.TPESampler(seed=random_seed)
        for it in range(0,10):
            torch.cuda.empty_cache()
            train_index = splits[it]['model_selection'][0]['train']
            val_index = splits[it]['model_selection'][0]['validation']

            val_dataset = dataset[val_index]
            train_dataset = dataset[train_index]

            val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            study = optuna.create_study(sampler=sampler)
            study.optimize(lambda trial: objective_function(trial, train_loader, val_loader), n_trials=100)
            best_hyperparams = study.best_params

            test_accs = []
            for _ in range(3):
                train_index = splits[it]['model_selection'][0]['train']
                val_index = splits[it]['model_selection'][0]['validation']
                test_index = splits[it]['test']

                test_dataset = dataset[test_index]
                val_dataset = dataset[val_index]
                train_dataset = dataset[train_index]

                val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

                print('---------------- Split {} ----------------'.format(it))
                test_acc = trainer_function(best_hyperparams, train_loader, val_loader, test_loader)
                test_accs.append(test_acc)

            accs.append(np.mean(test_accs))
            all_best_hyperparams.append(best_hyperparams)
            print(accs)
            print(all_best_hyperparams)
            with open(log_file, 'a') as file:
                file.write(f'SPLIT {it}\n')
                file.write(f'Accuracies {accs}\n')
                file.write(f'Params {all_best_hyperparams}\n')
                file.write(f'Mean {np.mean(test_accs)}, Std {np.std(test_accs)}\n')
                file.write('\n')

        accs = torch.tensor(accs)
        print('---------------- Final Result ----------------')
        print('Mean: {:7f}, Std: {:7f}'.format(accs.mean(), accs.std()))
        print(all_best_hyperparams)
        with open(log_file, 'a') as file:
            file.write(f'SPLIT {it}\n')
            file.write(f'Accuracies {accs}\n')
            file.write(f'Params {all_best_hyperparams}\n\n')