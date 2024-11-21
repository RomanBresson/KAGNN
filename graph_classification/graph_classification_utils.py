import os
import json
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import torch
import torch.nn.functional as F
import optuna

unlabeled_datasets = ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']

layers_per_dataset = {'IMDB-BINARY':2, 'IMDB-MULTI':2, 'MUTAG':2, 'PROTEINS_full':2, 'DD':3, 'ENZYMES':4, 'NCI1':5}

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
        data.x = F.one_hot(deg, num_classes=36).float()
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
        loss = F.nll_loss(model(data), data.y)
        optimizer.zero_grad()
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
    return s

def get_data_and_splits(args):
    use_node_attr = False
    if args.dataset == 'ENZYMES' or args.dataset == 'PROTEINS_full':
        use_node_attr = True
    if args.dataset in unlabeled_datasets:
        dataset = TUDataset(root='datasets/'+args.dataset, name=args.dataset, transform=Degree())
    else:
        dataset = TUDataset(root='datasets/'+args.dataset, name=args.dataset, use_node_attr=use_node_attr)
    with open(os.path.join('data_splits',args.dataset+'_splits.json'),'rt') as f:
        for line in f:
            splits = json.loads(line)
    return(splits, dataset)

def parameters_finder(trainer_function, objective_function, log_file, args):
    random_seed = args.random_seed
    all_accs = []
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    splits, dataset = get_data_and_splits(args)
    test_accs_for_this_seed = []
    all_best_hyperparams = []
    sizes = []
    for it in range(10):
        torch.cuda.empty_cache()
        train_index = splits[it]['model_selection'][0]['train']
        val_index = splits[it]['model_selection'][0]['validation']

        val_dataset = dataset[val_index]
        train_dataset = dataset[train_index]

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        study = optuna.create_study(sampler=sampler)
        study.optimize(lambda trial: objective_function(trial, train_loader, val_loader), n_trials=100, gc_after_trial=True)
        best_hyperparams = study.best_params
        best_hyperparams['model_type'] = args.model_type
        test_accs_for_this_split = []
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

            print(f'---------------- Split {it} ----------------')
            test_acc, model_size = trainer_function(best_hyperparams, train_loader, val_loader, test_loader)
            test_accs_for_this_split.append(test_acc)
            print(test_accs_for_this_split)
        all_best_hyperparams.append(best_hyperparams)
        sizes.append(model_size)
        test_accs_tensor = torch.tensor(test_accs_for_this_split)
        tat_mean = test_accs_tensor.mean().item()
        tat_std = test_accs_tensor.std().item()
        test_accs_for_this_seed.append(tat_mean)
        print(test_accs_for_this_seed)
        print(all_best_hyperparams)
        with open(log_file, 'a') as file:
            file.write(f'SPLIT {it}\n')
            file.write(f'Accuracies {test_accs_for_this_seed}\n')
            file.write(f'Params {all_best_hyperparams}\n')
            file.write(f'Size {sizes}\n')
            file.write(f'Mean {tat_mean}, Std {tat_std}\n')
            file.write('\n')

    tensor_accs = torch.tensor(test_accs_for_this_seed)
    all_accs += test_accs_for_this_seed
    print('---------------- Final Result ----------------')
    print(f'Mean: {tensor_accs.mean()}, Std: {tensor_accs.std()}\n')
    print(all_best_hyperparams)
    with open(log_file, 'a') as file:
        file.write(f'SPLIT {it}\n')
        file.write(f'Accuracies {tensor_accs}\n')
        file.write(f'Params {all_best_hyperparams}\n\n')
        file.write(f'Mean: {tensor_accs.mean()}, Std: {tensor_accs.std()}\n')
