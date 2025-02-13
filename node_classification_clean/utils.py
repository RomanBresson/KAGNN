from typing import Union
import random
import time
import numpy as np
import torch
import torch_geometric as pyg
from torch.optim import Optimizer
from models import GNN_Nodes, GKAN_Nodes, GFASTKAN_Nodes
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Actor, WebKB
from ogb.nodeproppred import PygNodePropPredDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

#nb of mp layers per dataset
dataset_layers = {'Cora':2, 'CiteSeer':2, 'Actor':4, 'Texas':3, 'Cornell':3, 'Wisconsin':3, 'ogbn-arxiv':3}

def count_params(model):
    s = 0
    for k in model.parameters():
        s+= torch.prod(torch.tensor(k.shape))
    return s

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data(dataset_name):
    if dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=dataset_name, root='data/'+dataset_name)
        split_idx = dataset.get_idx_split()
        test_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool).squeeze()
        valid_mask  = torch.zeros(dataset.num_nodes, dtype=torch.bool).squeeze()
        train_mask  = torch.zeros(dataset.num_nodes, dtype=torch.bool).squeeze()
        dataset.y = dataset.y.squeeze()
        train_mask[split_idx['train']] = True
        valid_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True
        dataset.train_masks = train_mask.repeat(10,1)
        dataset.val_masks = valid_mask.repeat(10,1)
        dataset.test_masks  = test_mask.repeat(10,1)
    elif dataset_name in ['Cora', 'CiteSeer']:
        dataset = Planetoid(root='data/'+dataset_name, name=dataset_name, transform=NormalizeFeatures())
        dataset.train_masks = dataset.train_mask.repeat(10,1)
        dataset.val_masks = dataset.val_mask.repeat(10,1)
        dataset.test_masks = dataset.test_mask.repeat(10,1)
    elif dataset_name in ['Actor']:
        dataset = Actor(root='data/'+dataset_name)
        dataset.transform = NormalizeFeatures()
        dataset.train_masks = dataset[0].train_mask.T
        dataset.val_masks = dataset[0].val_mask.T
        dataset.test_masks = dataset[0].test_mask.T
    else:
        dataset = WebKB(root='data/'+dataset_name, name=dataset_name, transform=NormalizeFeatures())
        dataset.train_masks = dataset[0].train_mask.T
        dataset.val_masks = dataset[0].val_mask.T
        dataset.test_masks = dataset[0].test_mask.T
    dataset.x = dataset.x.to(device)
    dataset.y = dataset.y.to(device)
    dataset.edge_index = dataset.edge_index.to(device)
    dataset.train_masks = dataset.train_masks.to(device)
    dataset.val_masks = dataset.val_masks.to(device)
    dataset.test_masks = dataset.test_masks.to(device)
    return(dataset)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        should_save = False
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            should_save = True
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping")
                return False, True
        return should_save, False

def make_model(params):
    if params['architecture']=='mlp':
        model = GNN_Nodes(conv_type=params['conv_type'],
                            mp_layers=params['mp_layers'],
                            num_features=params['num_features'],
                            hidden_channels=params['hidden_channels'],
                            num_classes=params['num_classes'],
                            skip=params['skip'],
                            hidden_layers=params['hidden_layers'],
                            dropout=params['dropout'],
                            heads=params['heads'])
    elif params['architecture']=='kan':
        model = GKAN_Nodes(conv_type=params['conv_type'],
                            mp_layers=params['mp_layers'],
                            num_features=params['num_features'],
                            hidden_channels=params['hidden_channels'],
                            num_classes=params['num_classes'],
                            skip=params['skip'],
                            hidden_layers=params['hidden_layers'],
                            dropout=params['dropout'],
                            grid_size=params['grid_size'],
                            spline_order=params['spline_order'],
                            heads=params['heads'])
    elif params['architecture']=='fastkan':
        model = GFASTKAN_Nodes(conv_type=params['conv_type'],
                            mp_layers=params['mp_layers'],
                            num_features=params['num_features'],
                            hidden_channels=params['hidden_channels'],
                            num_classes=params['num_classes'],
                            skip=params['skip'],
                            hidden_layers=params['hidden_layers'],
                            dropout=params['dropout'],
                            grid_size=params['grid_size'],
                            heads=params['heads'])
    print(count_params(model))
    return(model)

def train_one_epoch(model, data, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x,data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[train_mask], data.y[train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, out

def evaluate_accuracy(model, data, mask):
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
        return acc

def evaluate_loss(model, data, mask, criterion):
    with torch.no_grad():
        model.eval()
        out = model(data.x,data.edge_index)
        loss = criterion(out[mask], data.y[mask])
        return loss, out

def efficient_evaluation_accuracy(y, out, mask):
    with torch.no_grad():
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[mask] == y[mask]  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
        return acc

def efficient_evaluation_loss(y, out, mask, criterion):
    with torch.no_grad():
        loss = criterion(out[mask], y[mask])
        return loss

def train_total(model, params, data, train_mask, val_mask, test_mask=None):
    torch.save(model, f"models_saves/{params['dataset']}_{params['architecture']}_{params['conv_type']}")
    if test_mask is None:
        test_mask = val_mask
    early_stopper = EarlyStopper(patience=params['patience'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(params['epochs']):
        train_one_epoch(model, data, train_mask, optimizer, criterion)
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = efficient_evaluation_loss(data.y, out, val_mask, criterion)
        if not ((epoch+1)%params['rate_print']):
            with torch.no_grad():
                train_acc = efficient_evaluation_accuracy(data.y, out, train_mask)
                val_acc = efficient_evaluation_accuracy(data.y, out, val_mask)
                test_acc = efficient_evaluation_accuracy(data.y, out, test_mask)
                print(f'Train acc: {train_acc}, Val acc: {val_acc}, Test acc: {test_acc}')
        should_save, should_stop = early_stopper.early_stop(val_loss)
        if should_save:
            torch.save(model, f"models_saves/{params['dataset']}_{params['architecture']}_{params['conv_type']}")
        if should_stop:
            break
    print("load")
    model = torch.load(f"models_saves/{params['dataset']}_{params['architecture']}_{params['conv_type']}")
    train_acc = efficient_evaluation_accuracy(data.y, out, train_mask)
    val_acc = efficient_evaluation_accuracy(data.y, out, val_mask)
    test_acc = efficient_evaluation_accuracy(data.y, out, test_mask)
    val_loss = efficient_evaluation_loss(data.y, out, val_mask, criterion)
    print(f'Train acc: {train_acc}, Val acc: {val_acc}, Test acc: {test_acc}')
    return(model, train_acc, val_acc, val_loss, test_acc)

def all_splits(params, data):
    models = []
    train_accs = []
    val_accs = []
    val_losses = []
    test_accs = []
    for id_split in range(data.train_masks.shape[0]):
        print(f"Split {id_split}")
        model = make_model(params).to(device)
        train_mask, val_mask, test_mask = data.train_masks[id_split], data.val_masks[id_split], data.test_masks[id_split]
        model, train_acc, val_acc, val_loss, test_acc = train_total(model, params, data, train_mask, val_mask, test_mask)
        models.append(model)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        val_losses.append(val_loss)
    return(models, train_accs, val_accs, val_losses, test_accs)

def run_experiment(params, data_name):
    print("Testing params")
    print(params)
    log_file = f"logs/{data_name}_{params['architecture']}_{params['conv_type']}"
    data = load_data(data_name)
    params['mp_layers'] = dataset_layers[data_name]
    params['num_classes'] = data.num_classes
    params['num_features'] = data.num_features
    models, train_accs, val_accs, val_losses, test_accs = all_splits(params, data)
    val_losses_tens = torch.tensor(val_losses)
    test_accs_tens = torch.tensor(test_accs)
    mvl = val_losses_tens.mean()
    mta,mts = test_accs_tens.mean(), test_accs_tens.std()
    with open(log_file, 'a') as file:
        file.write(str(params))
        file.write('\n')
        file.write(f'Train Accs: {train_accs}')
        file.write('\n')
        file.write(f'Val_losses: {[v.item() for v in val_losses]}, mean: {mvl.item()}')
        file.write('\n')
        file.write(f'Test Accs: {test_accs}, mean: {mta.item()}, std: {mts.item()}')
        file.write('\n')
        file.write('\n')
    return mvl,mta,mts,test_accs_tens