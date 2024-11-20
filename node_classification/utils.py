from typing import Union
import random
import time
import numpy as np
import torch
import torch_geometric as pyg
from torch.optim import Optimizer
import numpy as np

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

def sparse_diag(d):
    i = torch.arange(d.size(0), device=d.device)
    indices = torch.stack([i, i], dim=0)
    return torch.sparse_coo_tensor(indices, d, (d.size(0), d.size(0)))

def train_node_class(mask: torch.tensor,
                      model,
                      data: pyg.data.Data,
                      optimizer: Optimizer,
                      criterion: torch.nn.CrossEntropyLoss):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x,data.edge_index)  # Perform a single forward pass.
    out = torch.softmax(out, dim=1)
    loss = criterion(out[mask], data.y[mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, out

def test_node_class(mask: torch.tensor,
                    model,
                    data: pyg.data.Data):
    model.eval()
    out = model(data.x,data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    return acc

def experiment_node_class(train_mask: torch.tensor,
                          valid_mask: torch.tensor,
                          test_mask: torch.tensor,
                          model,
                          data: pyg.data.Data,
                          optimizer:Optimizer,
                          criterion: torch.nn.CrossEntropyLoss,
                          n_epochs:int,
                          patience: int = 50):
    print(count_params(model))
    best_val_loss = torch.inf
    best_test_acc = 0
    early_stopping = 0
    t = time.time()
    # write the results in a file wicth the dataset name on it
    for _ in range(n_epochs):
        _, output = train_node_class(train_mask, model, data, optimizer, criterion)
        val_loss = criterion(output[valid_mask], data.y[valid_mask])
        if val_loss < best_val_loss:
            early_stopping=0
            best_val_loss = val_loss
            best_test_acc = test_node_class(test_mask, model, data)
        else:
            early_stopping += 1
            if early_stopping > patience:
                print("early stopping..")
                break

    return best_val_loss, best_test_acc, time.time()-t