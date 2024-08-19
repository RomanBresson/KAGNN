# Install required packages.
import json

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from torch_geometric.datasets import Actor, WebKB
from ogb.nodeproppred import PygNodePropPredDataset

from utils import set_seed, sparse_diag

from optuna_node_classification_fastkan import train_and_evaluate_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = 1000
    skip = True
    log = open("results/test_log_fastkan_time.txt", "a")
    set_seed(1)
    for dataset_name in ['Cora', 'CiteSeer', 'Actor', 'Texas','Cornell','Wisconsin', 'ogbn-arxiv']:
        for conv_type in ['gin','gcn']:
            print(dataset_name+ " "+conv_type)
            # free the torch memory
            torch.cuda.empty_cache()
            if dataset_name == 'ogbn-arxiv':
                dataset = PygNodePropPredDataset(name=dataset_name, root='data/'+dataset_name)
            elif dataset_name in ['Cora', 'CiteSeer']:
                dataset = Planetoid(root='data/'+dataset_name, name=dataset_name, transform=NormalizeFeatures())
            elif dataset_name in ['Actor']:
                dataset = Actor(root='data/'+dataset_name)
                dataset.transform = NormalizeFeatures()
            else:
                dataset = WebKB(root='data/'+dataset_name, name=dataset_name, transform=NormalizeFeatures())
            data = dataset[0] # Get the first graph object.
            with open(f'results/best_params_fastkan_{conv_type}_{dataset_name}.json', 'r') as f:
                best_params = json.load(f)
            hidden_channels = best_params['hidden_channels']
            lr = best_params['lr']
            grid_size = best_params['grid_size']
            hidden_layers = best_params['hidden_layers']
            regularizer = best_params['regularizer']
            if conv_type=='gcn':
                N = data.edge_index.max().item() + 1
                data.edge_index = data.edge_index.to("cpu")
                A = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)), (N, N))
                I = torch.sparse_coo_tensor(torch.arange(N).repeat(2,1), torch.ones(N), (N, N))
                A_hat = A + I
                # can do that because D_hat is a vector here
                D_hat = torch.sparse.sum(A_hat, dim=1).to_dense()
                D_hat =  1.0 / torch.sqrt(D_hat)
                D_hat_inv_sqrt = sparse_diag(D_hat)
                data.edge_index  = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
            data = data.to(device)
            test_accs = []
            times = []
            for i in range(10):
                torch.manual_seed(i)
                _, test_acc, time_ = train_and_evaluate_model(hidden_channels=hidden_channels, lr=lr,
                        hidden_layers=hidden_layers, regularizer=regularizer, data=data, dataset_name=dataset_name,
                        dataset=dataset, conv_type=conv_type, skip=skip, grid_size=grid_size, n_epochs=n_epochs, device=device)
                times.append(time_)
                test_accs.append(test_acc)
                print(test_accs)
            print(f"Test mean acc: {torch.tensor(test_accs).mean():.4f}, Test sd: {torch.tensor(test_accs).std():.4f}\n, Time: {time_:.4f}")
            log.write(f"{dataset_name} {conv_type} accuracy mean {torch.tensor(test_accs).mean():.4f} , sd {torch.tensor(test_accs).std():.4f}, Time: {time_:.4f}\n")

if __name__ == "__main__":
    main()