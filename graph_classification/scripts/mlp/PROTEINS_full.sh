#!/bin/bash

python optuna_graph_classification_mlp.py --dataset PROTEINS_full --patience 20 --nb_gnn_layers 2 --epochs 400 --batch-size 64 --random_seed 123
python optuna_graph_classification_mlp.py --dataset PROTEINS_full --patience 20 --nb_gnn_layers 2 --epochs 400 --batch-size 64 --random_seed 1234
python optuna_graph_classification_mlp.py --dataset PROTEINS_full --patience 20 --nb_gnn_layers 2 --epochs 400 --batch-size 64 --random_seed 12345