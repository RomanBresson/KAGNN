#!/bin/bash

python optuna_graph_classification_kan.py --dataset MUTAG --patience 20 --nb_gnn_layers 2 --epochs 400 --batch-size 32 --random_seed 123
python optuna_graph_classification_kan.py --dataset MUTAG --patience 20 --nb_gnn_layers 2 --epochs 400 --batch-size 32 --random_seed 1234
python optuna_graph_classification_kan.py --dataset MUTAG --patience 20 --nb_gnn_layers 2 --epochs 400 --batch-size 32 --random_seed 12345