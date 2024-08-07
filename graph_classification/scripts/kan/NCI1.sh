#!/bin/bash

python optuna_graph_classification_kan.py --dataset NCI1 --patience 20 --nb_gnn_layers 5 --epochs 400 --batch-size 129 --random_seed 123
python optuna_graph_classification_kan.py --dataset NCI1 --patience 20 --nb_gnn_layers 5 --epochs 400 --batch-size 129 --random_seed 1234
python optuna_graph_classification_kan.py --dataset NCI1 --patience 20 --nb_gnn_layers 5 --epochs 400 --batch-size 129 --random_seed 12345