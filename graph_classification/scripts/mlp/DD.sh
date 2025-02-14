#!/bin/bash

python optuna_graph_classification_mlp.py --dataset DD --patience 20 --nb_gnn_layers 3 --epochs 400 --batch-size 64 --model_type GAT
python optuna_graph_classification_mlp.py --dataset DD --patience 20 --nb_gnn_layers 3 --epochs 400 --batch-size 64 --model_type GIN
python optuna_graph_classification_mlp.py --dataset DD --patience 20 --nb_gnn_layers 3 --epochs 400 --batch-size 64 --model_type GCN