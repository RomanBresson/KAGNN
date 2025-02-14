#!/bin/bash

python optuna_graph_classification_kan.py --dataset ENZYMES --patience 20 --nb_gnn_layers 4 --epochs 400 --batch-size 64 --model_type GAT
python optuna_graph_classification_kan.py --dataset ENZYMES --patience 20 --nb_gnn_layers 4 --epochs 400 --batch-size 64 --model_type GIN
python optuna_graph_classification_kan.py --dataset ENZYMES --patience 20 --nb_gnn_layers 4 --epochs 400 --batch-size 64 --model_type GCN