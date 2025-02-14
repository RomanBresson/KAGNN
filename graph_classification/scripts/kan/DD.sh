#!/bin/bash

python optuna_graph_classification_kan.py --dataset DD --patience 20 --epochs 400 --batch-size 64 --model_type GAT
python optuna_graph_classification_kan.py --dataset DD --patience 20 --epochs 400 --batch-size 64 --model_type GIN
python optuna_graph_classification_kan.py --dataset DD --patience 20 --epochs 400 --batch-size 64 --model_type GCN
