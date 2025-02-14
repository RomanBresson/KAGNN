#!/bin/bash

python optuna_graph_classification_kan.py --dataset NCI1 --patience 20 --epochs 400 --batch-size 129 --model_type GAT
python optuna_graph_classification_kan.py --dataset NCI1 --patience 20 --epochs 400 --batch-size 129 --model_type GIN
python optuna_graph_classification_kan.py --dataset NCI1 --patience 20 --epochs 400 --batch-size 129 --model_type GCN