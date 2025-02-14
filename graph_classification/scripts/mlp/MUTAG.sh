#!/bin/bash

python optuna_graph_classification_mlp.py --dataset MUTAG --patience 20 --epochs 400 --batch-size 32 --model_type GAT
python optuna_graph_classification_mlp.py --dataset MUTAG --patience 20 --epochs 400 --batch-size 32 --model_type GIN
python optuna_graph_classification_mlp.py --dataset MUTAG --patience 20 --epochs 400 --batch-size 32 --model_type GCN