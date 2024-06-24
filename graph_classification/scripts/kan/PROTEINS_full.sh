#!/bin/bash

python optuna_graph_classification_kan.py --dataset PROTEINS_full --patience 10 --nb_gnn_layers 2 --epochs 400 --batch-size 64