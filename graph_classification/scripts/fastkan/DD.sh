#!/bin/bash

python optuna_graph_classification_fastkan.py --dataset DD --patience 20 --nb_gnn_layers 3 --epochs 400 --batch-size 64
