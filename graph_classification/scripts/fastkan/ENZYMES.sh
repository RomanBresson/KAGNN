#!/bin/bash

python optuna_graph_classification_fastkan.py --dataset ENZYMES --patience 20 --nb_gnn_layers 4 --epochs 400 --batch-size 64