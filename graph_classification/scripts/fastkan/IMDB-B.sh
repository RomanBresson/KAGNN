#!/bin/bash

python optuna_graph_classification_fastkan.py --dataset IMDB-BINARY --patience 20 --nb_gnn_layers 2 --epochs 400 --batch-size 64