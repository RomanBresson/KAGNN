#!/usr/bin/env bash

for DATASET in MUTAG PROTEINS_full IMDB-B IMDB-M DD ENZYMES NCI1
do
    for MODEL in mlp fastkan kan
    do
        echo "$MODEL"
        echo "$DATASET"
        bash scripts/$MODEL/$DATASET.sh
    done
done