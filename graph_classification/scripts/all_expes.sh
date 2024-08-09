#!/usr/bin/env bash

for DATASET in IMDB-B IMDB-M DD ENZYMES NCI1
do
    for MODEL in mlp fastkan
    do
        echo "$MODEL"
        echo "$DATASET"
        bash scripts/$MODEL/$DATASET.sh
    done
done