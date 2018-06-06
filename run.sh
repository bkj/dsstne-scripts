#!/bin/bash

# run.sh

mkdir -p {data,models,results}

# --
# IO

wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip && mv ml-20m data && rm ml-20m.zip

python prep.py --inpath data/ml-20m/ratings.csv --outpath data

# --
# Run

export PATH="/home/bjohnson/software/amazon-dsstne/src/amazon/dsstne/bin/:$PATH"

generateNetCDF -d gl_input -i data/train.txt -o data/gl_input.nc -f data/features_input -s data/samples_input -c
generateNetCDF -d gl_output -i data/train.txt -o data/gl_output.nc -f data/features_output -s data/samples_input -c

train -b 1024 -e 50 -n models/git_network.nc \
    -d gl \
    -i data/gl_input.nc \
    -o data/gl_output.nc \
    -c config.json

predict -b 2048 -k 10 -n models/git_network.nc \
    -d gl \
    -i data/features_input \
    -o data/features_output \
    -f data/train.txt \
    -r data/train.txt \
    -s results/recs

head results/recs

python inspect-results.py data/test.txt results/github