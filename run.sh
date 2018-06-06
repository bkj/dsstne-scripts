#!/bin/bash

# run.sh

mkdir -p {data,models,results}

# --
# IO

unzip ml-20m.zip

python prep.py

export PATH="/home/bjohnson/software/amazon-dsstne/src/amazon/dsstne/bin/:$PATH"

# Generate NetCDF files for input and output layers
generateNetCDF -d gl_input -i data/train.txt -o data/gl_input.nc -f data/features_input -s data/samples_input -c
generateNetCDF -d gl_output -i data/train.txt -o data/gl_output.nc -f data/features_output -s data/samples_input -c

# --
# Run

CONFIG="config.json"

train -b 1024 -e 50 -n models/git_network.nc \
    -d gl \
    -i data/gl_input.nc \
    -o data/gl_output.nc \
    -c $CONFIG 

predict -b 2048 -k 10 -n models/git_network.nc \
    -d gl \
    -i data/features_input \
    -o data/features_output \
    -f data/train.txt \
    -r data/train.txt \
    -s results/recs

head results/recs

python inspect-results.py data/test.txt results/github