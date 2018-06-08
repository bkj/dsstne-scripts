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

head -n 2000 data/train.txt > data/train-small.txt
head -n 2000 data/test.txt > data/test-small.txt

generateNetCDF -d gl_input -i data/train-small.txt -o data/gl_input.nc -f data/features_input -s data/samples_input -c
generateNetCDF -d gl_output -i data/train-small.txt -o data/gl_output.nc -f data/features_output -s data/samples_input -c

rm models/*
train -b 64 -e 10 -n models/network.nc \
    -d gl \
    -i data/gl_input.nc \
    -o data/gl_output.nc \
    -c config.json

predict -b 2048 -k 10 -n initial_network.nc \
    -d gl \
    -i data/features_input \
    -o data/features_output \
    -f data/train-small.txt \
    -r data/train-small.txt \
    -s results/recs

head results/recs

python inspect-results.py data/test-small.txt results/recs

# >>
# 400, 1 epoch
# p@01 -> 0.241781
# p@05 -> 0.190284
# p@10 -> 0.161136

# 400, 2 epoch
# p@01 -> 0.277884
# p@05 -> 0.224312
# p@10 -> 0.192758

# 400, 5 epoch
# p@01 -> 0.374106
# p@05 -> 0.290595
# p@10 -> 0.247069

# 400, 10 epoch
# p@01 -> 0.446644
# p@05 -> 0.348272
# p@10 -> 0.292293

# <<


# orig
# p@01 -> 0.521853
# p@05 -> 0.415397
# p@10 -> 0.350998

# 800d
# p@01 -> 0.526164
# p@05 -> 0.419376
# p@10 -> 0.353257

# 400d
# p@01 -> 0.503426
# p@05 -> 0.400458
# p@10 -> 0.337643

# 200d
# p@01 -> 0.408143
# p@05 -> 0.319654
# p@10 -> 0.269310

# 800d (no init)
# p@01 -> 0.276180
# p@05 -> 0.247192
# p@10 -> 0.223546

# 800d (no hidden init)
# p@01 -> 0.527565
# p@05 -> 0.420648
# p@10 -> 0.355017

# 800d (no init)
# p@01 -> 0.276180
# p@05 -> 0.247192
# p@10 -> 0.223546

# 800d (no hidden init, nodrop)
# p@01 -> 0.477389
# p@05 -> 0.377570
# p@10 -> 0.318918

# 800d (no hidden init)
# p@01 -> 0.527565
# p@05 -> 0.420648
# p@10 -> 0.355017

# 800d (no hidden init, -5.2 final init)
# p@01 -> 0.515759
# p@05 -> 0.413342
# p@10 -> 0.351118

# 800d (no hidden init, -3.2 final init)
# p@01 -> 0.507679
# p@05 -> 0.406660
# p@10 -> 0.345420

# 800d (no hidden init, -1.2 final init)
# p@01 -> 0.458940
# p@05 -> 0.366049
# p@10 -> 0.312313

# 800d (no hidden init, -0.2 final init)
# p@01 -> 0.392013
# p@05 -> 0.337236
# p@10 -> 0.291696

# 800d (no hidden init, +5.2 final init)
# p@01 -> 0.031807
# p@05 -> 0.038276
# p@10 -> 0.040178
