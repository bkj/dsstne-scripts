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

# head -n 2000 data/train.txt > data/train-small.txt
# head -n 2000 data/test.txt > data/test-small.txt

generateNetCDF -d gl_input -i data/train.txt -o data/gl_input.nc -f data/features_input -s data/samples_input -c
generateNetCDF -d gl_output -i data/train.txt -o data/gl_output.nc -f data/features_output -s data/samples_input -c

rm models/*
train -b 1024 -e 50 -n models/network.nc \
    -d gl \
    -i data/gl_input.nc \
    -o data/gl_output.nc \
    -c config.json

predict -b 2048 -k 10 -n models/network.nc \
    -d gl \
    -i data/features_input \
    -o data/features_output \
    -f data/train.txt \
    -r data/train.txt \
    -s results/recs

head results/recs

python inspect-results.py data/test.txt results/recs

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

# 800, 10 epoch, bs=128
# p@01 -> 0.510567
# p@05 -> 0.403447
# p@10 -> 0.338300

# 400, 10 epoch, bs=128
# p@01 -> 0.484024
# p@05 -> 0.380580
# p@10 -> 0.318329

# 400, 50 epoch, bs=128
# p@01 -> 0.473547
# p@05 -> 0.372754
# p@10 -> 0.312204

# 400, 50 epoch, bs=256
# p@01 -> 0.478717
# p@05 -> 0.377697
# p@10 -> 0.317628

# 400, 50 epoch, bs=1024
# p@01 -> 0.490400
# p@05 -> 0.387940
# p@10 -> 0.328886

# 800, 50 epoch, bs=1024
# p@01 -> 0.529767
# p@05 -> 0.419045
# p@10 -> 0.352933

# 800, 50 epoch, bs=1024, nohidden
# p@01 -> 0.416180
# p@05 -> 0.328593
# p@10 -> 0.281058

# <<


# orig
# p@01 -> 0.521853
# p@05 -> 0.415397
# p@10 -> 0.350998

# 800d (50e?)
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
