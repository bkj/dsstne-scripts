#!/bin/bash

# py_run.sh


CUDA_VISIBLE_DEVICES=1 python main.py --emb-dim 100 > results/py-100-shuffle.jl
CUDA_VISIBLE_DEVICES=2 python main.py --emb-dim 400 > results/py-400-shuffle.jl

CUDA_VISIBLE_DEVICES=3 python main.py --emb-dim 100 --sortish > results/py-100-sortish.jl
CUDA_VISIBLE_DEVICES=4 python main.py --emb-dim 400 --sortish > results/py-400-sortish.jl

CUDA_VISIBLE_DEVICES=1 python main.py --max-obs 1280 \
    --emb-dim 400 --eval-interval 1 --epochs 10 --no-verbose


CUDA_VISIBLE_DEVICES=1 python main.py \
    --emb-dim 800 \
    --eval-interval 1 \
    --epochs 10 \
    --batch-size 256 | tee results/tmp5


