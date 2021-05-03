#!/bin/bash
# Written by Sarah Moy de Vitry under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md


ROOTDIR=`readlink -f $0 | xargs dirname`
export PYTHONPATH=$PYTHONPATH:$ROOTDIR

for lr in 0.001 0.005 0.01
do
    for wd in 0.01 0.05 0.1
    do
        for bs in 16 64 128
        do
            python3 main.py \
            --data $ROOTDIR/data/TADPOLE_D1_D2.csv \
            --features $ROOTDIR/data/features \
            --folds 10 --strategy forward --batch_size $bs \
            --validation -d debug --i_drop 0.1 --h_drop 0.4 \
            --h_size 128 --nb_layers 2 --epochs 10 --lr $lr\
            --model ae  --out outpred.csv --weight_decay $wd
        done
    done
done