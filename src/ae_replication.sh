#!/bin/bash
# Written by Sarah Moy de Vitry under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md


ROOTDIR=`readlink -f $0 | xargs dirname`
export PYTHONPATH=$PYTHONPATH:$ROOTDIR
let counter=13
while [ $counter -le 15 ]
do
    python3 main.py \
    --data $ROOTDIR/data/TADPOLE_D1_D2.csv \
    --features $ROOTDIR/data/features \
    --folds 5 --strategy forward --batch_size 1 \
    --validation -d debug --i_drop 0.1 --h_drop 0.3 \
    --h_size 128 --nb_layers 2 --epochs 10 --lr 0.005\
    --model pro-ae  --out outpred.csv --weight_decay 0.005\
    --n_prototypes $counter
     ((counter++))
done


python3 main.py --data $ROOTDIR/data/TADPOLE_D1_D2.csv --features $ROOTDIR/data/features --folds 10 --strategy forward --batch_size 1 --validation -d debug --i_drop 0.1 --h_drop 0.3 --h_size 128 --nb_layers 2 --epochs 7 --lr 0.01 --weight_decay 0.0001 --model pro-ae --out outpred.csv
