#!/bin/bash
# Written by Sarah Moy de Vitry under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md


ROOTDIR=`readlink -f $0 | xargs dirname`
export PYTHONPATH=$PYTHONPATH:$ROOTDIR
let counter=1
while [ $counter -le 15 ]
do
    python3 main.py \
    --data $ROOTDIR/data/TADPOLE_D1_D2.csv \
    --features $ROOTDIR/data/features \
    --folds 5 --strategy forward --batch_size 8 \
    --validation -d debug --i_drop 0.1 --h_drop 0.3 \
    --h_size 128 --nb_layers 2 --epochs 7 --lr 0.05\
    --model aecluster  --out outpred.csv --weight_decay 0.001\
    --n_prototypes $counter \
    --encoder_model /home/moyde/Documents/Studies/Chalmers/Thesis/LIPTraAD/src/logs/results/170621-104936_lṛ_0.01_wd_0.0001_bs_8_n_prototypes_4_model_aecluster/ClusterAE.pt
     ((counter++))
done


