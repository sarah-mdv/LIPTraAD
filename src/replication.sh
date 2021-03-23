#!/bin/bash
# Written by Sarah Moy de Vitry under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
ROOTDIR=`readlink -f $0 | xargs dirname`
export PYTHONPATH=$PYTHONPATH:$ROOTDIR

python3 main.py \
--data $ROOTDIR/data/TADPOLE_D1_D2.csv\
 --features $ROOTDIR/data/features\
 --folds 10 --strategy forward\
 --batch_size 128 --validation\
 -d debug --i_drop 0.1\
 --h_drop 0.4 --h_size 128\
 --nb_layers 2 --epochs 20\
 --pre_epochs 20 --lr 0.001290666\
 --model rnnpro --weight_decay 1e-5 \
 --out outpred.csv \

 --encoder_model /home/moyde/Documents/Studies/Chalmers/Thesis/LIPTraAD/src/logs/RNN_130123.pt