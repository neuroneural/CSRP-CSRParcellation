#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode
python train.py --train_type='seg' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='./ckpts/experiment_5/model/' --data_name='adni' --n_epoch=205 --tag='exp5' --device='gpu'

