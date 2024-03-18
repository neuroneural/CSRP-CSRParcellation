#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode
python train.py --train_type='seg' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='./ckpts/experiment_4/model/' --data_name='hcp' --n_epoch=205 --tag='exp4' --device='gpu'

