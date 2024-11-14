#!/bin/bash
. /opt/miniconda3/bin/activate csrf
cd /cortexode
#python train.py --seg_model_type='MonaiUnet' --train_type='seg' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='./ckpts/exp_csrf_6.2/model/' --data_name='adni' --n_epoch=205 --tag='MonaiUnet' --device='gpu'
python train.py --seg_model_type='Unet' --train_type='seg' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='./ckpts/bsnip/hcp_0/model/' --data_name='adni' --n_epoch=205 --tag='Unet' --device='gpu'

