#!/bin/bash
. /opt/miniconda3/bin/activate csrf
cd /cortexode

python eval.py --test_type='init' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/bsnip/hcp_0/model/' --init_dir='/cortexode/ckpts/bsnip/hcp_0/init/test/' --data_usage='test' --data_name='adni' --surf_hemi='rh' --tag='Unet' --device='gpu'

python eval.py --test_type='init' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/bsnip/hcp_0/model/' --init_dir='/cortexode/ckpts/bsnip/hcp_0/init/train/' --data_usage='train' --data_name='adni' --surf_hemi='rh' --tag='Unet' --device='gpu'

python eval.py --test_type='init' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/bsnip/hcp_0/model/' --init_dir='/cortexode/ckpts/bsnip/hcp_0/init/valid/' --data_usage='valid' --data_name='adni' --surf_hemi='rh' --tag='Unet' --device='gpu'
