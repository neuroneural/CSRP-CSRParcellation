#!/bin/bash
. /opt/miniconda3/bin/activate csrf
cd /cortexode

python eval.py --test_type='init' --data_dir='/speedrun/cortexode-data-rp/test/' --model_dir='/cortexode/ckpts/hcp_csrf_gnn_0/model/' --init_dir='/cortexode/ckpts/hcp_csrf_gnn_0/init/test/' --data_name='hcp' --surf_hemi='rh' --tag='Unet' --device='gpu'

python eval.py --test_type='init' --data_dir='/speedrun/cortexode-data-rp/train/' --model_dir='/cortexode/ckpts/hcp_csrf_gnn_0/model/' --init_dir='/cortexode/ckpts/hcp_csrf_gnn_0/init/train/' --data_name='hcp' --surf_hemi='rh' --tag='Unet' --device='gpu'

python eval.py --test_type='init' --data_dir='/speedrun/cortexode-data-rp/valid/' --model_dir='/cortexode/ckpts/hcp_csrf_gnn_0/model/' --init_dir='/cortexode/ckpts/hcp_csrf_gnn_0/init/valid/' --data_name='hcp' --surf_hemi='rh' --tag='Unet' --device='gpu'
