#!/bin/bash
#. /opt/miniconda3/bin/activate cortexode
. /opt/miniconda3/bin/activate csrf
cd /cortexode

python train.py --patience='standard' --model_type "baseline" --gnn "baseline" --version 0 --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/hcp_csrf_gnn_0/model/' --init_dir='/cortexode/ckpts/hcp_csrf_gnn_0/init/' --data_name='hcp'  --surf_hemi='rh' --surf_type='wm' --n_epochs=401 --start_epoch=1 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu'
