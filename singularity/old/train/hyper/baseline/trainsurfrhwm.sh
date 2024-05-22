#!/bin/bash
#. /opt/miniconda3/bin/activate cortexode
. /opt/miniconda3/bin/activate csrf
cd /cortexode

python train.py --model_type "baseline" --gnn "baseline" --version 0 --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/exp_csrf_gnn_5/model/' --init_dir='/cortexode/ckpts/exp_csrf_gnn_5/init/' --data_name='adni'  --surf_hemi='rh' --surf_type='wm' --n_epochs=401 --start_epoch=1 --n_samples=150000 --tag='csrf' --solver='euler' --step_size=0.1 --device='gpu'
