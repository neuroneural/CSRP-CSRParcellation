#!/bin/bash
#. /opt/miniconda3/bin/activate cortexode
. /opt/miniconda3/bin/activate csrf

cd /cortexode

#python train.py  --version $version --gnn $gnn --gnn_layers $gnn_layer --gat_heads $gat_head --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/exp_csrf_gnn_1/model/' --init_dir='/cortexode/ckpts/exp_csrf_gnn_1/init/' --data_name='hcp'  --surf_hemi='lh' --surf_type='wm' --n_epochs=100 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu' 
#python train.py --model_file 'model_wm_hcp_lh_csrf_v1_gnngat_layers5_sf0.1_heads1_29epochs.pt' --version $version --gnn $gnn --gnn_layers $gnn_layer --gat_heads $gat_head --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/hcp_csrf_gnn_0/model/' --init_dir='/cortexode/ckpts/hcp_csrf_gnn_0/init/' --data_name='hcp'  --surf_hemi='lh' --surf_type='wm' --n_epochs=60 --start_epoch=31 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu' 
python train.py --patience='standard' --model_type "baseline" --gnn "baseline" --version 0 --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/hcp_csrf_gnn_0/model/' --init_dir='/cortexode/ckpts/hcp_csrf_gnn_0/init/' --data_name='hcp'  --surf_hemi='lh' --surf_type='wm' --n_epochs=401 --start_epoch=1 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu' 
