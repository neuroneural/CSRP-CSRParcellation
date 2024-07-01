#!/bin/bash
. /opt/miniconda3/bin/activate csrf
cd /cortexode

# Execute Python script with parameters set above
python train.py --model_file='model_gm_hcp_lh_csrf_v0_gnnbaseline_sf0.1_320epochs_euler.pt' --patience='standard' --model_type "baseline" --gnn "baseline" --version 0 --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/hcp_csrf_gnn_0/model/' --init_dir='/cortexode/ckpts/hcp_csrf_gnn_0/init/' --data_name='hcp'  --surf_hemi='lh' --surf_type='gm' --n_epochs=401 --start_epoch=321 --n_samples=150000 --tag='csrf' --solver='euler' --step_size=0.1 --device='gpu' 
