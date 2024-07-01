#!/bin/bash
#. /opt/miniconda3/bin/activate cortexode
. /opt/miniconda3/bin/activate csrf
cd /cortexode


classificationVersion='1'
deformationVersion='3'
gnn='GAT'
gnn_layer='2'
gat_head='1'

#python train.py  --version $version --gnn $gnn --gnn_layers $gnn_layer --gat_heads $gat_head --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/exp_csrf_gnn_1/model/' --init_dir='/cortexode/ckpts/exp_csrf_gnn_1/init/' --data_name='adni'  --surf_hemi='rh' --surf_type='gm' --n_epochs=100 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu' 
#python train.py --model_file 'model_gm_adni_rh_csrf_v1_gnngat_layers5_sf0.1_heads1_29epochs.pt' --version $version --gnn $gnn --gnn_layers $gnn_layer --gat_heads $gat_head --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/exp_csrf_gnn_7/model/' --init_dir='/cortexode/ckpts/exp_csrf_gnn_7/init/' --data_name='adni'  --surf_hemi='rh' --surf_type='gm' --n_epochs=60 --start_epoch=31 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu' 
python evaluateCSR_labeling.py --classificationVersion $classificationVersion --deformationVersion=$deformationVersion --gnn $gnn --gnn_layers $gnn_layer --gat_heads $gat_head --data_dir='/speedrun/cortexode-data-rp/' --deform_model_dir='/cortexode/ckpts/exp_vc_gnn_0/model/' --classification_model_dir='/cortexode/ckpts/exp_vc_gnn_0/model/' --deform_model_file='' --classify_model_file='' --result_dir='/cortexode/ckpts/exp_vc_gnn_0/result/' --data_name='hcp'  --surf_hemi='rh' --surf_type='gm' --n_samples=150000 --tagdeform='vc' --device='gpu' 
