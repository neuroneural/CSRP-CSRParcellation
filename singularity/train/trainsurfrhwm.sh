#!/bin/bash
#. /opt/miniconda3/bin/activate cortexode
. /opt/miniconda3/bin/activate csrf
cd /cortexode

# model_gm_adni_lh_exp6_300epochs.pt
# model_gm_adni_rh_exp6_300epochs.pt
# model_wm_adni_lh_exp6_270epochs.pt
# model_wm_adni_rh_exp6_270epochs.pt
#--model_type="notcsrf" 
#python train.py --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/exp_csrf_v2/model/' --init_dir='/cortexode/ckpts/exp_csrf_v2/init/' --data_name='adni'  --surf_hemi='rh' --surf_type='wm' --n_epochs=401 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu'
python train.py --train_type='surf' --data_dir='/speedrun/csrf-data-dev/' --model_dir='/cortexode/ckpts/exp_csrf_v3/model/' --init_dir='/cortexode/ckpts/exp_csrf_v3/init/' --data_name='adni'  --surf_hemi='rh' --surf_type='wm' --n_epochs=100 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu'
