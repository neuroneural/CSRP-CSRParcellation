#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode
# model_gm_adni_lh_exp6_300epochs.pt
# model_gm_adni_rh_exp6_300epochs.pt
# model_wm_adni_lh_exp6_270epochs.pt
# model_wm_adni_rh_exp6_270epochs.pt

python train.py --train_type='surf' --data_dir='/speedrun/csrf-data-dev/' --model_dir='/cortexode/ckpts/exp_csrf_v1/model/' --init_dir='/cortexode/ckpts/exp_csrf_v1/init/' --data_name='adni'  --surf_hemi='rh' --surf_type='gm' --n_epochs=100 --n_samples=150000 --tag='csrf' --solver='euler' --step_size=0.1 --device='gpu'
