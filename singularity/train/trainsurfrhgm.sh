#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode
# model_gm_adni_lh_exp6_300epochs.pt
# model_gm_adni_rh_exp6_300epochs.pt
# model_wm_adni_lh_exp6_270epochs.pt
# model_wm_adni_rh_exp6_270epochs.pt

python train.py --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/experiment_6_A100_part2/model/' --model_file='model_gm_adni_rh_exp6_300epochs.pt' --init_dir='/cortexode/ckpts/experiment_6_A100_part2/init/' --data_name='adni'  --surf_hemi='rh' --surf_type='gm' --n_epochs=400 --n_samples=150000 --tag='exp6' --solver='rk4' --step_size=0.1 --device='gpu'
