#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode

# model_gm_adni_lh_exp6_300epochs.pt
# model_gm_adni_rh_exp6_300epochs.pt
# model_wm_adni_lh_exp6_270epochs.pt
# model_wm_adni_rh_exp6_270epochs.pt
#--model_type="notcsrf"
#csrf-data-dev
python train.py --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/exp_csrf_v2/model/' --init_dir='/cortexode/ckpts/exp_csrf_v2/init/' --data_name='adni'  --surf_hemi='lh' --surf_type='gm' --n_epochs=401 --n_samples=150000 --tag='csrf' --solver='rk4' --step_size=0.1 --device='gpu' 
