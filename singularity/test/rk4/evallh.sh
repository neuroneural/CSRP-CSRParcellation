#!/bin/bash
. /opt/miniconda3/bin/activate csrf
cd /cortexode
python eval.py --seg_model_type='SwinUNETR' --test_type='eval' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/exp_csrf_7/model/' --result_dir='/cortexode/ckpts/exp_csrf_7/result/' --data_name='adni' --surf_hemi='none' --tag='SwinUNETR' --solver='rk4' --step_size=0.1 --device='gpu'
