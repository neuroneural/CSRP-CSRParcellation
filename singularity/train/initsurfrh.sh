#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode

python eval.py --test_type='init' --data_dir='/speedrun/cortexode-data-rp/train/' --model_dir='/cortexode/ckpts/experiment_5/model/' --init_dir='/cortexode/ckpts/experiment_5/init/train/' --data_name='adni' --surf_hemi='rh' --tag='exp5' --device='gpu'

python eval.py --test_type='init' --data_dir='/speedrun/cortexode-data-rp/valid/' --model_dir='/cortexode/ckpts/experiment_5/model/' --init_dir='/cortexode/ckpts/experiment_5/init/valid/' --data_name='adni' --surf_hemi='rh' --tag='exp5' --device='gpu'

