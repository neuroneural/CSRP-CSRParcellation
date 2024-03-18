#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode

python eval.py --test_type='init' --data_dir='/speedrun/cortexode-data-rp/train/' --model_dir='/cortexode/ckpts/experiment_4/model/' --init_dir='/cortexode/ckpts/experiment_4/init/train/' --data_name='hcp' --surf_hemi='lh' --tag='exp4' --device='gpu'

python eval.py --test_type='init' --data_dir='/speedrun/cortexode-data-rp/valid/' --model_dir='/cortexode/ckpts/experiment_4/model/' --init_dir='/cortexode/ckpts/experiment_4/init/valid/' --data_name='hcp' --surf_hemi='lh' --tag='exp4' --device='gpu'

