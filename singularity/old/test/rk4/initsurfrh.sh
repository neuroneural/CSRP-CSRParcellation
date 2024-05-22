#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode

python eval.py --test_type='init' --data_dir='/speedrun/cortexode-data-rp/test/' --model_dir='/cortexode/ckpts/experiment_5/model/' --init_dir='/cortexode/ckpts/experiment_5/init/test/' --data_name='adni' --surf_hemi='rh' --tag='exp5' --device='gpu'
