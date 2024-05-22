#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode
python eval.py --test_type='pred' --data_dir='/speedrun/cortexode-data-rp/test/' --model_dir='/cortexode/ckpts/experiment_5/model/' --result_dir='/cortexode/ckpts/experiment_5/result/' --data_name='adni' --surf_hemi='rh' --tag='exp5' --solver='euler' --step_size=0.1 --device='gpu'
