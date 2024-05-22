#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode
python eval.py --test_type='pred' --data_dir='/speedrun/cortexode-data-rp/test/' --model_dir='/cortexode/ckpts/experiment_6_A100_part2/model/' --result_dir='/cortexode/ckpts/experiment_6_A100_part2/result/' --data_name='adni' --surf_hemi='rh' --tag='exp6' --solver='rk4' --step_size=0.1 --device='gpu'
