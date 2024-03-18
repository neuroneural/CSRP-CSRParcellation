#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode
python eval.py --test_type='pred' --data_dir='/speedrun/cortexode-data-rp/test/' --model_dir='/cortexode/ckpts/experiment_4/model/' --result_dir='/cortexode/ckpts/rp2-hcp/' --data_name='hcp' --surf_hemi='lh' --tag='exp4' --solver='euler' --step_size=0.1 --device='gpu'
