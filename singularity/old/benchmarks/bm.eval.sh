#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode
python bm.eval.py --test_type='pred' --data_dir='/speedrun/cortexode-data-rp/bm/' --model_dir='/cortexode/ckpts/experiment_3_mod/model/best/' --result_dir='/cortexode/ckpts/rp-hcptrained-testset-bm/' --data_name='hcp' --surf_hemi='lh' --tag='hcp-trained' --solver='euler' --step_size=0.1 --device='gpu'
