#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode

python trainwm.py --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/experiment_3/model/' --init_dir='/cortexode/ckpts/experiment_3/init/' --data_name='hcp'  --surf_hemi='lh' --surf_type='wm' --n_epochs=120 --n_samples=150000 --tag='exp3' --solver='euler' --step_size=0.1 --device='gpu' &
PID=$!
python traingm.py --train_type='surf' --data_dir='/speedrun/cortexode-data-rp/' --model_dir='/cortexode/ckpts/experiment_3/model/' --init_dir='/cortexode/ckpts/experiment_3/init/' --data_name='hcp'  --surf_hemi='lh' --surf_type='gm' --n_epochs=40 --n_samples=150000 --tag='exp3' --solver='euler' --step_size=0.1 --device='gpu' &

wait $!
wait PID


